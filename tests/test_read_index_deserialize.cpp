/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include <faiss/Index.h>
#include <faiss/Index2Layer.h>
#include <faiss/IndexAdditiveQuantizerFastScan.h>
#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFIndependentQuantizer.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/hamming.h>

using namespace faiss;

/// Helper: append a scalar value to the buffer in little-endian format,
/// matching WRITE1.
template <typename T>
static void push_val(std::vector<uint8_t>& buf, T val) {
    const auto* p = reinterpret_cast<const uint8_t*>(&val);
    buf.insert(buf.end(), p, p + sizeof(T));
}

/// Helper: append a WRITEVECTOR-formatted vector (size_t length prefix
/// followed by raw element data).
template <typename T>
static void push_vector(std::vector<uint8_t>& buf, const std::vector<T>& vec) {
    push_val<size_t>(buf, vec.size());
    const auto* p = reinterpret_cast<const uint8_t*>(vec.data());
    buf.insert(buf.end(), p, p + vec.size() * sizeof(T));
}

/// Helper: append a fourcc string as a uint32_t.
static void push_fourcc(std::vector<uint8_t>& buf, const char s[4]) {
    const auto* x = reinterpret_cast<const unsigned char*>(s);
    uint32_t h = x[0] | (x[1] << 8) | (x[2] << 16) | (x[3] << 24);
    push_val<uint32_t>(buf, h);
}

/// Helper: append write_index_header fields.
static void push_index_header(
        std::vector<uint8_t>& buf,
        int d,
        int64_t ntotal,
        bool is_trained = true,
        int metric_type = 1 /* L2 */) {
    push_val<int>(buf, d);
    push_val<int64_t>(buf, ntotal);
    int64_t dummy = 1 << 20;
    push_val<int64_t>(buf, dummy);
    push_val<int64_t>(buf, dummy);
    push_val<bool>(buf, is_trained);
    push_val<int>(buf, metric_type);
}

/// Helper: append write_ProductQuantizer fields (d, M, nbits, centroids vec).
static void push_pq(
        std::vector<uint8_t>& buf,
        size_t d,
        size_t M,
        size_t nbits,
        const std::vector<float>& centroids = {}) {
    push_val<size_t>(buf, d);
    push_val<size_t>(buf, M);
    push_val<size_t>(buf, nbits);
    push_vector<float>(buf, centroids);
}

/// Try to read a VectorTransform from the given buffer and expect a
/// FaissException whose message contains the given substring.
static void expect_vt_read_throws_with(
        const std::vector<uint8_t>& data,
        const std::string& expected_substr) {
    VectorIOReader reader;
    reader.data = data;
    try {
        read_VectorTransform_up(&reader);
        FAIL() << "expected FaissException";
    } catch (const FaissException& e) {
        EXPECT_NE(
                std::string(e.what()).find(expected_substr), std::string::npos)
                << "expected '" << expected_substr << "' in: " << e.what();
    }
}

/// Try to read a float index from the given buffer and expect a FaissException.
static void expect_read_throws(const std::vector<uint8_t>& data) {
    VectorIOReader reader;
    reader.data = data;
    EXPECT_THROW(read_index_up(&reader), FaissException);
}

/// Try to read a float index and expect a FaissException whose message
/// contains the given substring.
static void expect_read_throws_with(
        const std::vector<uint8_t>& data,
        const std::string& expected_substr) {
    VectorIOReader reader;
    reader.data = data;
    try {
        read_index_up(&reader);
        FAIL() << "expected FaissException";
    } catch (const FaissException& e) {
        EXPECT_NE(
                std::string(e.what()).find(expected_substr), std::string::npos)
                << "expected '" << expected_substr << "' in: " << e.what();
    }
}

// -----------------------------------------------------------------------
// Test: ProductQuantizer with M=0 causes divide-by-zero in
// set_derived_values().  The fix validates M > 0 in read_ProductQuantizer
// before calling set_derived_values().
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, PQWithMZeroDivideByZero) {
    // Build a minimal MultiIndexQuantizer ("Imiq") payload with M=0.
    // Format: fourcc("Imiq") + index_header + PQ(d=4, M=0, nbits=8, [])
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Imiq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_pq(buf, /*d=*/4, /*M=*/0, /*nbits=*/8);

    expect_read_throws(buf);
}

// -----------------------------------------------------------------------
// Test: AdditiveQuantizer with nbits.size() != M causes out-of-bounds
// access on the nbits vector in set_derived_values().  The fix validates
// nbits.size() == M in read_AdditiveQuantizer.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AdditiveQuantizerNbitsSizeMismatch) {
    // Build a minimal IndexProductResidualQuantizerFastScan ("IPRf") payload
    // whose AdditiveQuantizer has M=10 but nbits vector has only 1 element.
    //
    // "IPRf" format:
    //   fourcc + index_header + read_AdditiveQuantizer(d, M, nbits,
    //     is_trained, codebooks, search_type, norm_min, norm_max)
    //   + set_derived_values()
    //
    // We must provide enough data to reach set_derived_values() so the
    // OOB access on nbits[i] actually triggers (rather than an earlier
    // read-error throwing first).
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IPRf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    // read_AdditiveQuantizer fields:
    push_val<size_t>(buf, 4);  // d
    push_val<size_t>(buf, 10); // M = 10
    // nbits vector with only 1 element (should be 10 to match M)
    push_vector<size_t>(buf, {8});
    // is_trained
    push_val<bool>(buf, true);
    // codebooks (empty vector is fine)
    push_vector<float>(buf, {});
    // search_type (ST_decompress = 0)
    push_val<int>(buf, 0);
    // norm_min, norm_max
    push_val<float>(buf, 0.0f);
    push_val<float>(buf, 1.0f);
    // After these reads, set_derived_values() will access nbits[1..9]
    // which are out of bounds.

    expect_read_throws_with(buf, "nbits size");
}

// -----------------------------------------------------------------------
// Test: ResidualQuantizer (old format) with nbits.size() != M also causes
// out-of-bounds access.  Uses the "IxRQ" fourcc path.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ResidualQuantizerOldNbitsSizeMismatch) {
    // "IxRQ" format: fourcc + index_header + read_ResidualQuantizer_old(...)
    // read_ResidualQuantizer_old reads: d, M, nbits_vec, is_trained, ...
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxRQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    // ResidualQuantizer_old fields:
    push_val<size_t>(buf, 4); // d
    push_val<size_t>(buf, 5); // M = 5
    // nbits vector with only 2 elements (should be 5 to match M)
    push_vector<size_t>(buf, {8, 8});

    expect_read_throws_with(buf, "nbits size");
}

// -----------------------------------------------------------------------
// Test: ProductQuantizer with d * ksub overflow in centroids allocation.
// The fix uses mul_no_overflow to detect the overflow.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, PQCentroidsOverflow) {
    // Build a minimal "Imiq" with d very large and nbits=24 so that
    // d * (1 << 24) overflows size_t.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Imiq");
    // Use a huge d that, when multiplied by ksub=2^24, overflows
    size_t huge_d = (size_t)1 << 48;
    push_index_header(buf, /*d=*/(int)huge_d, /*ntotal=*/0);
    // M must divide d; set M=1 so d % M == 0
    push_pq(buf, /*d=*/huge_d, /*M=*/1, /*nbits=*/24);

    expect_read_throws(buf);
}

// -----------------------------------------------------------------------
// Test: READVECTOR rejects a vector whose total byte size exceeds the
// configurable deserialization byte limit.  Uses a LinearTransform
// ("LTra") whose A vector is read via READVECTOR.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, READVECTORByteLimit) {
    // Build a "LTra" (LinearTransform) payload.
    // Format: fourcc + have_bias + READVECTOR(A) + READVECTOR(b)
    //         + d_in + d_out + is_trained
    // A contains 1024 floats = 4096 bytes.
    // READVECTOR check: size < limit / sizeof(float)
    const size_t old_limit = get_deserialization_vector_byte_limit();

    const int d_in = 32;
    const int d_out = 32;
    const size_t n_elements = d_in * d_out; // 1024

    std::vector<uint8_t> buf;
    push_fourcc(buf, "LTra");
    push_val<bool>(buf, false); // have_bias
    std::vector<float> A(n_elements, 0.0f);
    push_vector<float>(buf, A);
    // b vector: empty (no bias)
    push_vector<float>(buf, {});
    // Common VectorTransform fields
    push_val<int>(buf, d_in);
    push_val<int>(buf, d_out);
    push_val<bool>(buf, true); // is_trained

    // Exactly at the boundary: limit = n_elements * sizeof(float).
    // Check is strict less-than, so this should be rejected.
    set_deserialization_vector_byte_limit(n_elements * sizeof(float));
    {
        VectorIOReader reader;
        reader.data = buf;
        EXPECT_THROW(read_VectorTransform_up(&reader), FaissException);
    }

    // One element above the boundary: limit = (n_elements + 1) * sizeof(float).
    // Now n_elements < limit / sizeof(float) = n_elements + 1, so it passes.
    set_deserialization_vector_byte_limit((n_elements + 1) * sizeof(float));
    {
        VectorIOReader reader;
        reader.data = buf;
        EXPECT_NO_THROW(read_VectorTransform_up(&reader));
    }

    set_deserialization_vector_byte_limit(old_limit);
}

// -----------------------------------------------------------------------
// Test: ProductQuantizer centroids allocation is rejected when it would
// exceed the configurable deserialization byte limit.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, PQCentroidsByteLimit) {
    // d=8, M=1, nbits=8 → ksub=256, centroids = 8*256 = 2048 floats.
    // PQ check: n < limit / sizeof(float), where n = d * ksub = 2048.
    const size_t old_limit = get_deserialization_vector_byte_limit();

    const size_t d = 8;
    const size_t M = 1;
    const size_t nbits = 8;
    const size_t ksub = size_t{1} << nbits;
    const size_t n_elements = d * ksub; // 2048
    std::vector<float> centroids(n_elements, 0.0f);

    std::vector<uint8_t> buf;
    push_fourcc(buf, "Imiq");
    push_index_header(buf, /*d=*/d, /*ntotal=*/0);
    push_pq(buf, d, M, nbits, centroids);

    // Exactly at the boundary: limit = n_elements * sizeof(float).
    // Check is strict less-than, so this should be rejected.
    set_deserialization_vector_byte_limit(n_elements * sizeof(float));
    {
        VectorIOReader reader;
        reader.data = buf;
        EXPECT_THROW(read_index_up(&reader), FaissException);
    }

    // One element above the boundary: limit = (n_elements + 1) * sizeof(float).
    // Now n_elements < limit / sizeof(float) = n_elements + 1, so it passes.
    set_deserialization_vector_byte_limit((n_elements + 1) * sizeof(float));
    {
        VectorIOReader reader;
        reader.data = buf;
        EXPECT_NO_THROW(read_index_up(&reader));
    }

    set_deserialization_vector_byte_limit(old_limit);
}

// -----------------------------------------------------------------------
// Test: IndexLattice with nsq=0 causes divide-by-zero in the
// constructor's member initializer list (dsq = d / nsq).  The fix
// validates nsq > 0 in the deserialization path.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexLatticeNsqZeroDivideByZero) {
    // "IxLa" format: fourcc + d(int) + nsq(int) + scale_nbit(int)
    //                + r2(int) + index_header + READVECTOR(trained)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLa");
    push_val<int>(buf, 16); // d
    push_val<int>(buf, 0);  // nsq = 0 -> divide by zero
    push_val<int>(buf, 4);  // scale_nbit
    push_val<int>(buf, 14); // r2

    expect_read_throws(buf);
}

// -----------------------------------------------------------------------
// Test: IndexLattice with d not divisible by nsq causes undefined
// behavior in the constructor.  The fix validates d % nsq == 0 before
// construction.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexLatticeDNotDivisibleByNsq) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLa");
    push_val<int>(buf, 17); // d = 17 (not divisible by nsq=4)
    push_val<int>(buf, 4);  // nsq
    push_val<int>(buf, 4);  // scale_nbit
    push_val<int>(buf, 14); // r2

    expect_read_throws_with(buf, "divisible by nsq");
}

// -----------------------------------------------------------------------
// Test: IndexLattice with r2=0 causes heap-buffer-overflow in
// ZnSphereCodecRec constructor.  The fix validates r2 > 0.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexLatticeR2Zero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLa");
    push_val<int>(buf, 4); // d
    push_val<int>(buf, 1); // nsq
    push_val<int>(buf, 4); // scale_nbit
    push_val<int>(buf, 0); // r2 = 0 -> heap-buffer-overflow

    expect_read_throws_with(buf, "r2");
}

// -----------------------------------------------------------------------
// Test: IndexLattice with d/nsq not a power of 2 >= 2 causes
// heap-buffer-overflow in ZnSphereCodecRec constructor.  The fix
// validates that d/nsq is a power of 2 and >= 2.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexLatticeDsqNotPowerOf2) {
    // d=3, nsq=1 -> dsq=3 (not a power of 2)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLa");
    push_val<int>(buf, 3); // d
    push_val<int>(buf, 1); // nsq
    push_val<int>(buf, 4); // scale_nbit
    push_val<int>(buf, 1); // r2

    expect_read_throws_with(buf, "power of 2");
}

TEST(ReadIndexDeserialize, IndexLatticeDsqOne) {
    // d=1, nsq=1 -> dsq=1 (too small, causes cache_level=-1)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLa");
    push_val<int>(buf, 1); // d
    push_val<int>(buf, 1); // nsq
    push_val<int>(buf, 4); // scale_nbit
    push_val<int>(buf, 1); // r2

    expect_read_throws_with(buf, "power of 2");
}

// -----------------------------------------------------------------------
// Test: IndexLattice with r2 too large causes timeout in
// ZnSphereCodecRec constructor (exponential codeword count in the
// decode cache).  ZnSphereCodecRec caps the decode cache size to
// prevent this.  Use dsq=16 (log2_dim=4, cache_level=3) with r2=73
// which would take well over a minute of CPU time if not rejected.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexLatticeR2TooLarge) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLa");
    push_val<int>(buf, 16); // d
    push_val<int>(buf, 1);  // nsq
    push_val<int>(buf, 4);  // scale_nbit
    push_val<int>(buf, 73); // r2 = 73 (decode cache too large for dsq=16)

    expect_read_throws_with(buf, "decode cache");
}

// -----------------------------------------------------------------------
// Binary index helpers
// -----------------------------------------------------------------------

/// Helper: append read_index_binary_header fields.
/// Fields: int d, int code_size, idx_t ntotal, bool is_trained, int metric_type
static void push_binary_index_header(
        std::vector<uint8_t>& buf,
        int d,
        int64_t ntotal,
        bool is_trained = true,
        int metric_type = 1 /* L2 */) {
    int code_size = d / 8;
    push_val<int>(buf, d);
    push_val<int>(buf, code_size);
    push_val<int64_t>(buf, ntotal);
    push_val<bool>(buf, is_trained);
    push_val<int>(buf, metric_type);
}

/// Try to read a binary index from the given buffer and expect a
/// FaissException.
static void expect_binary_read_throws(const std::vector<uint8_t>& data) {
    VectorIOReader reader;
    reader.data = data;
    EXPECT_THROW(read_index_binary_up(&reader), FaissException);
}

/// Try to read a binary index and expect a FaissException whose message
/// contains the given substring.
static void expect_binary_read_throws_with(
        const std::vector<uint8_t>& data,
        const std::string& expected_substr) {
    VectorIOReader reader;
    reader.data = data;
    try {
        read_index_binary_up(&reader);
        FAIL() << "expected FaissException";
    } catch (const FaissException& e) {
        EXPECT_NE(
                std::string(e.what()).find(expected_substr), std::string::npos)
                << "expected '" << expected_substr << "' in: " << e.what();
    }
}

/// Helper: append empty direct_map bytes (NoMap type + empty array).
static void push_empty_direct_map(std::vector<uint8_t>& buf) {
    push_val<char>(buf, 0);   // DirectMap::NoMap
    push_val<size_t>(buf, 0); // empty array
}

/// Helper: append null inverted lists ("il00" fourcc).
static void push_null_invlists(std::vector<uint8_t>& buf) {
    push_fourcc(buf, "il00");
}

/// Helper: append a minimal IndexBinaryFlat ("IBxF") with the given
/// dimensions and ntotal=0.
static void push_minimal_binary_flat(std::vector<uint8_t>& buf, int d) {
    push_fourcc(buf, "IBxF");
    push_binary_index_header(buf, d, /*ntotal=*/0);
    push_vector<uint8_t>(buf, {}); // empty xb
}

// -----------------------------------------------------------------------
// Test: Unrecognized binary fourcc throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryUnrecognizedFourcc) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ZZZZ");

    expect_binary_read_throws(buf);
}

// -----------------------------------------------------------------------
// Test: IndexBinaryFlat with truncated input (fourcc only, no header).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryFlatTruncatedInput) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBxF");

    expect_binary_read_throws(buf);
}

// -----------------------------------------------------------------------
// Test: IndexBinaryFlat xb size mismatch (ntotal=1 but empty xb).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryFlatXbSizeMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBxF");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/1);
    push_vector<uint8_t>(buf, {}); // empty xb, but ntotal=1 expects 2 bytes

    expect_binary_read_throws_with(buf, "xb.size()");
}

// -----------------------------------------------------------------------
// Test: IndexBinaryFlat with negative dimension.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryFlatNegativeDimension) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBxF");
    // Manually push header with d=-1
    push_val<int>(buf, -1);    // d
    push_val<int>(buf, 0);     // code_size
    push_val<int64_t>(buf, 0); // ntotal
    push_val<bool>(buf, true); // is_trained
    push_val<int>(buf, 1);     // metric_type

    expect_binary_read_throws_with(buf, "dimension");
}

// -----------------------------------------------------------------------
// Test: IndexBinaryMultiHash with storage ntotal mismatch exercises the
// Leak 5 fix (dynamic_cast + assertion with unique_ptr guard).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryMultiHashStorageCastFailure) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHm");
    // Outer index header: ntotal=99
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/99);
    // Nested IBxF storage with ntotal=0 (mismatch with outer ntotal=99)
    push_minimal_binary_flat(buf, /*d=*/16);

    expect_binary_read_throws(buf);
}

// -----------------------------------------------------------------------
// Test: IndexBinaryIVF with inverted list nlist mismatch exercises the
// Leak 2 fix.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryIVFInvListNlistMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBwF");
    // Binary IVF header: binary_header + nlist + nprobe + quantizer +
    //                     direct_map
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<size_t>(buf, 10); // nlist = 10
    push_val<size_t>(buf, 1);  // nprobe
    // Nested quantizer (IBxF)
    push_minimal_binary_flat(buf, /*d=*/16);
    // Empty direct map
    push_empty_direct_map(buf);
    // ArrayInvertedLists with nlist=5 (mismatch with IVF nlist=10).
    // "ilar" fourcc + nlist + code_size + sizes
    push_fourcc(buf, "ilar");
    push_val<size_t>(buf, 5); // nlist = 5 (mismatches IVF nlist=10)
    push_val<size_t>(buf, 2); // code_size = d/8 = 2
    // 5 list sizes, all zero
    for (int i = 0; i < 5; i++) {
        push_val<size_t>(buf, 0);
    }

    expect_binary_read_throws(buf);
}

// -----------------------------------------------------------------------
// Test: IndexBinaryHash with empty hash invlist buffer but non-zero entry
// count. Exercises the null-deref fix in read_binary_hash_invlists.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryHashEmptyInvlistBuffer) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHh");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<int>(buf, 4);            // b
    push_val<int>(buf, 0);            // nflip
    push_val<size_t>(buf, size_t(1)); // sz = 1 (non-zero)
    push_val<int>(buf, 8);            // il_nbit
    push_vector<uint8_t>(buf, {});    // empty buffer (should fail)

    expect_binary_read_throws_with(buf, "binary hash invlists");
}

// -----------------------------------------------------------------------
// Test: NSG with R=0 triggers the R > 0 validation.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, NSGNegativeR) {
    // "INSf" format: fourcc + index_header + GK + build_type +
    //   nndescent_S/R/L/iter + read_NSG(ntotal, R, ...)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "INSf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<int>(buf, 0);  // GK
    push_val<int>(buf, 0);  // build_type
    push_val<int>(buf, 10); // nndescent_S
    push_val<int>(buf, 10); // nndescent_R
    push_val<int>(buf, 10); // nndescent_L
    push_val<int>(buf, 1);  // nndescent_iter
    // read_NSG fields:
    push_val<int>(buf, 0);  // ntotal
    push_val<int>(buf, -1); // R = -1 (invalid)

    expect_read_throws_with(buf, "invalid NSG R");
}

// -----------------------------------------------------------------------
// Test: ScalarQuantizer with out-of-range qtype throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ScalarQuantizerInvalidQtype) {
    // "IxSQ" format: fourcc + index_header + read_ScalarQuantizer(qtype, ...)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxSQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    // ScalarQuantizer fields:
    push_val<int>(buf, 99); // qtype = 99 (out of range)

    expect_read_throws_with(buf, "qtype");
}

// -----------------------------------------------------------------------
// Test: ProductAdditiveQuantizer with nsplits=0 throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ProductAdditiveQuantizerZeroNsplits) {
    // "IxPR" format: fourcc + index_header +
    //   read_ProductResidualQuantizer(read_ProductAdditiveQuantizer(
    //     read_AdditiveQuantizer(...) + nsplits))
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPR");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    // AdditiveQuantizer fields:
    push_val<size_t>(buf, 4);      // d
    push_val<size_t>(buf, 1);      // M
    push_vector<size_t>(buf, {8}); // nbits (1 element matching M=1)
    push_val<bool>(buf, true);     // is_trained
    // codebooks: d * total_codebook_size = 4 * 256 = 1024 floats
    push_vector<float>(buf, std::vector<float>(4 * 256, 0.0f));
    push_val<int>(buf, 0);      // search_type = ST_decompress
    push_val<float>(buf, 0.0f); // norm_min
    push_val<float>(buf, 1.0f); // norm_max
    // ProductAdditiveQuantizer field:
    push_val<size_t>(buf, 0); // nsplits = 0 (invalid)

    expect_read_throws_with(buf, "nsplits");
}

// -----------------------------------------------------------------------
// Test: PreTransform with negative chain length throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, PreTransformNegativeChainLength) {
    // "IxPT" format: fourcc + index_header + nt + VT chain + nested index
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPT");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<int>(buf, -1); // nt = -1 (invalid)

    expect_read_throws_with(buf, "chain length");
}

// -----------------------------------------------------------------------
// Test: IndexBinaryMultiHash with nhash=0 throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryMultiHashZeroNhash) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHm");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    // Nested IBxF storage (ntotal=0 matches outer)
    push_minimal_binary_flat(buf, /*d=*/16);
    push_val<int>(buf, 4); // b
    push_val<int>(buf, 0); // nhash = 0 (invalid)

    expect_binary_read_throws_with(buf, "nhash");
}

// -----------------------------------------------------------------------
// Test: IndexBinaryHash with b=0 triggers the b > 0 validation.
// Without this check, BitstringReader::read(0) would silently produce
// garbage hash values on every inverted-list entry.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryHashBZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHh");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<int>(buf, 0); // b = 0 (invalid)

    expect_binary_read_throws_with(buf, "IndexBinaryHash b=");
}

// -----------------------------------------------------------------------
// Test: read_binary_hash_invlists with negative il_nbit triggers the
// il_nbit >= 0 validation.  Without this check, the negative value would
// wrap to a huge size_t in the bits-per-entry calculation, causing an
// out-of-bounds read in BitstringReader.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryHashNegativeIlNbit) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHh");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<int>(buf, 4); // b
    push_val<int>(buf, 0); // nflip
    // read_binary_hash_invlists fields:
    push_val<size_t>(buf, size_t(0)); // sz = 0
    push_val<int>(buf, -1);           // il_nbit = -1 (invalid)

    expect_binary_read_throws_with(buf, "il_nbit=");
}

// -----------------------------------------------------------------------
// Test: read_binary_hash_invlists with il_nbit=0 but sz > 0 triggers
// the il_nbit > 0 validation.  Without this check, every inverted-list
// size would silently read as 0, corrupting the deserialized index.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryHashIlNbitZeroWithEntries) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHh");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<int>(buf, 4); // b
    push_val<int>(buf, 0); // nflip
    // read_binary_hash_invlists fields:
    push_val<size_t>(buf, size_t(1)); // sz = 1 (non-zero)
    push_val<int>(buf, 0);            // il_nbit = 0 (invalid when sz > 0)

    expect_binary_read_throws_with(buf, "il_nbit=");
}

// -----------------------------------------------------------------------
// Test: IndexBinaryHash with b exceeding code_size*8 triggers validation.
// Without this check, BitstringReader::read() would access past the
// allocated code buffer, causing a heap-buffer-overflow.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryHashBExceedsCodeSize) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHh");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    // d=16 → code_size=2 → 16 bits available
    push_val<int>(buf, 17); // b = 17 (exceeds 16 bits)

    expect_binary_read_throws_with(buf, "IndexBinaryHash b=");
}

// -----------------------------------------------------------------------
// Test: read_binary_multi_hash_map with crafted ilsz values that sum to
// more than ntotal.  Without the check, the inner loop would read past
// the end of the BitstringReader buffer.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryMultiHashMapIlszExceedsNtotal) {
    // ntotal=4 → id_bits=2 (writer: while (4 > (1<<id_bits)) id_bits++)
    // b=4, sz=2 entries
    // Total bits formula: (b + id_bits) * sz + ntotal * id_bits
    //                   = (4 + 2) * 2 + 4 * 2 = 20
    // buf size = (20 + 7) / 8 = 3 bytes
    //
    // Craft entry 0: hash=0, ilsz=3  (3 ids follow)
    // Craft entry 1: hash=1, ilsz=3  (3 ids follow)
    // Total ilsz = 6, but ntotal = 4 → should throw on entry 1.

    const int b = 4;
    const int id_bits = 2;
    const size_t ntotal = 4;
    const size_t sz = 2;
    const size_t nbit = (b + id_bits) * sz + ntotal * id_bits;
    const size_t bitbuf_size = (nbit + 7) / 8;

    // Pack the bitstring with BitstringWriter.
    std::vector<uint8_t> bitbuf(bitbuf_size, 0);
    BitstringWriter wr(bitbuf.data(), bitbuf.size());
    // Entry 0: hash=0, ilsz=3, ids={0,0,0}
    wr.write(0, b);
    wr.write(3, id_bits);
    wr.write(0, id_bits);
    wr.write(0, id_bits);
    wr.write(0, id_bits);
    // Entry 1: hash=1, ilsz=3
    // (reader should throw before reading ids for this entry)
    wr.write(1, b);
    wr.write(3, id_bits);

    // Build the full IBHm serialized buffer.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHm");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/ntotal);

    // Nested IBxF storage with matching ntotal.
    // xb needs ntotal * code_size bytes (code_size = d/8 = 2).
    push_fourcc(buf, "IBxF");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/ntotal);
    std::vector<uint8_t> xb(ntotal * 2, 0);
    push_vector<uint8_t>(buf, xb);

    push_val<int>(buf, b); // b
    push_val<int>(buf, 1); // nhash = 1
    push_val<int>(buf, 0); // nflip

    // Multi hash map fields (1 map):
    push_val<int>(buf, id_bits);       // id_bits
    push_val<size_t>(buf, sz);         // sz = 2 entries
    push_vector<uint8_t>(buf, bitbuf); // packed bitstring

    expect_binary_read_throws_with(buf, "would exceed ntotal");
}

// -----------------------------------------------------------------------
// Test: IndexBinaryIDMap ("IBMp") with id_map size != ntotal.  Without
// the check, construct_rev_map (IBM2) or subsequent search operations
// would use an inconsistent ID mapping.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BinaryIDMapIdMapSizeMismatch) {
    // Outer IBMp header has ntotal=2, but the id_map vector has 5
    // entries.  The check on id_map.size() == ntotal must reject this.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBMp");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/2);

    // Nested IBxF storage with ntotal=2 (matches outer header).
    push_fourcc(buf, "IBxF");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/2);
    std::vector<uint8_t> xb(2 * 2, 0); // ntotal * code_size
    push_vector<uint8_t>(buf, xb);

    // id_map with 5 entries (mismatches ntotal=2).
    std::vector<int64_t> id_map = {10, 20, 30, 40, 50};
    push_vector<int64_t>(buf, id_map);

    expect_binary_read_throws_with(buf, "id_map");
}

// -----------------------------------------------------------------------
// Test: IndexIDMap2 ("IxM2") with id_map size != ntotal.  Without the
// check, construct_rev_map() reads past id_map bounds causing a crash.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexIDMap2IdMapSizeMismatch) {
    // Outer IxM2 header has ntotal=2, but the id_map vector has 5
    // entries.  The check on id_map.size() == ntotal must reject this.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxM2");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);

    // Nested IndexFlatL2 ("IxF2") with ntotal=2.
    push_fourcc(buf, "IxF2");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);
    // WRITEXBVECTOR: ntotal * d floats
    size_t num_floats = 2 * 4;
    push_val<size_t>(buf, num_floats);
    for (size_t i = 0; i < num_floats; ++i) {
        push_val<float>(buf, 0.0f);
    }

    // id_map with 5 entries (mismatches ntotal=2).
    std::vector<int64_t> id_map = {10, 20, 30, 40, 50};
    push_vector<int64_t>(buf, id_map);

    expect_read_throws_with(buf, "id_map");
}

// -----------------------------------------------------------------------
// InvertedLists helpers
// -----------------------------------------------------------------------

/// Try to read an InvertedLists from the given buffer and expect a
/// FaissException whose message contains the given substring.
static void expect_invlists_read_throws_with(
        const std::vector<uint8_t>& data,
        const std::string& expected_substr) {
    VectorIOReader reader;
    reader.data = data;
    try {
        read_InvertedLists_up(&reader);
        FAIL() << "expected FaissException";
    } catch (const FaissException& e) {
        EXPECT_NE(
                std::string(e.what()).find(expected_substr), std::string::npos)
                << "expected '" << expected_substr << "' in: " << e.what();
    }
}

// -----------------------------------------------------------------------
// Test: BlockInvertedLists with n_per_block=0 causes divide-by-zero
// in resize() and add_entries().  The fix validates n_per_block > 0
// during deserialization.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BlockInvertedListsNPerBlockZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ilbl");
    push_val<size_t>(buf, 1);   // nlist
    push_val<size_t>(buf, 32);  // code_size
    push_val<size_t>(buf, 0);   // n_per_block = 0 (invalid)
    push_val<size_t>(buf, 128); // block_size

    expect_invlists_read_throws_with(buf, "n_per_block");
}

// -----------------------------------------------------------------------
// Test: BlockInvertedLists with block_size=0 causes divide-by-zero
// in add_entries().  The fix validates block_size > 0 during
// deserialization.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BlockInvertedListsBlockSizeZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ilbl");
    push_val<size_t>(buf, 1);  // nlist
    push_val<size_t>(buf, 32); // code_size
    push_val<size_t>(buf, 32); // n_per_block
    push_val<size_t>(buf, 0);  // block_size = 0 (invalid)

    expect_invlists_read_throws_with(buf, "block_size");
}

// -----------------------------------------------------------------------
// Test: BlockInvertedLists with codes vector size inconsistent with ids
// size, n_per_block, and block_size.  Without the check, a search or
// add operation would read/write past the end of the codes buffer.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BlockInvertedListsCodesSizeMismatch) {
    const size_t n_per_block = 32;
    const size_t block_size = 128;

    std::vector<uint8_t> buf;
    push_fourcc(buf, "ilbl");
    push_val<size_t>(buf, 1);           // nlist = 1
    push_val<size_t>(buf, 32);          // code_size
    push_val<size_t>(buf, n_per_block); // n_per_block
    push_val<size_t>(buf, block_size);  // block_size

    // ids: 10 entries → ceil(10/32) = 1 block → expected codes = 128 bytes
    std::vector<int64_t> ids(10, 0);
    push_vector<int64_t>(buf, ids);
    // codes: 64 bytes (wrong, should be 128)
    std::vector<uint8_t> codes(64, 0);
    push_vector<uint8_t>(buf, codes);

    expect_invlists_read_throws_with(buf, "codes size");
}

// -----------------------------------------------------------------------
// Test: BlockInvertedLists with n_block * block_size overflow.  Without
// the mul_no_overflow check, the multiplication wraps around and the
// size comparison passes spuriously.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BlockInvertedListsCodesOverflow) {
    // Choose n_per_block=1 and block_size near SIZE_MAX so that
    // n_block * block_size overflows.
    const size_t n_per_block = 1;
    const size_t block_size = (size_t)-1; // SIZE_MAX

    std::vector<uint8_t> buf;
    push_fourcc(buf, "ilbl");
    push_val<size_t>(buf, 1);           // nlist = 1
    push_val<size_t>(buf, 32);          // code_size
    push_val<size_t>(buf, n_per_block); // n_per_block
    push_val<size_t>(buf, block_size);  // block_size

    // ids: 2 entries → n_block = 2 → 2 * SIZE_MAX overflows
    std::vector<int64_t> ids(2, 0);
    push_vector<int64_t>(buf, ids);
    // codes: any size, doesn't matter — overflow should be caught first
    std::vector<uint8_t> codes(0);
    push_vector<uint8_t>(buf, codes);

    expect_invlists_read_throws_with(buf, "overflow");
}

// -----------------------------------------------------------------------
// Test: BlockInvertedLists with valid ids and codes passes validation.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, BlockInvertedListsValidCodesSize) {
    const size_t n_per_block = 32;
    const size_t block_size = 128;
    const size_t nlist = 2;

    std::vector<uint8_t> buf;
    push_fourcc(buf, "ilbl");
    push_val<size_t>(buf, nlist);       // nlist
    push_val<size_t>(buf, 32);          // code_size
    push_val<size_t>(buf, n_per_block); // n_per_block
    push_val<size_t>(buf, block_size);  // block_size

    // List 0: 10 ids → ceil(10/32)=1 block → 128 bytes
    std::vector<int64_t> ids0(10, 0);
    push_vector<int64_t>(buf, ids0);
    std::vector<uint8_t> codes0(128, 0);
    push_vector<uint8_t>(buf, codes0);

    // List 1: 0 ids → 0 blocks → 0 bytes
    std::vector<int64_t> ids1;
    push_vector<int64_t>(buf, ids1);
    std::vector<uint8_t> codes1;
    push_vector<uint8_t>(buf, codes1);

    VectorIOReader reader;
    reader.data = buf;
    auto il = read_InvertedLists_up(&reader);
    EXPECT_NE(il, nullptr);
}

// -----------------------------------------------------------------------
// Test: ProductQuantizer centroids size mismatch.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, PQCentroidsSizeMismatch) {
    // "Imiq" (MultiIndexQuantizer): fourcc + index_header + PQ
    // PQ: d=4, M=2, nbits=8 -> ksub=256, expected centroids = 4*256 = 1024
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Imiq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    std::vector<float> bad_centroids(100, 0.0f);
    push_pq(buf, /*d=*/4, /*M=*/2, /*nbits=*/8, bad_centroids);

    expect_read_throws_with(buf, "centroids size");
}

// -----------------------------------------------------------------------
// Test: ScalarQuantizer trained vector size mismatch.
// For QT_4bit (qtype=1), expected trained.size() = 2 * d.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, SQTrainedSizeMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxSQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    // ScalarQuantizer fields:
    push_val<int>(
            buf, ScalarQuantizer::QT_4bit); // expects trained.size()=2*d=8
    push_val<int>(buf, 0);                  // rangestat
    push_val<float>(buf, 0.0f);             // rangestat_arg
    push_val<size_t>(buf, 4);               // d
    push_val<size_t>(buf, 1);               // code_size
    // trained: 3 floats instead of expected 8
    push_vector<float>(buf, {1.0f, 2.0f, 3.0f});

    expect_read_throws_with(buf, "ScalarQuantizer trained size");
}

// -----------------------------------------------------------------------
// Test: IndexPQ codes vector size mismatch.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexPQCodesSizeMismatch) {
    // "IxPq": fourcc + index_header + PQ + codes + search_type +
    //         encode_signs + polysemous_ht
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);
    // PQ: d=4, M=2, nbits=8 -> code_size=2, ksub=256
    std::vector<float> centroids(4 * 256, 0.0f);
    push_pq(buf, /*d=*/4, /*M=*/2, /*nbits=*/8, centroids);
    // codes: should be ntotal * code_size = 2 * 2 = 4 bytes
    // Provide wrong size (10 bytes)
    push_vector<uint8_t>(buf, std::vector<uint8_t>(10, 0));
    // IxPq format extras:
    push_val<int>(buf, 0); // search_type
    push_val<int>(buf, 0); // encode_signs
    push_val<int>(buf, 0); // polysemous_ht

    expect_read_throws_with(buf, "codes.size()");
}

/// Helper: append a minimal AdditiveQuantizer (d, M, nbits, is_trained,
/// codebooks, search_type=ST_decompress, norm_min, norm_max).
/// Writes zero-filled codebooks of size codebook_d * total_codebook_size so
/// that the deserialization codebooks-size validation passes.
/// For standalone quantizers codebook_d == d.  For ProductAdditiveQuantizer
/// codebook_d == d / nsplits.  Pass codebook_d = 0 (default) to use d.
static void push_additive_quantizer(
        std::vector<uint8_t>& buf,
        size_t d,
        size_t M,
        const std::vector<size_t>& nbits,
        size_t codebook_d = 0) {
    if (codebook_d == 0) {
        codebook_d = d;
    }

    // Compute total_codebook_size = sum(2^nbits[i]).
    size_t total_codebook_size = 0;
    for (auto nb : nbits) {
        total_codebook_size += size_t{1} << nb;
    }

    push_val<size_t>(buf, d);
    push_val<size_t>(buf, M);
    push_vector<size_t>(buf, nbits);
    push_val<bool>(buf, true); // is_trained
    push_vector<float>(
            buf,
            std::vector<float>(
                    codebook_d * total_codebook_size, 0.0f)); // codebooks
    push_val<int>(buf, 0);      // search_type = ST_decompress
    push_val<float>(buf, 0.0f); // norm_min
    push_val<float>(buf, 1.0f); // norm_max
}

/// Helper: append a minimal ResidualQuantizer (AdditiveQuantizer +
/// train_type + max_beam_size).  Uses Skip_codebook_tables to avoid
/// compute_codebook_tables on empty codebooks.
static void push_residual_quantizer(
        std::vector<uint8_t>& buf,
        size_t d,
        size_t M,
        const std::vector<size_t>& nbits) {
    push_additive_quantizer(buf, d, M, nbits);
    push_val<int>(buf, 2048); // train_type = Skip_codebook_tables
    push_val<int>(buf, 1);    // max_beam_size
}

/// Helper: append a minimal LocalSearchQuantizer (AdditiveQuantizer +
/// K + train_iters + encode_ils_iters + train_ils_iters + icm_iters +
/// p + lambd + chunk_size + random_seed + nperts +
/// update_codebooks_with_double).
static void push_local_search_quantizer(
        std::vector<uint8_t>& buf,
        size_t d,
        size_t M,
        const std::vector<size_t>& nbits) {
    push_additive_quantizer(buf, d, M, nbits);
    push_val<size_t>(buf, 256); // K
    push_val<size_t>(buf, 25);  // train_iters
    push_val<size_t>(buf, 8);   // encode_ils_iters
    push_val<size_t>(buf, 8);   // train_ils_iters
    push_val<size_t>(buf, 4);   // icm_iters
    push_val<float>(buf, 0.5f); // p
    push_val<float>(buf, 0.0f); // lambd
    push_val<size_t>(buf, 0);   // chunk_size
    push_val<int>(buf, 123);    // random_seed
    push_val<size_t>(buf, 4);   // nperts
    push_val<bool>(buf, false); // update_codebooks_with_double
}

// -----------------------------------------------------------------------
// Test: IndexResidualQuantizer codes vector size mismatch.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexResidualQuantizerCodesSizeMismatch) {
    // "IxRq": fourcc + index_header + ResidualQuantizer + code_size + codes
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxRq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);
    push_residual_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{8});
    // code_size = 1 (M=1, nbits=8 → 1 byte per code)
    push_val<size_t>(buf, 1); // code_size
    // codes: should be ntotal * code_size = 2 * 1 = 2 bytes
    // Provide wrong size (10 bytes)
    push_vector<uint8_t>(buf, std::vector<uint8_t>(10, 0));

    expect_read_throws_with(buf, "codes.size()");
}

// -----------------------------------------------------------------------
// Test: IndexLocalSearchQuantizer codes vector size mismatch.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexLocalSearchQuantizerCodesSizeMismatch) {
    // "IxLS": fourcc + index_header + LocalSearchQuantizer + code_size + codes
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLS");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);
    push_local_search_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{8});
    // code_size = 1 (M=1, nbits=8 → 1 byte per code)
    push_val<size_t>(buf, 1); // code_size
    // codes: should be ntotal * code_size = 2 * 1 = 2 bytes
    // Provide wrong size (10 bytes)
    push_vector<uint8_t>(buf, std::vector<uint8_t>(10, 0));

    expect_read_throws_with(buf, "codes.size()");
}

// -----------------------------------------------------------------------
// Test: IndexProductResidualQuantizer codes vector size mismatch.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexProductResidualQuantizerCodesSizeMismatch) {
    // "IxPR": fourcc + index_header + ProductResidualQuantizer + code_size
    //         + codes
    // ProductResidualQuantizer = ProductAdditiveQuantizer(AdditiveQuantizer
    //   + nsplits) + nsplits * ResidualQuantizer
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPR");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);
    // ProductAdditiveQuantizer: AdditiveQuantizer + nsplits
    push_additive_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{8});
    push_val<size_t>(buf, 1); // nsplits = 1
    // 1 nested ResidualQuantizer (sub-dimension = d / nsplits = 4)
    push_residual_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{8});
    // code_size = 1
    push_val<size_t>(buf, 1); // code_size
    // codes: should be ntotal * code_size = 2 * 1 = 2 bytes
    // Provide wrong size (10 bytes)
    push_vector<uint8_t>(buf, std::vector<uint8_t>(10, 0));

    expect_read_throws_with(buf, "codes.size()");
}

// -----------------------------------------------------------------------
// Test: IndexProductLocalSearchQuantizer codes vector size mismatch.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexProductLocalSearchQuantizerCodesSizeMismatch) {
    // "IxPL": fourcc + index_header + ProductLocalSearchQuantizer + code_size
    //         + codes
    // ProductLocalSearchQuantizer = ProductAdditiveQuantizer(AdditiveQuantizer
    //   + nsplits) + nsplits * LocalSearchQuantizer
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPL");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);
    // ProductAdditiveQuantizer: AdditiveQuantizer + nsplits
    push_additive_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{8});
    push_val<size_t>(buf, 1); // nsplits = 1
    // 1 nested LocalSearchQuantizer (sub-dimension = d / nsplits = 4)
    push_local_search_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{8});
    // code_size = 1
    push_val<size_t>(buf, 1); // code_size
    // codes: should be ntotal * code_size = 2 * 1 = 2 bytes
    // Provide wrong size (10 bytes)
    push_vector<uint8_t>(buf, std::vector<uint8_t>(10, 0));

    expect_read_throws_with(buf, "codes.size()");
}

// -----------------------------------------------------------------------
// Graph index helpers
// -----------------------------------------------------------------------

/// Helper: append a minimal IndexFlatL2 ("IxF2") with the given d and ntotal.
/// Uses the WRITEXBVECTOR format (size_t num_floats prefix, then raw data).
static void push_minimal_flat(
        std::vector<uint8_t>& buf,
        int d,
        int64_t ntotal = 0) {
    push_fourcc(buf, "IxF2");
    push_index_header(buf, d, ntotal);
    size_t num_floats = (size_t)ntotal * (size_t)d;
    push_val<size_t>(buf, num_floats);
    buf.resize(buf.size() + num_floats * sizeof(float), 0);
}

/// Helper: append a minimal valid HNSW structure with the given number of
/// nodes.  All nodes are at level 1 with zero neighbors, which passes
/// validate_HNSW.
static void push_minimal_hnsw(std::vector<uint8_t>& buf, int ntotal) {
    // assign_probas (empty)
    push_vector<double>(buf, {});
    // cum_nneighbor_per_level: {0, 0} — 0 cumulative neighbors at each level
    push_vector<int>(buf, {0, 0});
    // levels: one entry per node, all at level 1 (1-indexed in HNSW)
    std::vector<int> levels(ntotal, 1);
    push_vector<int>(buf, levels);
    // offsets: ntotal + 1 entries, all 0
    std::vector<size_t> offsets(ntotal + 1, 0);
    push_vector<size_t>(buf, offsets);
    // neighbors (empty)
    push_vector<int32_t>(buf, {});
    // entry_point
    push_val<int32_t>(buf, ntotal > 0 ? 0 : -1);
    // max_level
    push_val<int>(buf, 0);
    // efConstruction
    push_val<int>(buf, 40);
    // efSearch
    push_val<int>(buf, 16);
    // upper_beam (deprecated, always 1)
    push_val<int>(buf, 1);
}

/// Helper: append NNDescent fields.
static void push_nndescent(
        std::vector<uint8_t>& buf,
        int ntotal,
        int d,
        int K,
        bool has_built,
        const std::vector<int>& final_graph) {
    push_val<int>(buf, ntotal);
    push_val<int>(buf, d);
    push_val<int>(buf, K);
    push_val<int>(buf, 10); // S
    push_val<int>(buf, 10); // R
    push_val<int>(buf, 10); // L
    push_val<int>(buf, 1);  // iter
    push_val<int>(buf, 10); // search_L
    push_val<int>(buf, 42); // random_seed
    push_val<bool>(buf, has_built);
    push_vector<int>(buf, final_graph);
}

// -----------------------------------------------------------------------
// Test: NNDescent final_graph size mismatch (should be ntotal * K).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, NNDescentGraphSizeMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "INNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);
    // NNDescent: ntotal=2, K=3, has_built=true
    // Expected graph size = 2*3 = 6, but provide 10
    push_nndescent(
            buf,
            /*ntotal=*/2,
            /*d=*/4,
            /*K=*/3,
            /*has_built=*/true,
            std::vector<int>(10, 0));

    expect_read_throws_with(buf, "NNDescent final_graph size");
}

// -----------------------------------------------------------------------
// Test: NNDescent final_graph contains out-of-range neighbor ID.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, NNDescentGraphInvalidId) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "INNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/2);
    // NNDescent: ntotal=2, K=2, graph size = 4
    // ID 99 is out of range [-1, 2)
    push_nndescent(
            buf,
            /*ntotal=*/2,
            /*d=*/4,
            /*K=*/2,
            /*has_built=*/true,
            {0, 1, 99, 0});

    expect_read_throws_with(buf, "out of range");
}

// -----------------------------------------------------------------------
// Test: HNSW levels.size() != index ntotal.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSWLevelsSizeMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IHNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/5);
    // HNSW with 3 entries (mismatch with ntotal=5)
    push_minimal_hnsw(buf, /*ntotal=*/3);

    expect_read_throws_with(buf, "HNSW levels size");
}

// -----------------------------------------------------------------------
// Test: HNSW storage ntotal != index ntotal.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSWStorageNtotalMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IHNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/3);
    // Valid HNSW with 3 entries (matches header)
    push_minimal_hnsw(buf, /*ntotal=*/3);
    // Storage with ntotal=99 (mismatch)
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/99);

    expect_read_throws_with(buf, "HNSW storage ntotal");
}

// -----------------------------------------------------------------------
// Helper: append an HNSW structure with explicit neighbor data.
// All nodes at level 1.  cum_nneighbor_per_level = {0, neighbors_per_node}.
// Callers supply the full neighbors vector (ntotal * neighbors_per_node
// entries, using -1 for empty slots).
// -----------------------------------------------------------------------
static void push_hnsw_with_neighbors(
        std::vector<uint8_t>& buf,
        int ntotal,
        int neighbors_per_node,
        const std::vector<int32_t>& neighbors,
        int32_t entry_point = 0) {
    // assign_probas (empty)
    push_vector<double>(buf, {});
    // cum_nneighbor_per_level: {0, neighbors_per_node}
    push_vector<int>(buf, {0, neighbors_per_node});
    // levels: one entry per node, all at level 1
    std::vector<int> levels(ntotal, 1);
    push_vector<int>(buf, levels);
    // offsets: ntotal + 1 entries, each node occupies neighbors_per_node slots
    std::vector<size_t> offsets(ntotal + 1);
    for (int i = 0; i <= ntotal; i++) {
        offsets[i] = (size_t)i * neighbors_per_node;
    }
    push_vector<size_t>(buf, offsets);
    // neighbors
    push_vector<int32_t>(buf, neighbors);
    // entry_point
    push_val<int32_t>(buf, entry_point);
    // max_level
    push_val<int>(buf, 0);
    // efConstruction
    push_val<int>(buf, 40);
    // efSearch
    push_val<int>(buf, 16);
    // upper_beam (deprecated)
    push_val<int>(buf, 1);
}

// -----------------------------------------------------------------------
// Test: HNSW neighbors contain a negative ID (not -1).
// validate_HNSW must reject this at deserialization time.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSWNeighborNegativeId) {
    int ntotal = 3, npn = 2;
    // Node 0 neighbors: {1, -5}  — -5 is invalid (only -1 is allowed)
    // Node 1 neighbors: {0, -1}
    // Node 2 neighbors: {0, -1}
    std::vector<int32_t> neighbors = {1, -5, 0, -1, 0, -1};

    std::vector<uint8_t> buf;
    push_fourcc(buf, "IHNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/ntotal);
    push_hnsw_with_neighbors(buf, ntotal, npn, neighbors);
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/ntotal);

    expect_read_throws_with(buf, "HNSW neighbors");
}

// -----------------------------------------------------------------------
// Test: HNSW neighbors contain an ID >= ntotal (out of range).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSWNeighborIdTooLarge) {
    int ntotal = 3, npn = 2;
    // Node 0 neighbors: {1, 99}  — 99 >= ntotal
    std::vector<int32_t> neighbors = {1, 99, 0, -1, 0, -1};

    std::vector<uint8_t> buf;
    push_fourcc(buf, "IHNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/ntotal);
    push_hnsw_with_neighbors(buf, ntotal, npn, neighbors);
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/ntotal);

    expect_read_throws_with(buf, "HNSW neighbors");
}

// -----------------------------------------------------------------------
// Test: HNSW entry_point is out of range (>= ntotal).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSWEntryPointOutOfRange) {
    int ntotal = 3, npn = 2;
    std::vector<int32_t> neighbors = {1, -1, 0, -1, 0, -1};

    std::vector<uint8_t> buf;
    push_fourcc(buf, "IHNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/ntotal);
    push_hnsw_with_neighbors(buf, ntotal, npn, neighbors, /*entry_point=*/99);
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/ntotal);

    expect_read_throws_with(buf, "HNSW entry_point");
}

// -----------------------------------------------------------------------
// Test: HNSW with valid neighbor data deserializes and searches correctly.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSWValidNeighborsSearchWorks) {
    int ntotal = 4, npn = 2;
    // Simple valid graph: each node links to 2 others
    std::vector<int32_t> neighbors = {1, 2, 0, 3, 0, 3, 1, 2};

    std::vector<uint8_t> buf;
    push_fourcc(buf, "IHNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/ntotal);
    push_hnsw_with_neighbors(buf, ntotal, npn, neighbors);

    // Flat storage with 4 zero vectors (valid, just not useful for recall)
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/ntotal);

    VectorIOReader reader;
    reader.data = buf;
    std::unique_ptr<Index> idx;
    ASSERT_NO_THROW(idx = read_index_up(&reader));
    ASSERT_NE(idx, nullptr);

    // Search should succeed without crashing
    std::vector<float> xq(4, 0.0f);
    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);
    EXPECT_NO_THROW(
            idx->search(1, xq.data(), 1, distances.data(), labels.data()));
    EXPECT_GE(labels[0], 0);
    EXPECT_LT(labels[0], ntotal);
}

// -----------------------------------------------------------------------
// Test: IndexHNSW2Level with wrong storage type is rejected.
// Protects against corrupt serialized data where storage is not
// Index2Layer or IndexIVFPQ, causing null-deref from failed dynamic_cast.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSW2LevelWrongStorageType) {
    // Build an IHN2 with IndexFlat storage (wrong type — must be
    // Index2Layer or IndexIVFPQ).
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IHN2");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_minimal_hnsw(buf, /*ntotal=*/0);
    // IndexFlat storage — wrong type for HNSW2Level
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/0);

    expect_read_throws_with(buf, "Index2Layer or IndexIVFPQ");
}

// -----------------------------------------------------------------------
// Test: Index2Layer deserialization sets own_fields=true so the
// sub-quantizer is freed when the deserialized index is destroyed.
// -----------------------------------------------------------------------

struct CountedDestructorIndex : IndexFlatL2 {
    static int destructor_count;
    using IndexFlatL2::IndexFlatL2;
    ~CountedDestructorIndex() override {
        destructor_count++;
    }
};
int CountedDestructorIndex::destructor_count = 0;

TEST(ReadIndexDeserialize, Index2LayerQuantizerOwnership) {
    int d = 4, nb = 512, nlist = 2;

    // Build and serialize a valid Index2Layer.
    auto original_q = std::make_unique<IndexFlatL2>(d);
    Index2Layer original(original_q.release(), nlist, 2);
    original.q1.own_fields = true;
    std::vector<float> xb(d * nb);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist;
        for (auto& v : xb) {
            v = dist(rng);
        }
    }
    original.train(nb, xb.data());
    original.add(nb, xb.data());

    VectorIOWriter writer;
    write_index(&original, &writer);

    // Deserialize and swap in a CountedDestructorIndex as the quantizer.
    // When the deserialized index is destroyed, own_fields controls
    // whether the quantizer is freed. The destructor counter verifies
    // this independently of any flag check.
    CountedDestructorIndex::destructor_count = 0;
    {
        VectorIOReader reader;
        reader.data = writer.data;
        auto idx = std::unique_ptr<Index>(read_index(&reader));
        auto* idx2l = dynamic_cast<Index2Layer*>(idx.get());
        ASSERT_NE(idx2l, nullptr);
        ASSERT_NE(idx2l->q1.quantizer, nullptr);

        // Replace the deserialized quantizer with a CountedDestructorIndex.
        delete idx2l->q1.quantizer;
        idx2l->q1.quantizer = new CountedDestructorIndex(d);
        EXPECT_EQ(CountedDestructorIndex::destructor_count, 0);
        // idx goes out of scope here. If own_fields is true,
        // ~Level1Quantizer deletes the CountedDestructorIndex.
    }
    EXPECT_EQ(CountedDestructorIndex::destructor_count, 1);
}

// -----------------------------------------------------------------------
// Test: Index2Layer with a null quantizer (serialized as "null" fourcc)
// still sets own_fields=true. Deleting a nullptr is safe, so this
// verifies the fix doesn't assume the quantizer is non-null.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, Index2LayerNullQuantizerOwnership) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Ix2L");
    push_index_header(buf, 4, 0);
    // Null sub-quantizer.
    push_fourcc(buf, "null");
    push_val<size_t>(buf, size_t(1)); // nlist
    // Truncate here — the next READ1 will fail and trigger cleanup.
    // With own_fields=true and quantizer=nullptr, the destructor
    // must not crash (delete nullptr is well-defined).
    EXPECT_THROW(
            {
                VectorIOReader reader;
                reader.data = buf;
                read_index(&reader);
            },
            FaissException);
}

// -----------------------------------------------------------------------
// Test: IndexHNSW2Level with null storage is rejected.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSW2LevelNullStorage) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IHN2");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_minimal_hnsw(buf, /*ntotal=*/0);
    // Null storage
    push_fourcc(buf, "null");

    expect_read_throws_with(buf, "non-null storage");
}

// -----------------------------------------------------------------------
// Test: NSG ntotal != index ntotal.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, NSGNtotalMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "INSf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/5);
    push_val<int>(buf, 0);  // GK
    push_val<char>(buf, 0); // build_type (char, not int)
    push_val<int>(buf, 10); // nndescent_S
    push_val<int>(buf, 10); // nndescent_R
    push_val<int>(buf, 10); // nndescent_L
    push_val<int>(buf, 1);  // nndescent_iter
    // NSG: ntotal=3 (mismatch with index ntotal=5)
    push_val<int>(buf, 3);     // ntotal
    push_val<int>(buf, 2);     // R
    push_val<int>(buf, 10);    // L
    push_val<int>(buf, 10);    // C
    push_val<int>(buf, 10);    // search_L
    push_val<int>(buf, 0);     // enterpoint
    push_val<bool>(buf, true); // is_built
    // Graph: 3 nodes, R=2, all empty (EMPTY_ID terminates each)
    for (int i = 0; i < 3; i++) {
        push_val<int>(buf, -1); // EMPTY_ID
    }

    expect_read_throws_with(buf, "NSG ntotal");
}

// -----------------------------------------------------------------------
// Test: NNDescent ntotal != index ntotal.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, NNDescentNtotalMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "INNf");
    push_index_header(buf, /*d=*/4, /*ntotal=*/5);
    // NNDescent: ntotal=3, has_built=true (mismatch with index ntotal=5)
    // Graph is internally valid: ntotal=3, K=2, 6 entries all in range
    push_nndescent(
            buf,
            /*ntotal=*/3,
            /*d=*/4,
            /*K=*/2,
            /*has_built=*/true,
            {0, 1, 0, 2, 1, 2});

    expect_read_throws_with(buf, "NNDescent ntotal");
}

// -----------------------------------------------------------------------
// Test: IndexHNSWCagra search with base_level_only on empty index.
// Without the fix, this would access uninitialized graph nodes.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, HNSWCagraEmptyIndexSearch) {
    IndexHNSWCagra idx(4, 16);
    idx.base_level_only = true;

    std::vector<float> xq(4, 1.0f);
    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);

    EXPECT_NO_THROW(
            idx.search(1, xq.data(), 1, distances.data(), labels.data()));
    EXPECT_EQ(labels[0], -1);
}

// -----------------------------------------------------------------------
// Test: IndexIVF search with null invlists (e.g. loaded with
// IO_FLAG_SKIP_IVF_DATA) throws instead of crashing.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexIVFNullInvlistsSearch) {
    IndexFlatL2 quantizer(4);
    IndexIVFFlat idx(&quantizer, 4, 10);
    idx.own_fields = false;
    // Simulate IO_FLAG_SKIP_IVF_DATA by deleting invlists
    delete idx.invlists;
    idx.invlists = nullptr;

    std::vector<float> xq(4, 1.0f);
    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);

    EXPECT_THROW(
            idx.search(1, xq.data(), 1, distances.data(), labels.data()),
            FaissException);
}

// -----------------------------------------------------------------------
// Test: IndexIVF add_with_ids with null invlists throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexIVFNullInvlistsAdd) {
    IndexFlatL2 quantizer(4);
    IndexIVFFlat idx(&quantizer, 4, 10);
    idx.own_fields = false;
    idx.is_trained = true;
    delete idx.invlists;
    idx.invlists = nullptr;

    std::vector<float> xb(4, 1.0f);

    EXPECT_THROW(idx.add(1, xb.data()), FaissException);
}

// -----------------------------------------------------------------------
// IVF quantizer ntotal / nlist deserialization acceptance tests.
// The quantizer may legitimately have ntotal != nlist (e.g., sharded
// indexes, custom inverted list management, untrained quantizers).
// -----------------------------------------------------------------------

// Surplus quantizer centroids: ntotal > nlist. Produced by
// shard_ivf_index_centroids(), which distributes all of the original
// quantizer's centroids across shards without adjusting nlist.
// The search-time key < nlist bounds check prevents OOB access if
// the quantizer returns out-of-range keys.
TEST(ReadIndexDeserialize, IVFQuantizerSurplus) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IwFl");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<size_t>(buf, 2); // nlist = 2
    push_val<size_t>(buf, 1); // nprobe
    // Quantizer with ntotal=5 (more centroids than nlist)
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/5);
    push_empty_direct_map(buf);
    push_null_invlists(buf);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// Trained quantizer: ntotal == nlist (normal trained IVF).
TEST(ReadIndexDeserialize, IVFQuantizerTrained) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IwFl");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<size_t>(buf, 2); // nlist = 2
    push_val<size_t>(buf, 1); // nprobe
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/2);
    push_empty_direct_map(buf);
    push_null_invlists(buf);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// Sharded quantizer: 0 < ntotal < nlist. Produced by
// shard_ivf_index_centroids(), where each shard's quantizer holds a
// subset of the full index's centroids.
TEST(ReadIndexDeserialize, IVFQuantizerSubset) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IwFl");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<size_t>(buf, 10); // nlist = 10
    push_val<size_t>(buf, 1);  // nprobe
    // Quantizer with ntotal=3 (subset of centroids, as in sharding)
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/3);
    push_empty_direct_map(buf);
    push_null_invlists(buf);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// Untrained quantizer: ntotal == 0 (custom inverted list management).
TEST(ReadIndexDeserialize, IVFQuantizerUntrained) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IwFl");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<size_t>(buf, 10); // nlist = 10
    push_val<size_t>(buf, 1);  // nprobe
    push_minimal_flat(buf, /*d=*/4, /*ntotal=*/0);
    push_empty_direct_map(buf);
    push_null_invlists(buf);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// -----------------------------------------------------------------------
// Test: IndexIVFScalarQuantizer with empty trained vector and
// is_trained=false deserializes successfully (legitimate untrained index),
// but searching it throws because IndexIVF::search checks is_trained.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IVFScalarQuantizerUntrainedSearchRejected) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IwSq");
    // IVF header: index_header + nlist + nprobe + quantizer + direct_map
    push_index_header(buf, /*d=*/4, /*ntotal=*/0, /*is_trained=*/false);
    push_val<size_t>(buf, 1); // nlist
    push_val<size_t>(buf, 1); // nprobe
    push_minimal_flat(buf, /*d=*/4);
    push_empty_direct_map(buf);
    // ScalarQuantizer fields:
    push_val<int>(buf, 0);       // qtype = QT_8bit
    push_val<int>(buf, 0);       // rangestat
    push_val<float>(buf, 0.0f);  // rangestat_arg
    push_val<size_t>(buf, 4);    // d
    push_val<size_t>(buf, 4);    // code_size
    push_vector<float>(buf, {}); // trained (empty — untrained)
    // IwSq additional fields:
    push_val<size_t>(buf, 4);   // code_size
    push_val<bool>(buf, false); // by_residual
    push_null_invlists(buf);

    // Deserialization should succeed — untrained indexes are legitimate
    VectorIOReader reader;
    reader.data = buf;
    auto idx = read_index_up(&reader);
    ASSERT_NE(idx, nullptr);
    EXPECT_FALSE(idx->is_trained);

    // search should throw — is_trained check in IndexIVF::search
    std::vector<float> xq(4, 0.0f);
    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);
    EXPECT_THROW(
            idx->search(1, xq.data(), 1, distances.data(), labels.data()),
            FaissException);

    // range_search should throw — is_trained check in IndexIVF::range_search
    RangeSearchResult rsr(1);
    EXPECT_THROW(idx->range_search(1, xq.data(), 1.0f, &rsr), FaissException);

    // search_preassigned should throw directly
    auto* ivf = dynamic_cast<IndexIVF*>(idx.get());
    ASSERT_NE(ivf, nullptr);
    idx_t key = 0;
    float coarse_dis = 0.0f;
    EXPECT_THROW(
            ivf->search_preassigned(
                    1,
                    xq.data(),
                    1,
                    &key,
                    &coarse_dis,
                    distances.data(),
                    labels.data(),
                    false,
                    nullptr,
                    nullptr),
            FaissException);

    // range_search_preassigned should throw directly
    RangeSearchResult rsr2(1);
    EXPECT_THROW(
            ivf->range_search_preassigned(
                    1,
                    xq.data(),
                    1.0f,
                    &key,
                    &coarse_dis,
                    &rsr2,
                    false,
                    nullptr,
                    nullptr),
            FaissException);
}

// -----------------------------------------------------------------------
// Test: IndexIVFScalarQuantizer with is_trained=true but empty trained
// is rejected at deserialization time — corrupt data.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IVFScalarQuantizerTrainedEmptyTrained) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IwSq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0, /*is_trained=*/true);
    push_val<size_t>(buf, 1); // nlist
    push_val<size_t>(buf, 1); // nprobe
    push_minimal_flat(buf, /*d=*/4);
    push_empty_direct_map(buf);
    // ScalarQuantizer fields:
    push_val<int>(buf, 0);       // qtype = QT_8bit
    push_val<int>(buf, 0);       // rangestat
    push_val<float>(buf, 0.0f);  // rangestat_arg
    push_val<size_t>(buf, 4);    // d
    push_val<size_t>(buf, 4);    // code_size
    push_vector<float>(buf, {}); // trained (empty — but is_trained=true!)

    expect_read_throws_with(buf, "ScalarQuantizer trained size");
}

// -----------------------------------------------------------------------
// Test: initialize_IVFPQ_precomputed_table rejects a null quantizer.
// Protects against null-deref from corrupt serialized data where the
// quantizer sub-index is absent (fourcc "null").
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IVFPQNullQuantizerPrecomputeTableRejected) {
    ProductQuantizer pq(4, 1, 8);
    AlignedTable<float> precomputed_table;
    int use_precomputed_table = 0;
    EXPECT_THROW(
            initialize_IVFPQ_precomputed_table(
                    use_precomputed_table,
                    /*quantizer=*/nullptr,
                    pq,
                    precomputed_table,
                    /*by_residual=*/true,
                    /*verbose=*/false),
            faiss::FaissException);
}

TEST(ReadIndexDeserialize, IVFNullQuantizerSearchRejected) {
    IndexIVFFlat ivf;
    ivf.quantizer = nullptr;
    ivf.is_trained = true;
    std::vector<float> x(4);
    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);
    EXPECT_THROW(
            ivf.search(1, x.data(), 1, distances.data(), labels.data()),
            faiss::FaissException);
}

TEST(ReadIndexDeserialize, IVFNullQuantizerRangeSearchRejected) {
    IndexIVFFlat ivf;
    ivf.quantizer = nullptr;
    ivf.is_trained = true;
    std::vector<float> x(4);
    RangeSearchResult result(1);
    EXPECT_THROW(
            ivf.range_search(1, x.data(), 1.0, &result), faiss::FaissException);
}

TEST(ReadIndexDeserialize, IVFNullQuantizerAddRejected) {
    IndexIVFFlat ivf;
    ivf.quantizer = nullptr;
    std::vector<float> x(4);
    EXPECT_THROW(ivf.add(1, x.data()), faiss::FaissException);
}

TEST(ReadIndexDeserialize, IVFNullQuantizerTrainRejected) {
    IndexIVFFlat ivf;
    ivf.quantizer = nullptr;
    std::vector<float> x(4);
    EXPECT_THROW(ivf.train(1, x.data()), faiss::FaissException);
}

// -----------------------------------------------------------------------
// VectorTransform deserialization validation tests
// -----------------------------------------------------------------------

// Test: NormalizationTransform with d_in != d_out is rejected.
// Protects against corrupt serialized data where dimension mismatch
// causes memcpy to overflow the output buffer in apply_noalloc.
TEST(ReadIndexDeserialize, NormalizationTransformDinDoutMismatch) {
    // VNrm format: fourcc("VNrm") + norm(float) + d_in(int) + d_out(int) +
    //              is_trained(bool)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "VNrm");
    push_val<float>(buf, 2.0f); // norm (L2)
    push_val<int>(buf, 16);     // d_in
    push_val<int>(buf, 8);      // d_out (mismatch!)
    push_val<bool>(buf, true);  // is_trained

    expect_vt_read_throws_with(buf, "d_in == d_out");
}

// Test: NormalizationTransform with d_in == d_out is accepted.
TEST(ReadIndexDeserialize, NormalizationTransformDinDoutMatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "VNrm");
    push_val<float>(buf, 2.0f); // norm (L2)
    push_val<int>(buf, 16);     // d_in
    push_val<int>(buf, 16);     // d_out (match)
    push_val<bool>(buf, true);  // is_trained

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_VectorTransform_up(&reader));
}

// Test: CenteringTransform with d_in != d_out is rejected.
TEST(ReadIndexDeserialize, CenteringTransformDinDoutMismatch) {
    // VCnt format: fourcc("VCnt") + mean(vector<float>) + d_in(int) +
    //              d_out(int) + is_trained(bool)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "VCnt");
    push_vector<float>(buf, std::vector<float>(16, 0.0f)); // mean
    push_val<int>(buf, 16);                                // d_in
    push_val<int>(buf, 8);                                 // d_out (mismatch!)
    push_val<bool>(buf, true);

    expect_vt_read_throws_with(buf, "d_in == d_out");
}

// Test: IndexPreTransform with mismatched chain dimensions is rejected.
TEST(ReadIndexDeserialize, PreTransformChainDimensionMismatch) {
    // Build an IxPT with a NormalizationTransform (d_in=d_out=4) followed
    // by a sub-index with d=8. The chain's d_out (4) != sub-index d (8).
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPT");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<int>(buf, 1); // nt = 1 transform
    // NormalizationTransform with d_in=d_out=4
    push_fourcc(buf, "VNrm");
    push_val<float>(buf, 2.0f);
    push_val<int>(buf, 4); // d_in
    push_val<int>(buf, 4); // d_out
    push_val<bool>(buf, true);
    // Sub-index: IndexFlat with d=8 (mismatch with chain d_out=4)
    push_minimal_flat(buf, /*d=*/8);

    expect_read_throws_with(buf, "d_out=4");
}

TEST(ReadIndexDeserialize, HadamardRotationInvalidDout) {
    // HRot format: fourcc("HRot") + seed(int) + d_in(int) + d_out(int) +
    //              is_trained(bool)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "HRot");
    push_val<int>(buf, 42);      // seed
    push_val<int>(buf, 16);      // d_in
    push_val<int>(buf, 1 << 21); // d_out (not power-of-2 match for d_in)
    push_val<bool>(buf, true);   // is_trained

    expect_vt_read_throws_with(buf, "d_out must be the smallest power of 2");
}

TEST(ReadIndexDeserialize, HadamardRotationDinZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "HRot");
    push_val<int>(buf, 42);    // seed
    push_val<int>(buf, 0);     // d_in = 0 (invalid)
    push_val<int>(buf, 1);     // d_out
    push_val<bool>(buf, true); // is_trained

    expect_vt_read_throws_with(buf, "HadamardRotation d_in=");
}

TEST(ReadIndexDeserialize, HadamardRotationDoutZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "HRot");
    push_val<int>(buf, 42);    // seed
    push_val<int>(buf, 16);    // d_in
    push_val<int>(buf, 0);     // d_out = 0 (invalid)
    push_val<bool>(buf, true); // is_trained

    expect_vt_read_throws_with(buf, "HadamardRotation d_out=");
}

TEST(ReadIndexDeserialize, RemapDimensionsTransformMapTooSmall) {
    // RmDT format: fourcc("RmDT") + WRITEVECTOR(map) + d_in(int) +
    //              d_out(int) + is_trained(bool)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "RmDT");
    push_vector<int>(buf, {0, 1}); // map with 2 entries
    push_val<int>(buf, 4);         // d_in
    push_val<int>(buf, 10);        // d_out = 10 > map.size()=2
    push_val<bool>(buf, true);     // is_trained

    expect_vt_read_throws_with(buf, "RemapDimensionsTransform map size");
}

TEST(ReadIndexDeserialize, CenteringTransformMeanTooSmall) {
    // VCnt format: fourcc("VCnt") + WRITEVECTOR(mean) + d_in(int) +
    //              d_out(int) + is_trained(bool)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "VCnt");
    push_vector<float>(buf, {1.0f, 2.0f}); // mean with 2 entries
    push_val<int>(buf, 8);                 // d_in = 8 > mean.size()=2
    push_val<int>(buf, 8);                 // d_out
    push_val<bool>(buf, true);             // is_trained

    expect_vt_read_throws_with(buf, "CenteringTransform mean size");
}

TEST(ReadIndexDeserialize, ITQTransformMeanTooSmall) {
    // Viqt format: fourcc("Viqt") + WRITEVECTOR(mean) + do_pca(int) +
    //              sub_vt(ITQMatrix) + sub_vt(LinearTransform) +
    //              d_in(int) + d_out(int) + is_trained(bool)
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Viqt");
    push_vector<float>(buf, {1.0f}); // mean with 1 entry
    push_val<bool>(buf, false);      // do_pca

    // Sub-VT 1: ITQMatrix ("Viqm")
    push_fourcc(buf, "Viqm");
    push_val<int>(buf, 10);      // max_iter
    push_val<int>(buf, 42);      // seed
    push_val<bool>(buf, false);  // have_bias
    push_vector<float>(buf, {}); // A
    push_vector<float>(buf, {}); // b
    push_val<int>(buf, 0);       // d_in
    push_val<int>(buf, 0);       // d_out
    push_val<bool>(buf, true);   // is_trained

    // Sub-VT 2: LinearTransform ("LTra")
    push_fourcc(buf, "LTra");
    push_val<bool>(buf, false);  // have_bias
    push_vector<float>(buf, {}); // A
    push_vector<float>(buf, {}); // b
    push_val<int>(buf, 0);       // d_in
    push_val<int>(buf, 0);       // d_out
    push_val<bool>(buf, true);   // is_trained

    // Outer Viqt footer
    push_val<int>(buf, 4);     // d_in = 4 > mean.size()=1
    push_val<int>(buf, 4);     // d_out
    push_val<bool>(buf, true); // is_trained

    expect_vt_read_throws_with(buf, "ITQTransform mean size");
}

// -----------------------------------------------------------------------
// RaBitQ qb deserialization validation tests
// -----------------------------------------------------------------------

/// Helper: push a minimal RaBitQuantizer (single-bit format, multi_bit=false).
static void push_rabitq(std::vector<uint8_t>& buf, size_t d) {
    push_val<size_t>(buf, d); // d
    push_val<size_t>(buf, 1); // code_size
    push_val<int>(buf, 1);    // metric_type (L2)
}

/// Helper: push a minimal RaBitQuantizer (multi-bit format, multi_bit=true).
static void push_rabitq_multibit(std::vector<uint8_t>& buf, size_t d) {
    push_val<size_t>(buf, d); // d
    push_val<size_t>(buf, 1); // code_size
    push_val<int>(buf, 1);    // metric_type (L2)
    push_val<size_t>(buf, 1); // nb_bits
}

/// Helper: push an IVF header (index_header + nlist + nprobe + flat quantizer
/// + empty direct_map).
static void push_ivf_header(std::vector<uint8_t>& buf, int d) {
    push_index_header(buf, d, /*ntotal=*/0);
    push_val<size_t>(buf, 1); // nlist
    push_val<size_t>(buf, 1); // nprobe
    push_minimal_flat(buf, d);
    push_empty_direct_map(buf);
}

// -----------------------------------------------------------------------
// Test: Inverted list with oversized entry exceeding deserialization byte
// limit is rejected. Protects against corrupt data causing OOM via
// unchecked vector::resize in read_InvertedLists_up.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, InvertedListOversizedEntry) {
    const size_t old_limit = get_deserialization_vector_byte_limit();
    set_deserialization_vector_byte_limit(1024); // 1 KB limit

    std::vector<uint8_t> buf;
    push_fourcc(buf, "IwFl");
    push_ivf_header(buf, /*d=*/4);
    // "ilar" inverted lists with 1 list, code_size=16
    push_fourcc(buf, "ilar");
    push_val<size_t>(buf, 1);  // nlist = 1
    push_val<size_t>(buf, 16); // code_size = 16
    // "full" list type with 1 size entry
    push_fourcc(buf, "full");
    // sizes vector: 1 entry with absurdly large value
    push_val<size_t>(buf, 1);               // vector length
    push_val<size_t>(buf, (size_t)1 << 40); // sizes[0] = 1 TB

    expect_read_throws_with(buf, "deserialization byte limit");

    set_deserialization_vector_byte_limit(old_limit);
}

// -- Ixrq (IndexRaBitQ, single-bit) --
// qb=0 is valid for Ixrq: disables query quantization, uses raw fp32 values.
// See IndexRaBitQ.h comment on qb and RaBitQDistanceComputerNotQ.

TEST(ReadIndexDeserialize, RaBitQQbTooLarge_Ixrq) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Ixrq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_rabitq(buf, 4);
    push_vector<uint8_t>(buf, {});         // codes
    push_vector<float>(buf, {0, 0, 0, 0}); // center
    push_val<uint8_t>(buf, 9);             // qb = 9 (> 8, invalid)

    expect_read_throws_with(buf, "RaBitQ qb=");
}

TEST(ReadIndexDeserialize, RaBitQQbZeroAccepted_Ixrq) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Ixrq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_rabitq(buf, 4);
    push_vector<uint8_t>(buf, {});         // codes
    push_vector<float>(buf, {0, 0, 0, 0}); // center
    push_val<uint8_t>(buf, 0);             // qb = 0 (valid for Ixrq)

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// -- Ixrr (IndexRaBitQ, multi-bit) --
// qb=0 is valid for Ixrr: disables query quantization, uses raw fp32 values.
// See IndexRaBitQ.h comment on qb and RaBitQDistanceComputerNotQ.

TEST(ReadIndexDeserialize, RaBitQQbTooLarge_Ixrr) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Ixrr");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_rabitq_multibit(buf, 4);
    push_vector<uint8_t>(buf, {});         // codes
    push_vector<float>(buf, {0, 0, 0, 0}); // center
    push_val<uint8_t>(buf, 9);             // qb = 9 (> 8, invalid)

    expect_read_throws_with(buf, "RaBitQ qb=");
}

TEST(ReadIndexDeserialize, RaBitQQbZeroAccepted_Ixrr) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Ixrr");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_rabitq_multibit(buf, 4);
    push_vector<uint8_t>(buf, {});         // codes
    push_vector<float>(buf, {0, 0, 0, 0}); // center
    push_val<uint8_t>(buf, 0);             // qb = 0 (valid for Ixrr)

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// -- Irfn (IndexRaBitQFastScan, new format) --
// qb=0 is not supported: FastScan requires quantized queries for SIMD.

TEST(ReadIndexDeserialize, RaBitQQbTooLarge_Irfn) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Irfn");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_rabitq_multibit(buf, 4);
    push_vector<float>(buf, {0, 0, 0, 0}); // center
    push_val<uint8_t>(buf, 9);             // qb = 9 (> 8, invalid)

    expect_read_throws_with(buf, "RaBitQ qb=");
}

TEST(ReadIndexDeserialize, RaBitQQbZeroRejected_Irfn) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Irfn");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_rabitq_multibit(buf, 4);
    push_vector<float>(buf, {0, 0, 0, 0}); // center
    push_val<uint8_t>(buf, 0);             // qb = 0 (invalid for FastScan)

    expect_read_throws_with(buf, "RaBitQ qb=");
}

// -- Iwrq (IndexIVFRaBitQ, single-bit) --
// qb=0 is valid for Iwrq: disables query quantization, uses raw fp32 values.
// See IndexIVFRaBitQ.h comment on qb and RaBitQDistanceComputerNotQ.

TEST(ReadIndexDeserialize, RaBitQQbTooLarge_Iwrq) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Iwrq");
    push_ivf_header(buf, /*d=*/4);
    push_rabitq(buf, 4);
    push_val<size_t>(buf, 1);   // code_size
    push_val<bool>(buf, false); // by_residual
    push_val<uint8_t>(buf, 9);  // qb = 9 (> 8, invalid)

    expect_read_throws_with(buf, "RaBitQ qb=");
}

TEST(ReadIndexDeserialize, RaBitQQbZeroAccepted_Iwrq) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Iwrq");
    push_ivf_header(buf, /*d=*/4);
    push_rabitq(buf, 4);
    push_val<size_t>(buf, 1);   // code_size
    push_val<bool>(buf, false); // by_residual
    push_val<uint8_t>(buf, 0);  // qb = 0 (valid for Iwrq)
    push_null_invlists(buf);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// -- Iwrr (IndexIVFRaBitQ, multi-bit) --
// qb=0 is valid for Iwrr: disables query quantization, uses raw fp32 values.
// See IndexIVFRaBitQ.h comment on qb and RaBitQDistanceComputerNotQ.

TEST(ReadIndexDeserialize, RaBitQQbTooLarge_Iwrr) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Iwrr");
    push_ivf_header(buf, /*d=*/4);
    push_rabitq_multibit(buf, 4);
    push_val<size_t>(buf, 1);   // code_size
    push_val<bool>(buf, false); // by_residual
    push_val<uint8_t>(buf, 9);  // qb = 9 (> 8, invalid)

    expect_read_throws_with(buf, "RaBitQ qb=");
}

TEST(ReadIndexDeserialize, RaBitQQbZeroAccepted_Iwrr) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Iwrr");
    push_ivf_header(buf, /*d=*/4);
    push_rabitq_multibit(buf, 4);
    push_val<size_t>(buf, 1);   // code_size
    push_val<bool>(buf, false); // by_residual
    push_val<uint8_t>(buf, 0);  // qb = 0 (valid for Iwrr)
    push_null_invlists(buf);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// -- Iwrn (IndexIVFRaBitQFastScan, new format) --
// qb=0 is not supported: IVF FastScan requires quantized queries.

TEST(ReadIndexDeserialize, RaBitQQbTooLarge_Iwrn) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Iwrn");
    push_ivf_header(buf, /*d=*/4);
    push_rabitq_multibit(buf, 4);
    push_val<bool>(buf, false); // by_residual
    push_val<size_t>(buf, 1);   // code_size
    push_val<int>(buf, 32);     // bbs
    push_val<size_t>(buf, 0);   // qbs2
    push_val<size_t>(buf, 1);   // M2
    push_val<int>(buf, 0);      // implem
    push_val<uint8_t>(buf, 9);  // qb = 9 (> 8, invalid)

    expect_read_throws_with(buf, "RaBitQ qb=");
}

TEST(ReadIndexDeserialize, RaBitQQbZeroRejected_Iwrn) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "Iwrn");
    push_ivf_header(buf, /*d=*/4);
    push_rabitq_multibit(buf, 4);
    push_val<bool>(buf, false); // by_residual
    push_val<size_t>(buf, 1);   // code_size
    push_val<int>(buf, 32);     // bbs
    push_val<size_t>(buf, 0);   // qbs2
    push_val<size_t>(buf, 1);   // M2
    push_val<int>(buf, 0);      // implem
    push_val<uint8_t>(buf, 0);  // qb = 0 (invalid for IVF FastScan)

    expect_read_throws_with(buf, "RaBitQ qb=");
}

// -----------------------------------------------------------------------
// Binary index deserialization validation tests
// -----------------------------------------------------------------------

TEST(ReadIndexDeserialize, BinaryHeaderDimensionNotMultipleOf8) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBxF");
    push_val<int>(buf, 17);    // d = 17 (not multiple of 8)
    push_val<int>(buf, 2);     // code_size
    push_val<int64_t>(buf, 0); // ntotal
    push_val<bool>(buf, true); // is_trained
    push_val<int>(buf, 1);     // metric_type

    expect_binary_read_throws_with(buf, "multiple of 8");
}

TEST(ReadIndexDeserialize, BinaryHeaderCodeSizeMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBxF");
    push_val<int>(buf, 16);    // d = 16
    push_val<int>(buf, 5);     // code_size = 5 (should be 2)
    push_val<int64_t>(buf, 0); // ntotal
    push_val<bool>(buf, true); // is_trained
    push_val<int>(buf, 1);     // metric_type

    expect_binary_read_throws_with(buf, "code_size");
}

TEST(ReadIndexDeserialize, BinaryHNSWLevelsSizeMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHf");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/5);
    push_minimal_hnsw(buf, /*ntotal=*/3); // 3 != 5

    expect_binary_read_throws_with(buf, "HNSW levels size");
}

TEST(ReadIndexDeserialize, BinaryHNSWNeighborNegativeId) {
    int ntotal = 3, npn = 2;
    std::vector<int32_t> neighbors = {1, -5, 0, -1, 0, -1};

    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHf");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/ntotal);
    push_hnsw_with_neighbors(buf, ntotal, npn, neighbors);

    expect_binary_read_throws_with(buf, "HNSW neighbors");
}

TEST(ReadIndexDeserialize, BinaryHNSWStorageNtotalMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHf");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/3);
    push_minimal_hnsw(buf, /*ntotal=*/3);
    push_minimal_binary_flat(buf, /*d=*/16); // ntotal=0 != 3

    expect_binary_read_throws_with(buf, "storage ntotal");
}

TEST(ReadIndexDeserialize, BinaryHNSWStorageNotFlat) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHf");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_minimal_hnsw(buf, /*ntotal=*/0);
    // Nest an IBwF (IVF) instead of IBxF (flat)
    push_fourcc(buf, "IBwF");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<size_t>(buf, 1);                // nlist
    push_val<size_t>(buf, 1);                // nprobe
    push_minimal_binary_flat(buf, /*d=*/16); // quantizer
    push_empty_direct_map(buf);
    push_null_invlists(buf);

    expect_binary_read_throws_with(buf, "IndexBinaryFlat storage");
}

TEST(ReadIndexDeserialize, BinaryHNSWCagraLevelsSizeMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHc");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/5);
    push_val<bool>(buf, true);            // keep_max_size_level0
    push_val<bool>(buf, false);           // base_level_only
    push_val<int>(buf, 10);               // num_base_level_search_entrypoints
    push_minimal_hnsw(buf, /*ntotal=*/3); // 3 != 5

    expect_binary_read_throws_with(buf, "HNSW levels size");
}

TEST(ReadIndexDeserialize, BinaryHNSWCagraStorageNotFlat) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHc");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<bool>(buf, true);  // keep_max_size_level0
    push_val<bool>(buf, false); // base_level_only
    push_val<int>(buf, 10);     // num_base_level_search_entrypoints
    push_minimal_hnsw(buf, /*ntotal=*/0);
    // Nest an IBwF (IVF) instead of IBxF (flat)
    push_fourcc(buf, "IBwF");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<size_t>(buf, 1);                // nlist
    push_val<size_t>(buf, 1);                // nprobe
    push_minimal_binary_flat(buf, /*d=*/16); // quantizer
    push_empty_direct_map(buf);
    push_null_invlists(buf);

    expect_binary_read_throws_with(buf, "IndexBinaryFlat storage");
}

TEST(ReadIndexDeserialize, BinaryHNSWCagraStorageNtotalMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHc");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/3);
    push_val<bool>(buf, true);  // keep_max_size_level0
    push_val<bool>(buf, false); // base_level_only
    push_val<int>(buf, 10);     // num_base_level_search_entrypoints
    push_minimal_hnsw(buf, /*ntotal=*/3);
    push_minimal_binary_flat(buf, /*d=*/16); // ntotal=0 != 3

    expect_binary_read_throws_with(buf, "storage ntotal");
}

TEST(ReadIndexDeserialize, BinaryHashInvlistsVecsSizeMismatch) {
    // IBHh: fourcc + binary_header + b + nflip + hash_invlists
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHh");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_val<int>(buf, 4); // b
    push_val<int>(buf, 0); // nflip

    // hash invlists with 1 entry
    push_val<size_t>(buf, size_t(1)); // sz = 1
    push_val<int>(buf, 8);            // il_nbit = 8

    // Bitstring: 1 entry needs (b + il_nbit) = 12 bits -> 2 bytes
    std::vector<uint8_t> bitbuf(2, 0);
    BitstringWriter wr(bitbuf.data(), bitbuf.size());
    wr.write(0, 4); // hash = 0
    wr.write(1, 8); // ilsz = 1
    push_vector<uint8_t>(buf, bitbuf);

    // ids: 1 entry
    push_vector<int64_t>(buf, {0});
    // vecs: wrong size (3 bytes instead of code_size=2)
    push_vector<uint8_t>(buf, {0, 0, 0});

    expect_binary_read_throws_with(buf, "binary hash invlists: vecs size");
}

TEST(ReadIndexDeserialize, BinaryMultiHashMapIdOutOfRange) {
    // IBHm with ntotal=1, one map containing id=1 (>= ntotal).
    const int b = 4;
    const int id_bits = 1;
    const size_t ntotal = 1;
    const size_t sz = 1;
    // bits: (b + id_bits) * sz + 1 * id_bits = 6 bits -> 1 byte
    const size_t nbit = (b + id_bits) * sz + 1 * id_bits;
    const size_t bitbuf_size = (nbit + 7) / 8;

    std::vector<uint8_t> bitbuf(bitbuf_size, 0);
    BitstringWriter wr2(bitbuf.data(), bitbuf.size());
    wr2.write(0, b);       // hash = 0
    wr2.write(1, id_bits); // ilsz = 1
    wr2.write(1, id_bits); // id = 1 (>= ntotal=1, invalid)

    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHm");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/ntotal);

    // Nested IBxF storage
    push_fourcc(buf, "IBxF");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/ntotal);
    std::vector<uint8_t> xb(ntotal * 2, 0); // code_size=2
    push_vector<uint8_t>(buf, xb);

    push_val<int>(buf, b); // b
    push_val<int>(buf, 1); // nhash = 1
    push_val<int>(buf, 0); // nflip

    // Multi hash map fields (1 map):
    push_val<int>(buf, id_bits);       // id_bits
    push_val<size_t>(buf, sz);         // sz = 1 entry
    push_vector<uint8_t>(buf, bitbuf); // packed bitstring

    expect_binary_read_throws_with(buf, "multi hash map: id=");
}

TEST(ReadIndexDeserialize, BinaryMultiHashBZero) {
    // b must be positive; BitstringReader::read(0) produces garbage hash
    // values and b is used as a bit-width for hash extraction.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHm");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);

    // Nested IBxF storage with ntotal=0
    push_fourcc(buf, "IBxF");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_vector<uint8_t>(buf, {}); // empty xb

    push_val<int>(buf, 0); // b = 0 (invalid)

    expect_binary_read_throws_with(buf, "IndexBinaryMultiHash b=");
}

TEST(ReadIndexDeserialize, BinaryMultiHashNhashTimesBExceedsCodeSize) {
    // nhash * b must not exceed code_size * 8, otherwise BitstringReader
    // overflows the query buffer during search.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBHm");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);

    // Nested IBxF storage with ntotal=0
    push_fourcc(buf, "IBxF");
    push_binary_index_header(buf, /*d=*/16, /*ntotal=*/0);
    push_vector<uint8_t>(buf, {}); // empty xb

    push_val<int>(buf, 12); // b = 12
    push_val<int>(buf, 2);  // nhash = 2 => nhash*b = 24 > code_size*8 = 16
    // (code_size = d/8 = 2, so 2*8 = 16 bits available)

    expect_binary_read_throws_with(buf, "exceeds code_size");
}

// ---- IndexBinaryHNSW runtime safety checks ----

TEST(ReadIndexDeserialize, BinaryHNSWGetDistanceComputerNonFlatThrows) {
    // Construct an IndexBinaryHNSW with non-flat storage and verify
    // get_distance_computer throws instead of asserting.
    IndexBinaryHNSW idx;
    idx.storage = nullptr;
    idx.own_fields = false;
    // storage is nullptr so dynamic_cast will fail
    EXPECT_THROW(
            {
                try {
                    idx.get_distance_computer();
                } catch (const faiss::FaissException& e) {
                    EXPECT_NE(
                            std::string(e.what()).find("IndexBinaryFlat"),
                            std::string::npos);
                    throw;
                }
            },
            faiss::FaissException);
}

TEST(ReadIndexDeserialize, BinaryHNSWCagraEmptyIndexSearch) {
    IndexBinaryHNSWCagra idx(16, 4);
    idx.base_level_only = true;
    // ntotal is 0, searching should throw
    std::vector<int32_t> distances(1);
    std::vector<idx_t> labels(1);
    std::vector<uint8_t> query(2); // d=16 -> code_size=2
    EXPECT_THROW(
            {
                try {
                    idx.search(
                            1,
                            query.data(),
                            1,
                            distances.data(),
                            labels.data());
                } catch (const faiss::FaissException& e) {
                    EXPECT_NE(
                            std::string(e.what()).find("empty index"),
                            std::string::npos);
                    throw;
                }
            },
            faiss::FaissException);
}

TEST(ReadIndexDeserialize, BinaryHNSWCagraZeroEntrypoints) {
    IndexBinaryHNSWCagra idx(16, 4);
    // Add a vector first (add requires base_level_only == false)
    std::vector<uint8_t> vec(2, 0xFF);
    idx.add(1, vec.data());
    idx.base_level_only = true;
    idx.num_base_level_search_entrypoints = 0;
    std::vector<int32_t> distances(1);
    std::vector<idx_t> labels(1);
    std::vector<uint8_t> query(2);
    EXPECT_THROW(
            {
                try {
                    idx.search(
                            1,
                            query.data(),
                            1,
                            distances.data(),
                            labels.data());
                } catch (const faiss::FaissException& e) {
                    EXPECT_NE(
                            std::string(e.what()).find(
                                    "num_base_level_search_entrypoints"),
                            std::string::npos);
                    throw;
                }
            },
            faiss::FaissException);
}

// -----------------------------------------------------------------------
// Test: ResidualCoarseQuantizer with huge ntotal triggers centroid_norms
// byte limit check (pre-existing guard on ntotal vs byte_limit/sizeof(float)).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ResidualCoarseQuantizerHugeNtotal) {
    // "ImRQ": fourcc + index_header + ResidualQuantizer + beam_factor
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    // ntotal = 2^60: huge value that would cause OOM in search()
    push_index_header(buf, /*d=*/4, /*ntotal=*/int64_t{1} << 60);
    push_residual_quantizer(buf, /*d=*/4, /*M=*/2, /*nbits=*/{4, 4});
    push_val<float>(buf, -1.0f); // beam_factor = -1 (skip tables)

    expect_read_throws_with(buf, "centroid norms allocation");
}

// -----------------------------------------------------------------------
// Test: ResidualCoarseQuantizer ntotal * M exceeds byte limit even when
// ntotal alone passes the centroid_norms check.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ResidualCoarseQuantizerNtotalTimesM) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    // ntotal=200000 passes centroid_norms check (200000 < 1MB/4 = 262144)
    // but ntotal*M*sizeof(int32_t) = 200000*2*4 = 1600000 > 1MB
    push_index_header(buf, /*d=*/4, /*ntotal=*/200000);
    push_residual_quantizer(buf, /*d=*/4, /*M=*/2, /*nbits=*/{4, 4});
    push_val<float>(buf, -1.0f); // beam_factor = -1 (skip tables)

    auto old_limit = get_deserialization_vector_byte_limit();
    set_deserialization_vector_byte_limit(1 << 20); // 1 MB
    expect_read_throws_with(buf, "deserialization vector byte limit");
    set_deserialization_vector_byte_limit(old_limit);
}

// -----------------------------------------------------------------------
// Test: ResidualCoarseQuantizer with huge beam_factor throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ResidualCoarseQuantizerHugeBeamFactor) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/16);
    push_residual_quantizer(buf, /*d=*/4, /*M=*/2, /*nbits=*/{4, 4});
    push_val<float>(buf, 1e10f); // beam_factor = 1e10 (way too large)

    expect_read_throws_with(buf, "beam_factor");
}

// -----------------------------------------------------------------------
// Test: ResidualQuantizer with max_beam_size=0 throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ResidualQuantizerMaxBeamSizeZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/16);
    // Build AdditiveQuantizer manually + bad max_beam_size
    push_additive_quantizer(buf, /*d=*/4, /*M=*/2, /*nbits=*/{4, 4});
    push_val<int>(buf, 2048); // train_type = Skip_codebook_tables
    push_val<int>(buf, 0);    // max_beam_size = 0 (invalid)

    expect_read_throws_with(buf, "max_beam_size");
}

// -----------------------------------------------------------------------
// Test: ResidualQuantizer with huge max_beam_size exceeds byte limit.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ResidualQuantizerHugeMaxBeamSize) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/16);
    push_additive_quantizer(buf, /*d=*/4, /*M=*/2, /*nbits=*/{4, 4});
    push_val<int>(buf, 2048);    // train_type = Skip_codebook_tables
    push_val<int>(buf, 1 << 30); // max_beam_size = 2^30 (huge)

    // With a tight byte limit, this should fail
    auto old_limit = get_deserialization_vector_byte_limit();
    set_deserialization_vector_byte_limit(1 << 20); // 1 MB
    expect_read_throws_with(buf, "deserialization vector byte limit");
    set_deserialization_vector_byte_limit(old_limit);
}

// -----------------------------------------------------------------------
// Test: Negative ntotal is rejected by read_index_header.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ResidualCoarseQuantizerNegativeNtotal) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/-1);
    push_residual_quantizer(buf, /*d=*/4, /*M=*/2, /*nbits=*/{4, 4});
    push_val<float>(buf, -1.0f); // beam_factor

    expect_read_throws_with(buf, "invalid ntotal");
}

// -----------------------------------------------------------------------
// Test: AdditiveQuantizer with M=0 is rejected during deserialization.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AdditiveQuantizerMZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/16);
    // Build AdditiveQuantizer manually with M=0
    push_additive_quantizer(buf, /*d=*/4, /*M=*/0, /*nbits=*/{});
    push_val<int>(buf, 2048); // train_type = Skip_codebook_tables
    push_val<int>(buf, 1);    // max_beam_size

    expect_read_throws_with(buf, "invalid AdditiveQuantizer M");
}

// -----------------------------------------------------------------------
// Test: ResidualCoarseQuantizer with ntotal=0 deserializes successfully
// (empty index is valid).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ResidualCoarseQuantizerNtotalZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_residual_quantizer(buf, /*d=*/4, /*M=*/2, /*nbits=*/{4, 4});
    push_val<float>(buf, -1.0f); // beam_factor = -1 (skip tables)

    VectorIOReader reader;
    reader.data = buf;
    auto idx = read_index_up(&reader);
    EXPECT_EQ(idx->ntotal, 0);
}

// ---- IndexBinaryIVF runtime safety checks ----

TEST(ReadIndexDeserialize, BinaryIVFNullInvlistsSearch) {
    IndexBinaryIVF idx;
    idx.d = 16;
    idx.code_size = 2;
    idx.invlists = nullptr;
    idx.own_invlists = false;
    std::vector<int32_t> distances(1);
    std::vector<idx_t> labels(1);
    std::vector<uint8_t> query(2);
    EXPECT_THROW(
            {
                try {
                    idx.search(
                            1,
                            query.data(),
                            1,
                            distances.data(),
                            labels.data());
                } catch (const faiss::FaissException& e) {
                    EXPECT_NE(
                            std::string(e.what()).find("inverted lists"),
                            std::string::npos);
                    throw;
                }
            },
            faiss::FaissException);
}

// -----------------------------------------------------------------------
// Test: VectorTransform deserialization rejects d_in * d_out that would
// exceed the deserialization vector byte limit.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, VectorTransformDimExceedsByteLimit) {
    // VNrm format: fourcc + float(norm) + d_in + d_out + is_trained
    // d_in=1024, d_out=1024 => d_in*d_out = 1M floats = 4 MB.
    // Set byte limit to 1 MB so the check fails.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "VNrm");
    push_val<float>(buf, 2.0f);
    push_val<int>(buf, 1024);
    push_val<int>(buf, 1024);
    push_val<bool>(buf, true);

    auto old_limit = get_deserialization_vector_byte_limit();
    set_deserialization_vector_byte_limit(1 << 20); // 1 MB < 4 MB

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_THROW(read_VectorTransform_up(&reader), FaissException);

    set_deserialization_vector_byte_limit(old_limit);
}

// -----------------------------------------------------------------------
// Test: VectorTransform deserialization accepts d_in * d_out within
// the deserialization vector byte limit.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, VectorTransformDimWithinByteLimit) {
    // d_in=16, d_out=16 => d_in*d_out = 256 floats = 1 KB.
    // Set byte limit to 1 MB so the check passes.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "VNrm");
    push_val<float>(buf, 2.0f);
    push_val<int>(buf, 16);
    push_val<int>(buf, 16);
    push_val<bool>(buf, true);

    auto old_limit = get_deserialization_vector_byte_limit();
    set_deserialization_vector_byte_limit(1 << 20); // 1 MB >> 1 KB

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_VectorTransform_up(&reader));

    set_deserialization_vector_byte_limit(old_limit);
}

// -----------------------------------------------------------------------
// Test: VectorTransform deserialization rejects negative d_out.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, VectorTransformDOutNegative) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "VNrm");
    push_val<float>(buf, 2.0f);
    push_val<int>(buf, 16);
    push_val<int>(buf, -1);
    push_val<bool>(buf, true);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_THROW(read_VectorTransform_up(&reader), FaissException);
}

// -----------------------------------------------------------------------
// Test: VectorTransform deserialization rejects negative d_in.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, VectorTransformDInNegative) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "VNrm");
    push_val<float>(buf, 2.0f);
    push_val<int>(buf, -1);
    push_val<int>(buf, 16);
    push_val<bool>(buf, true);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_THROW(read_VectorTransform_up(&reader), FaissException);
}

// -----------------------------------------------------------------------
// Test: IndexIVFIndependentQuantizer deserialization rejects a VT whose
// d_in does not match the outer index dimension.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IwIQVtDinMismatch) {
    // Create a valid IndexIVFIndependentQuantizer and serialize it.
    const int d = 16;
    const int nlist = 4;
    auto quantizer = std::make_unique<IndexFlat>(d);
    auto ivf = std::make_unique<IndexIVFFlat>(quantizer.get(), d, nlist);
    ivf->own_fields = false;
    auto vt = std::make_unique<NormalizationTransform>(d);

    IndexIVFIndependentQuantizer indep(quantizer.get(), ivf.get(), vt.get());
    indep.own_fields = false;

    VectorIOWriter writer;
    write_index(&indep, &writer);

    // Locate the VNrm fourcc to find where VT dimensions are stored.
    // VNrm layout: fourcc(4) + norm(4) + d_in(4) + d_out(4) + is_trained(1)
    auto& data = writer.data;
    uint32_t vnrm_h;
    {
        const unsigned char s[4] = {'V', 'N', 'r', 'm'};
        vnrm_h = s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24);
    }
    size_t vnrm_pos = 0;
    bool found = false;
    for (size_t i = 0; i + 4 <= data.size(); ++i) {
        uint32_t val;
        memcpy(&val, &data[i], sizeof(val));
        if (val == vnrm_h) {
            vnrm_pos = i;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found);

    // Corrupt d_in to create mismatch with indep->d.
    size_t d_in_offset = vnrm_pos + 4 + 4; // after fourcc + norm
    auto corrupted = data;
    int bad_d_in = d + 1;
    memcpy(&corrupted[d_in_offset], &bad_d_in, sizeof(int));

    VectorIOReader reader;
    reader.data = corrupted;
    EXPECT_THROW(read_index(&reader), FaissException);
}

// -----------------------------------------------------------------------
// Test: IndexIVFIndependentQuantizer deserialization rejects a VT whose
// d_out does not match the inner IVF index dimension.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IwIQVtDoutMismatch) {
    const int d = 16;
    const int nlist = 4;
    auto quantizer = std::make_unique<IndexFlat>(d);
    auto ivf = std::make_unique<IndexIVFFlat>(quantizer.get(), d, nlist);
    ivf->own_fields = false;
    auto vt = std::make_unique<NormalizationTransform>(d);

    IndexIVFIndependentQuantizer indep(quantizer.get(), ivf.get(), vt.get());
    indep.own_fields = false;

    VectorIOWriter writer;
    write_index(&indep, &writer);

    auto& data = writer.data;
    uint32_t vnrm_h;
    {
        const unsigned char s[4] = {'V', 'N', 'r', 'm'};
        vnrm_h = s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24);
    }
    size_t vnrm_pos = 0;
    bool found = false;
    for (size_t i = 0; i + 4 <= data.size(); ++i) {
        uint32_t val;
        memcpy(&val, &data[i], sizeof(val));
        if (val == vnrm_h) {
            vnrm_pos = i;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found);

    // Corrupt d_out to create mismatch with index_ivf->d.
    size_t d_out_offset = vnrm_pos + 4 + 4 + 4; // after fourcc + norm + d_in
    auto corrupted = data;
    int bad_d_out = d + 1;
    memcpy(&corrupted[d_out_offset], &bad_d_out, sizeof(int));

    VectorIOReader reader;
    reader.data = corrupted;
    EXPECT_THROW(read_index(&reader), FaissException);
}

/// Helper: append a minimal valid serialized IndexPQFastScan ("IPfs").
/// Caller can override bbs and M2 to inject invalid values.
static std::vector<uint8_t> build_IndexPQFastScan_buf(
        int bbs = 32,
        size_t M2 = 2,
        int qbs = 0) {
    // PQ: d=4, M=2, nbits=4 → ksub=16, centroids size = d*ksub = 64 floats
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IPfs");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_pq(buf, /*d=*/4, /*M=*/2, /*nbits=*/4, std::vector<float>(64, 0.0f));
    push_val<int>(buf, 0);         // implem
    push_val<int>(buf, bbs);       // bbs
    push_val<int>(buf, qbs);       // qbs
    push_val<size_t>(buf, 0);      // ntotal2
    push_val<size_t>(buf, M2);     // M2
    push_vector<uint8_t>(buf, {}); // codes
    return buf;
}

/// Helper: append a minimal valid serialized
/// IndexResidualQuantizerFastScan ("IRfs").
/// Writes AdditiveQuantizer + ResidualQuantizer fields, then FastScan fields.
/// Caller can override FastScan M, ksub, bbs to inject invalid values.
static std::vector<uint8_t> build_AQFastScan_buf(
        size_t fastscan_M = 3,
        size_t fastscan_ksub = 16,
        int bbs = 32,
        int qbs = 0,
        size_t fastscan_M2 = 0) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IRfs");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);

    // AdditiveQuantizer fields:
    push_val<size_t>(buf, 4);      // d
    push_val<size_t>(buf, 1);      // M (AQ M, not FastScan M)
    push_vector<size_t>(buf, {4}); // nbits = [4]
    push_val<bool>(buf, false);    // is_trained
    push_vector<float>(buf, std::vector<float>(64, 0.0f)); // codebooks (d*16)
    push_val<int>(buf, 0);      // search_type = ST_decompress
    push_val<float>(buf, 0.0f); // norm_min
    push_val<float>(buf, 1.0f); // norm_max

    // ResidualQuantizer extras:
    push_val<int>(buf, 2048); // train_type = Skip_codebook_tables
    push_val<int>(buf, 1);    // max_beam_size

    // FastScan fields (IndexAdditiveQuantizerFastScan):
    push_val<int>(buf, 0);                // implem
    push_val<int>(buf, bbs);              // bbs
    push_val<int>(buf, qbs);              // qbs
    push_val<size_t>(buf, fastscan_M);    // M
    push_val<size_t>(buf, 4);             // nbits
    push_val<size_t>(buf, fastscan_ksub); // ksub
    push_val<size_t>(buf, 2);             // code_size
    push_val<size_t>(buf, 0);             // ntotal2
    // M2: use override if provided, otherwise roundup(M, 2)
    size_t M2 = fastscan_M2 ? fastscan_M2 : (fastscan_M + 1) & ~size_t{1};
    push_val<size_t>(buf, M2);
    push_val<bool>(buf, true);     // rescale_norm
    push_val<int>(buf, 1);         // norm_scale
    push_val<size_t>(buf, 48);     // max_train_points
    push_vector<uint8_t>(buf, {}); // codes

    return buf;
}

/// Helper: serialize a valid IndexRaBitQFastScan and return the raw bytes.
/// Then locate the bbs field and patch it to the given value.
static std::vector<uint8_t> build_RaBitQFastScan_buf(int bbs) {
    IndexRaBitQFastScan idx(4, METRIC_L2, 32, 1);

    // Serialize the valid index
    VectorIOWriter writer;
    write_index(&idx, &writer);

    auto buf = std::move(writer.data);

    // Locate bbs (int, value 32) in the serialized data.
    // Format: ... qb (uint8_t=8) then bbs (int=32).
    // Search for qb=8 (1 byte) followed by bbs=32 (4 bytes).
    uint8_t target_qb = 8;
    int target_bbs = 32;
    bool patched = false;
    for (size_t i = 0; i + sizeof(uint8_t) + sizeof(int) <= buf.size(); i++) {
        uint8_t val_qb;
        int val_bbs;
        memcpy(&val_qb, &buf[i], sizeof(uint8_t));
        memcpy(&val_bbs, &buf[i + sizeof(uint8_t)], sizeof(int));
        if (val_qb == target_qb && val_bbs == target_bbs) {
            memcpy(&buf[i + sizeof(uint8_t)], &bbs, sizeof(int));
            patched = true;
            break;
        }
    }
    EXPECT_TRUE(patched) << "Could not find bbs field in serialized data";
    return buf;
}

// -----------------------------------------------------------------------
// IndexPQFastScan deserialization: bbs=0 must be rejected.
// Without this check, search() would divide by zero in ntotal2/bbs.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexPQFastScanBbsZero) {
    auto buf = build_IndexPQFastScan_buf(/*bbs=*/0);
    expect_read_throws_with(buf, "invalid bbs");
}

// -----------------------------------------------------------------------
// IndexPQFastScan deserialization: bbs not a multiple of 32.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexPQFastScanBbsNotAligned) {
    auto buf = build_IndexPQFastScan_buf(/*bbs=*/33);
    expect_read_throws_with(buf, "invalid bbs");
}

// -----------------------------------------------------------------------
// IndexPQFastScan deserialization: M2 must equal roundup(M, 2).
// A corrupted file with M2=0 while M>0 causes compute_quantized_LUT
// to write M*ksub bytes into a buffer sized for M2*ksub=0 bytes.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexPQFastScanM2Zero) {
    auto buf = build_IndexPQFastScan_buf(/*bbs=*/32, /*M2=*/0);
    expect_read_throws_with(buf, "invalid M2");
}

// -----------------------------------------------------------------------
// IndexPQFastScan deserialization: M2 too small (1 < roundup(2, 2) = 2).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexPQFastScanM2TooSmall) {
    auto buf = build_IndexPQFastScan_buf(/*bbs=*/32, /*M2=*/1);
    expect_read_throws_with(buf, "invalid M2");
}

// -----------------------------------------------------------------------
// IndexPQFastScan deserialization: ksub * M2 overflow.
// M2 is read directly from the file and could be corrupted to a huge
// value that causes ksub * M2 to overflow size_t.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexPQFastScanKsubM2Overflow) {
    // M2=SIZE_MAX is now caught by M2 != roundup(M, 2) before overflow.
    auto buf = build_IndexPQFastScan_buf(
            /*bbs=*/32, /*M2=*/std::numeric_limits<size_t>::max());
    expect_read_throws_with(buf, "invalid M2");
}

// -----------------------------------------------------------------------
// IndexAdditiveQuantizerFastScan deserialization: M=0 must be rejected.
// Without this check, search() would crash in compute_float_LUT.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AQFastScanMZero) {
    auto buf = build_AQFastScan_buf(/*fastscan_M=*/0);
    expect_read_throws_with(buf, "invalid quantizer state");
}

// -----------------------------------------------------------------------
// IndexAdditiveQuantizerFastScan deserialization: ksub=0 must be rejected.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AQFastScanKsubZero) {
    auto buf = build_AQFastScan_buf(
            /*fastscan_M=*/3, /*fastscan_ksub=*/0);
    expect_read_throws_with(buf, "invalid quantizer state");
}

// -----------------------------------------------------------------------
// IndexAdditiveQuantizerFastScan deserialization: bbs=0 must be rejected.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AQFastScanBbsZero) {
    auto buf = build_AQFastScan_buf(
            /*fastscan_M=*/3, /*fastscan_ksub=*/16, /*bbs=*/0);
    expect_read_throws_with(buf, "invalid bbs");
}

// -----------------------------------------------------------------------
// IndexAdditiveQuantizerFastScan deserialization: M2 mismatch.
// M2=0 while M=3 causes compute_quantized_LUT to write out of bounds.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AQFastScanM2Mismatch) {
    auto buf = build_AQFastScan_buf(
            /*fastscan_M=*/3,
            /*fastscan_ksub=*/16,
            /*bbs=*/32,
            /*qbs=*/0,
            /*fastscan_M2=*/1);
    expect_read_throws_with(buf, "invalid M2");
}

// -----------------------------------------------------------------------
// IndexAdditiveQuantizerFastScan deserialization: ksub * M overflow.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AQFastScanKsubMOverflow) {
    auto buf = build_AQFastScan_buf(
            /*fastscan_M=*/std::numeric_limits<size_t>::max(),
            /*fastscan_ksub=*/16,
            /*bbs=*/32);
    expect_read_throws_with(buf, "overflow");
}

// -----------------------------------------------------------------------
// IndexRaBitQFastScan deserialization: bbs=0 must be rejected.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, RaBitQFastScanBbsZero) {
    auto buf = build_RaBitQFastScan_buf(/*bbs=*/0);
    expect_read_throws_with(buf, "invalid bbs");
}

// -----------------------------------------------------------------------
// IndexRaBitQFastScan deserialization: bbs not a multiple of 32.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, RaBitQFastScanBbsNotAligned) {
    auto buf = build_RaBitQFastScan_buf(/*bbs=*/17);
    expect_read_throws_with(buf, "invalid bbs");
}

// -----------------------------------------------------------------------
// Test: read_AdditiveQuantizer rejects codebooks that are too small for
// d * total_codebook_size.  Previously this was only caught at
// compute_LUT() time; now deserialization itself must throw.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AdditiveQuantizerCodebooksTooSmall) {
    // Build an IndexResidualQuantizer ("IxRq") with d=4, M=1, nbits={8}.
    // Write AdditiveQuantizer fields manually with empty codebooks so that
    // the deserialization check fires (codebooks.size()=0 < d*256=1024).
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxRq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    // AdditiveQuantizer fields (inline, not via push_additive_quantizer
    // which now writes correctly-sized codebooks):
    push_val<size_t>(buf, 4);      // d
    push_val<size_t>(buf, 1);      // M
    push_vector<size_t>(buf, {8}); // nbits
    push_val<bool>(buf, true);     // is_trained
    push_vector<float>(buf, {});   // codebooks (empty — triggers check)
    push_val<int>(buf, 0);         // search_type = ST_decompress
    push_val<float>(buf, 0.0f);    // norm_min
    push_val<float>(buf, 1.0f);    // norm_max
    // ResidualQuantizer fields:
    push_val<int>(buf, 2048); // train_type = Skip_codebook_tables
    push_val<int>(buf, 1);    // max_beam_size
    // IndexResidualQuantizer fields:
    push_val<size_t>(buf, 1);      // code_size
    push_vector<uint8_t>(buf, {}); // codes (empty, matches ntotal=0)

    expect_read_throws_with(buf, "not a positive multiple");
}

// -----------------------------------------------------------------------
// Test: AdditiveQuantizer with d=0 is rejected during deserialization.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AdditiveQuantizerDZero) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxRq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    // AdditiveQuantizer fields with d=0:
    push_val<size_t>(buf, 0);      // d = 0 (invalid)
    push_val<size_t>(buf, 1);      // M
    push_vector<size_t>(buf, {8}); // nbits
    push_val<bool>(buf, true);     // is_trained
    push_vector<float>(buf, {});   // codebooks
    push_val<int>(buf, 0);         // search_type = ST_decompress
    push_val<float>(buf, 0.0f);    // norm_min
    push_val<float>(buf, 1.0f);    // norm_max

    expect_read_throws_with(buf, "invalid AdditiveQuantizer d");
}

// -----------------------------------------------------------------------
// Test: AdditiveQuantizer with codebooks size not a multiple of
// total_codebook_size is rejected.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, AdditiveQuantizerCodebooksNotMultiple) {
    // d=4, M=1, nbits={8} → total_codebook_size=256.
    // codebooks should be a multiple of 256 floats.  Provide 257 floats.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxRq");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<size_t>(buf, 4);                               // d
    push_val<size_t>(buf, 1);                               // M
    push_vector<size_t>(buf, {8});                          // nbits
    push_val<bool>(buf, true);                              // is_trained
    push_vector<float>(buf, std::vector<float>(257, 0.0f)); // not a multiple
    push_val<int>(buf, 0);      // search_type = ST_decompress
    push_val<float>(buf, 0.0f); // norm_min
    push_val<float>(buf, 1.0f); // norm_max

    expect_read_throws_with(buf, "not a positive multiple");
}

// -----------------------------------------------------------------------
// Test: ProductAdditiveQuantizer with d not divisible by nsplits throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ProductAdditiveQuantizerDNotDivisible) {
    // d=5, nsplits=2 → d % nsplits != 0
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPR");
    push_index_header(buf, /*d=*/5, /*ntotal=*/0);
    // AdditiveQuantizer fields: d=5, M=2, nbits={4,4}
    // For a PAQ with nsplits=2, codebook_d should be d/nsplits, but since
    // d is not divisible we can just write codebooks sized for d
    // (the divisibility check fires before the codebooks size check).
    push_val<size_t>(buf, 5);                                  // d
    push_val<size_t>(buf, 2);                                  // M
    push_vector<size_t>(buf, {4, 4});                          // nbits
    push_val<bool>(buf, true);                                 // is_trained
    push_vector<float>(buf, std::vector<float>(5 * 32, 0.0f)); // codebooks
    push_val<int>(buf, 0);      // search_type = ST_decompress
    push_val<float>(buf, 0.0f); // norm_min
    push_val<float>(buf, 1.0f); // norm_max
    // ProductAdditiveQuantizer:
    push_val<size_t>(buf, 2); // nsplits = 2

    expect_read_throws_with(buf, "not divisible by nsplits");
}

// -----------------------------------------------------------------------
// Test: ProductResidualQuantizer sub-quantizer d mismatch throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ProductResidualQuantizerSubDMismatch) {
    // d=8, nsplits=2 → d_sub should be 4, but sub-quantizer has d=6
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPR");
    push_index_header(buf, /*d=*/8, /*ntotal=*/0);
    // ProductAdditiveQuantizer: AQ(d=8) + nsplits=2 → codebook_d = 4
    push_additive_quantizer(
            buf, /*d=*/8, /*M=*/2, /*nbits=*/{4, 4}, /*codebook_d=*/4);
    push_val<size_t>(buf, 2); // nsplits = 2
    // First sub-quantizer: d=6 instead of expected d_sub=4
    push_residual_quantizer(buf, /*d=*/6, /*M=*/1, /*nbits=*/{4});
    // (second sub-quantizer not needed — should throw on first)

    expect_read_throws_with(buf, "sub-quantizer");
}

// -----------------------------------------------------------------------
// Test: ProductLocalSearchQuantizer sub-quantizer d mismatch throws.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ProductLocalSearchQuantizerSubDMismatch) {
    // d=8, nsplits=2 → d_sub should be 4, but sub-quantizer has d=6
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPL");
    push_index_header(buf, /*d=*/8, /*ntotal=*/0);
    // ProductAdditiveQuantizer: AQ(d=8) + nsplits=2 → codebook_d = 4
    push_additive_quantizer(
            buf, /*d=*/8, /*M=*/2, /*nbits=*/{4, 4}, /*codebook_d=*/4);
    push_val<size_t>(buf, 2); // nsplits = 2
    // First sub-quantizer: d=6 instead of expected d_sub=4
    push_local_search_quantizer(buf, /*d=*/6, /*M=*/1, /*nbits=*/{4});
    // (second sub-quantizer not needed — should throw on first)

    expect_read_throws_with(buf, "sub-quantizer");
}

// -----------------------------------------------------------------------
// Test: ProductResidualQuantizer with nsplits > 1 round-trips correctly
// (codebooks sized for d_sub, not d).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ProductResidualQuantizerMultiSplitValid) {
    // d=8, nsplits=2, each sub-RQ has d_sub=4, M=1, nbits={4}
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPR");
    push_index_header(buf, /*d=*/8, /*ntotal=*/0);
    // ProductAdditiveQuantizer: codebook_d = d/nsplits = 4
    push_additive_quantizer(
            buf, /*d=*/8, /*M=*/2, /*nbits=*/{4, 4}, /*codebook_d=*/4);
    push_val<size_t>(buf, 2); // nsplits = 2
    // Two sub-RQs with d_sub=4
    push_residual_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{4});
    push_residual_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{4});
    // code_size = 1 (M=2, nbits={4,4} → 8 bits → 1 byte)
    push_val<size_t>(buf, 1);      // code_size
    push_vector<uint8_t>(buf, {}); // codes (empty, ntotal=0)

    VectorIOReader reader;
    reader.data = buf;
    auto idx = read_index_up(&reader);
    EXPECT_EQ(idx->ntotal, 0);
    EXPECT_EQ(idx->d, 8);
}

// -----------------------------------------------------------------------
// Test: ProductLocalSearchQuantizer with nsplits > 1 round-trips correctly
// (codebooks sized for d_sub, not d).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, ProductLocalSearchQuantizerMultiSplitValid) {
    // d=8, nsplits=2, each sub-LSQ has d_sub=4, M=1, nbits={4}
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxPL");
    push_index_header(buf, /*d=*/8, /*ntotal=*/0);
    // ProductAdditiveQuantizer: codebook_d = d/nsplits = 4
    push_additive_quantizer(
            buf, /*d=*/8, /*M=*/2, /*nbits=*/{4, 4}, /*codebook_d=*/4);
    push_val<size_t>(buf, 2); // nsplits = 2
    // Two sub-LSQs with d_sub=4
    push_local_search_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{4});
    push_local_search_quantizer(buf, /*d=*/4, /*M=*/1, /*nbits=*/{4});
    // code_size = 1
    push_val<size_t>(buf, 1);      // code_size
    push_vector<uint8_t>(buf, {}); // codes (empty, ntotal=0)

    VectorIOReader reader;
    reader.data = buf;
    auto idx = read_index_up(&reader);
    EXPECT_EQ(idx->ntotal, 0);
    EXPECT_EQ(idx->d, 8);
}

// -----------------------------------------------------------------------
// Negative qbs must be rejected at deserialization time to prevent
// infinite loops in pq4_qbs_to_nq (arithmetic right shift preserves
// the sign bit, so the loop never terminates for negative values).
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, IndexPQFastScanNegativeQbs) {
    auto buf = build_IndexPQFastScan_buf(/*bbs=*/32, /*M2=*/2, /*qbs=*/-1);
    expect_read_throws_with(buf, "qbs must be non-negative");
}

TEST(ReadIndexDeserialize, AQFastScanNegativeQbs) {
    auto buf = build_AQFastScan_buf(
            /*fastscan_M=*/3,
            /*fastscan_ksub=*/16,
            /*bbs=*/32,
            /*qbs=*/-1);
    expect_read_throws_with(buf, "qbs must be non-negative");
}

/// Helper: serialize a valid IndexIVFResidualQuantizerFastScan, then patch
/// the qbs field to a given value. The qbs field (int, default 0) immediately
/// follows bbs (int, value 32) in the serialized IVRf format.
static std::vector<uint8_t> build_IVFAQFastScan_buf(int qbs) {
    IndexFlat coarse_quantizer(4, METRIC_L2);
    IndexIVFResidualQuantizerFastScan idx(
            &coarse_quantizer, /*d=*/4, /*nlist=*/1, /*M=*/1, /*nbits=*/4);

    // Train the index so serialization produces valid codebooks.
    std::vector<float> train_data(4 * 64);
    for (size_t i = 0; i < train_data.size(); i++) {
        train_data[i] = static_cast<float>(i);
    }
    idx.train(64, train_data.data());

    // Set a distinctive qbs value so the byte pattern is unique.
    int sentinel_qbs = 0x1234;
    idx.qbs = sentinel_qbs;

    VectorIOWriter writer;
    write_index(&idx, &writer);

    auto buf = std::move(writer.data);

    // Locate qbs (int, value sentinel_qbs) which follows bbs (int, value 32).
    int target_bbs = 32;
    int target_qbs = sentinel_qbs;
    bool patched = false;
    for (size_t i = 0; i + 2 * sizeof(int) <= buf.size(); i++) {
        int val_bbs, val_qbs;
        memcpy(&val_bbs, &buf[i], sizeof(int));
        memcpy(&val_qbs, &buf[i + sizeof(int)], sizeof(int));
        if (val_bbs == target_bbs && val_qbs == target_qbs) {
            memcpy(&buf[i + sizeof(int)], &qbs, sizeof(int));
            patched = true;
            break;
        }
    }
    EXPECT_TRUE(patched) << "Could not find qbs field in serialized data";
    return buf;
}

TEST(ReadIndexDeserialize, IVFAQFastScanNegativeQbs) {
    auto buf = build_IVFAQFastScan_buf(/*qbs=*/-1);
    expect_read_throws_with(buf, "qbs must be non-negative");
}

// -----------------------------------------------------------------------
// Test: ScalarQuantizer with empty trained vector for qtypes that require
// training data must be rejected at deserialization time.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, SQEmptyTrainedNonUniform) {
    // QT_8bit (NON_UNIFORM) expects trained.size() == 2*d == 8.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxSQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<int>(buf, ScalarQuantizer::QT_8bit);
    push_val<int>(buf, 0);         // rangestat
    push_val<float>(buf, 0.0f);    // rangestat_arg
    push_val<size_t>(buf, 4);      // d
    push_val<size_t>(buf, 1);      // code_size
    push_vector<float>(buf, {});   // empty trained
    push_vector<uint8_t>(buf, {}); // codes (ntotal=0)

    expect_read_throws_with(buf, "ScalarQuantizer trained size");
}

TEST(ReadIndexDeserialize, SQEmptyTrainedUniform) {
    // QT_8bit_uniform (UNIFORM) expects trained.size() == 2.
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxSQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<int>(buf, ScalarQuantizer::QT_8bit_uniform);
    push_val<int>(buf, 0);         // rangestat
    push_val<float>(buf, 0.0f);    // rangestat_arg
    push_val<size_t>(buf, 4);      // d
    push_val<size_t>(buf, 1);      // code_size
    push_vector<float>(buf, {});   // empty trained
    push_vector<uint8_t>(buf, {}); // codes (ntotal=0)

    expect_read_throws_with(buf, "ScalarQuantizer trained size");
}

// -----------------------------------------------------------------------
// Test: ScalarQuantizer d must match the index header d.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, SQDimensionMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxSQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_val<int>(buf, ScalarQuantizer::QT_8bit);
    push_val<int>(buf, 0);      // rangestat
    push_val<float>(buf, 0.0f); // rangestat_arg
    push_val<size_t>(buf, 8);   // d = 8, mismatches header d = 4
    push_val<size_t>(buf, 1);   // code_size
    push_vector<float>(buf, std::vector<float>(16, 0.0f)); // trained
    push_vector<uint8_t>(buf, {});                         // codes (ntotal=0)

    expect_read_throws_with(buf, "ScalarQuantizer d");
}

// -----------------------------------------------------------------------
// Test: ScalarQuantizer with training data but is_trained=false in header.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, SQIsTrainedMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxSQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0, /*is_trained=*/false);
    push_val<int>(buf, ScalarQuantizer::QT_8bit);
    push_val<int>(buf, 0);      // rangestat
    push_val<float>(buf, 0.0f); // rangestat_arg
    push_val<size_t>(buf, 4);   // d
    push_val<size_t>(buf, 1);   // code_size
    // Provide correctly-sized trained data (2*d=8 floats)
    push_vector<float>(buf, std::vector<float>(8, 0.0f));
    push_vector<uint8_t>(buf, {}); // codes (ntotal=0)

    expect_read_throws_with(buf, "is_trained");
}

// -----------------------------------------------------------------------
// Test: Untrained ScalarQuantizer (is_trained=false, empty trained) must
// deserialize successfully — this is a legitimate untrained index.
// -----------------------------------------------------------------------
TEST(ReadIndexDeserialize, SQUntrainedEmptyTrainedAccepted) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxSQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0, /*is_trained=*/false);
    push_val<int>(buf, ScalarQuantizer::QT_8bit);
    push_val<int>(buf, 0);         // rangestat
    push_val<float>(buf, 0.0f);    // rangestat_arg
    push_val<size_t>(buf, 4);      // d
    push_val<size_t>(buf, 1);      // code_size
    push_vector<float>(buf, {});   // empty trained (untrained)
    push_vector<uint8_t>(buf, {}); // codes (ntotal=0)

    VectorIOReader reader;
    reader.data = buf;
    auto idx = read_index_up(&reader);
    EXPECT_FALSE(idx->is_trained);
}

// Test: IndexResidualQuantizer rejects AQ dimension != index dimension.
TEST(ReadIndexDeserialize, IndexResidualQuantizerAQDimensionMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxRq");
    // Index header says d=4, but AQ will have d=8 → mismatch.
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_residual_quantizer(buf, /*d=*/8, /*M=*/1, /*nbits=*/{4});
    push_val<size_t>(buf, 1);      // code_size
    push_vector<uint8_t>(buf, {}); // codes

    expect_read_throws_with(buf, "does not match index d");
}

// Test: IndexLocalSearchQuantizer rejects AQ dimension != index dimension.
TEST(ReadIndexDeserialize, IndexLocalSearchQuantizerAQDimensionMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLS");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_local_search_quantizer(buf, /*d=*/8, /*M=*/1, /*nbits=*/{4});
    push_val<size_t>(buf, 1);      // code_size
    push_vector<uint8_t>(buf, {}); // codes

    expect_read_throws_with(buf, "does not match index d");
}

// Test: ResidualCoarseQuantizer rejects AQ dimension != index dimension.
TEST(ReadIndexDeserialize, ResidualCoarseQuantizerAQDimensionMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ImRQ");
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_residual_quantizer(buf, /*d=*/8, /*M=*/1, /*nbits=*/{4});
    push_val<float>(buf, -1.0f); // beam_factor

    expect_read_throws_with(buf, "does not match index d");
}

// Test: IndexResidualQuantizerFastScan rejects AQ dimension != index dimension.
TEST(ReadIndexDeserialize, IndexRQFastScanAQDimensionMismatch) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IRfs");
    // Index header says d=4, but AQ will have d=8 → mismatch.
    push_index_header(buf, /*d=*/4, /*ntotal=*/0);
    push_residual_quantizer(buf, /*d=*/8, /*M=*/1, /*nbits=*/{4});
    // FastScan fields won't be reached due to early validation.
    // But include a few just in case:
    push_val<int>(buf, 0);  // implem
    push_val<int>(buf, 32); // bbs
    push_val<int>(buf, 0);  // qbs

    expect_read_throws_with(buf, "does not match index d");
}

// ============================================================
// SVS fourcc rejection / deserialization safety (Group F: T262015608)
// ============================================================

#ifdef FAISS_ENABLE_SVS

#include <faiss/svs/IndexSVSVamana.h>

// An invalid storage_kind value should be rejected at deserialization time
// with a FaissException, not abort via FAISS_ASSERT in to_svs_storage_kind().
TEST(ReadIndexDeserialize, SVSVamanaInvalidStorageKind) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ISVD");
    push_index_header(buf, 8, 0);
    push_val<size_t>(buf, 32);  // graph_max_degree
    push_val<float>(buf, 1.2f); // alpha
    push_val<size_t>(buf, 10);  // search_window_size
    push_val<size_t>(buf, 10);  // search_buffer_capacity
    push_val<size_t>(buf, 64);  // construction_window_size
    push_val<size_t>(buf, 750); // max_candidate_pool_size
    push_val<size_t>(buf, 28);  // prune_to
    push_val<bool>(buf, false); // use_full_search_history
    push_val<int>(
            buf,
            static_cast<int>(SVS_count)); // storage_kind — first invalid value
    push_val<bool>(buf, true);            // initialized

    expect_read_throws_with(buf, "storage_kind");
}

// When SVS is enabled, deserializing an SVS Vamana index with invalid SVS
// stream data should throw a FaissException (from the SVS runtime load
// failure) rather than crashing with a null-pointer dereference.
// Previously, deserialize_impl called impl->load() on a null impl pointer.
TEST(ReadIndexDeserialize, SVSVamanaInvalidStreamThrows) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ISVD");
    push_index_header(buf, 8, 0);
    // SVS Vamana deserialization fields:
    push_val<size_t>(buf, 32);  // graph_max_degree
    push_val<float>(buf, 1.2f); // alpha
    push_val<size_t>(buf, 10);  // search_window_size
    push_val<size_t>(buf, 10);  // search_buffer_capacity
    push_val<size_t>(buf, 64);  // construction_window_size
    push_val<size_t>(buf, 750); // max_candidate_pool_size
    push_val<size_t>(buf, 28);  // prune_to
    push_val<bool>(buf, false); // use_full_search_history
    push_val<int>(buf, 0);      // storage_kind (SVS_Float16)
    push_val<bool>(buf, true); // initialized = true → triggers deserialize_impl
    // Provide garbage SVS stream data — load should fail gracefully.
    for (int i = 0; i < 256; i++) {
        push_val<uint8_t>(buf, 0);
    }

    // Should throw from SVS runtime load failure, NOT crash with SIGSEGV.
    EXPECT_THROW(
            {
                auto reader = faiss::VectorIOReader();
                reader.data = buf;
                faiss::read_index(&reader);
            },
            faiss::FaissException);
}

// Same test for SVS Flat index.
TEST(ReadIndexDeserialize, SVSFlatInvalidStreamThrows) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ISVF");
    push_index_header(buf, 8, 0);
    push_val<bool>(buf, true); // initialized = true → triggers deserialize_impl
    // Provide garbage SVS stream data.
    for (int i = 0; i < 256; i++) {
        push_val<uint8_t>(buf, 0);
    }

    EXPECT_THROW(
            {
                auto reader = faiss::VectorIOReader();
                reader.data = buf;
                faiss::read_index(&reader);
            },
            faiss::FaissException);
}

// -----------------------------------------------------------------------
// Tests: IndexRefine / IndexRefinePanorama k_factor validation.
//
// Format "IxRF" (IndexRefineFlat) and "IxRP" (IndexRefinePanorama):
//   fourcc + index_header + base_index + refine_index + k_factor
//
// k_factor must be finite and in [1, 1000].
// -----------------------------------------------------------------------

// Helper: build a minimal IxRF or IxRP payload with the given k_factor.
static void push_index_refine(
        std::vector<uint8_t>& buf,
        const char fourcc_str[4],
        float k_factor) {
    int d = 4;
    push_fourcc(buf, fourcc_str);
    push_index_header(buf, d, /*ntotal=*/0);
    push_minimal_flat(buf, d); // base_index
    push_minimal_flat(buf, d); // refine_index
    push_val<float>(buf, k_factor);
}

TEST(ReadIndexDeserialize, IndexRefineKFactorValid) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRF", 1.0f);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

TEST(ReadIndexDeserialize, IndexRefineKFactorMax) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRF", 1000.0f);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

TEST(ReadIndexDeserialize, IndexRefineKFactorTooLarge) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRF", 1001.0f);
    expect_read_throws_with(buf, "k_factor");
}

TEST(ReadIndexDeserialize, IndexRefineKFactorNegative) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRF", -1.0f);
    expect_read_throws_with(buf, "k_factor");
}

TEST(ReadIndexDeserialize, IndexRefineKFactorZero) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRF", 0.0f);
    expect_read_throws_with(buf, "k_factor");
}

TEST(ReadIndexDeserialize, IndexRefineKFactorInfinity) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRF", INFINITY);
    expect_read_throws_with(buf, "k_factor");
}

TEST(ReadIndexDeserialize, IndexRefineKFactorNaN) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRF", NAN);
    expect_read_throws_with(buf, "k_factor");
}

TEST(ReadIndexDeserialize, IndexRefinePanoramaKFactorTooLarge) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRP", 1e10f);
    expect_read_throws_with(buf, "k_factor");
}

TEST(ReadIndexDeserialize, IndexRefinePanoramaKFactorValid) {
    std::vector<uint8_t> buf;
    push_index_refine(buf, "IxRP", 4.0f);

    VectorIOReader reader;
    reader.data = buf;
    EXPECT_NO_THROW(read_index_up(&reader));
}

// -----------------------------------------------------------------------
// Tests: IndexIVFPQR k_factor validation via round-trip.
//
// Create a real IndexIVFPQR, serialize it, patch the k_factor bytes in
// the serialized blob, and verify deserialization rejects invalid values.
// -----------------------------------------------------------------------

// Helper: serialize an index to a byte vector.
static std::vector<uint8_t> serialize_index(const Index* idx) {
    VectorIOWriter writer;
    write_index(idx, &writer);
    return writer.data;
}

// Helper: find and patch a float value in a byte buffer.
// Searches backwards from the end (k_factor is the last field written
// for IVFPQR).
static void patch_last_float(std::vector<uint8_t>& buf, float new_val) {
    size_t offset = buf.size() - sizeof(float);
    std::memcpy(buf.data() + offset, &new_val, sizeof(float));
}

// Helper: create a trained IndexIVFPQR, serialize it, return the bytes.
// Uses nbits=4 (16 centroids) so training succeeds with few vectors.
static std::vector<uint8_t> make_ivfpqr_bytes(float k_factor) {
    int d = 8;
    IndexFlatL2 quantizer(d);
    IndexIVFPQR ivfpqr(
            &quantizer,
            d,
            /*nlist=*/1,
            /*M=*/2,
            /*nbits_per_idx=*/4,
            /*M_refine=*/2,
            /*nbits_per_idx_refine=*/4);

    int ntrain = 64;
    std::vector<float> train_data(d * ntrain, 0.0f);
    for (size_t i = 0; i < train_data.size(); i++) {
        train_data[i] = float(i) / float(train_data.size());
    }
    ivfpqr.train(ntrain, train_data.data());
    ivfpqr.k_factor = k_factor;
    return serialize_index(&ivfpqr);
}

TEST(ReadIndexDeserialize, IndexIVFPQRKFactorTooLarge) {
    auto buf = make_ivfpqr_bytes(4.0f);
    patch_last_float(buf, 1e10f);
    expect_read_throws_with(buf, "k_factor");
}

TEST(ReadIndexDeserialize, IndexIVFPQRKFactorNegative) {
    auto buf = make_ivfpqr_bytes(4.0f);
    patch_last_float(buf, -1.0f);
    expect_read_throws_with(buf, "k_factor");
}

TEST(ReadIndexDeserialize, IndexIVFPQRKFactorValid) {
    auto buf = make_ivfpqr_bytes(64.0f); // AutoTune max

    VectorIOReader reader;
    reader.data = buf;
    auto idx = read_index_up(&reader);
    auto* result = dynamic_cast<IndexIVFPQR*>(idx.get());
    ASSERT_NE(result, nullptr);
    EXPECT_FLOAT_EQ(result->k_factor, 64.0f);
}

#else // !FAISS_ENABLE_SVS

// When SVS is not enabled, attempting to read an index with an SVS fourcc
// should fail with an "unknown fourcc" error rather than crashing.
TEST(ReadIndexDeserialize, SVSVamanaFourccRejected) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "ISVD");
    push_index_header(buf, 8, 0);
    for (int i = 0; i < 128; i++) {
        push_val<uint8_t>(buf, 0);
    }

    expect_read_throws_with(buf, "fourcc");
}

#endif // FAISS_ENABLE_SVS
