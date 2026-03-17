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
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissException.h>
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
    push_vector<float>(buf, {});   // codebooks (empty)
    push_val<int>(buf, 0);         // search_type = ST_decompress
    push_val<float>(buf, 0.0f);    // norm_min
    push_val<float>(buf, 1.0f);    // norm_max
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
