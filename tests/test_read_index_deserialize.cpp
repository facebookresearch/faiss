/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

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
