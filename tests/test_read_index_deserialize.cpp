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
