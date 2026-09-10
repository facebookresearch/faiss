/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/simd_levels.h>

#include "test_rabitq_simd_util.h"

using faiss::SIMDLevel;

// Random qb-bit-per-dimension query codes (one byte each, value in [0, 2^qb)).
static std::vector<uint8_t> random_codes(size_t d, size_t qb, uint32_t seed) {
    std::mt19937 rng(seed);
    const uint8_t code_mask = static_cast<uint8_t>((1u << qb) - 1);
    std::vector<uint8_t> q(d);
    for (size_t i = 0; i < d; i++) {
        q[i] = static_cast<uint8_t>(rng()) & code_mask;
    }
    return q;
}

static std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<uint8_t>(rng());
    }
    return v;
}

// 32-d chunks and chunk boundaries.
static const std::vector<size_t> kDims = {
        1,
        8,
        16,
        31,
        32,
        33,
        100,
        128,
        255,
        256,
        257,
        384,
        512,
        768,
        1024,
        2048};

// Note: scalar kernel's own correctness is covered end-to-end by
// tests/test_rabitq.py. This target is x86-only (see BUCK).
TEST(RaBitQRearrangeBitPlanes, Avx2MatchesScalar) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }

    for (size_t d : kDims) {
        for (size_t qb = 1; qb <= 8; qb++) {
            const auto q = random_codes(d, qb, 10996);
            const size_t out_bytes = ((d + 7) / 8) * qb;

            std::vector<uint8_t> scalar(out_bytes);
            faiss::rabitq::rearrange_bit_planes<SIMDLevel::NONE>(
                    q.data(), d, qb, scalar.data());

            std::vector<uint8_t> avx2(out_bytes);
            faiss::rabitq::rearrange_bit_planes<SIMDLevel::AVX2>(
                    q.data(), d, qb, avx2.data());

            EXPECT_EQ(avx2, scalar) << "d=" << d << " qb=" << qb;
        }
    }
}

TEST(RaBitQQuantization, ZeroCenteredQueryPreservesCorrectionScale) {
    constexpr size_t d = 64;
    std::vector<float> query(d, 0.0f);
    std::vector<float> rotated_query;
    std::vector<uint8_t> quantized_query;

    const auto factors = faiss::rabitq_utils::compute_query_factors(
            query.data(),
            d,
            nullptr,
            8,
            true,
            faiss::METRIC_L2,
            rotated_query,
            quantized_query);

    EXPECT_EQ(factors.int_dot_scale, 0.0f);
    EXPECT_EQ(quantized_query, std::vector<uint8_t>(d, 0));
}

TEST(RaBitQQuantization, Avx2MinmaxMatchesScalar) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    float scalar_min = 1.0f;
    float scalar_max = 2.0f;
    float avx2_min = scalar_min;
    float avx2_max = scalar_max;
    faiss::rabitq::minmax_values<SIMDLevel::NONE>(
            nullptr, 0, scalar_min, scalar_max);
    faiss::rabitq::minmax_values<SIMDLevel::AVX2>(
            nullptr, 0, avx2_min, avx2_max);
    EXPECT_EQ(avx2_min, scalar_min);
    EXPECT_EQ(avx2_max, scalar_max);

    for (size_t d : {1, 7, 8, 9, 15, 16, 17, 255}) {
        std::vector<float> values(d);
        for (float& value : values) {
            value = dist(rng);
        }

        faiss::rabitq::minmax_values<SIMDLevel::NONE>(
                values.data(), d, scalar_min, scalar_max);
        faiss::rabitq::minmax_values<SIMDLevel::AVX2>(
                values.data(), d, avx2_min, avx2_max);
        EXPECT_EQ(avx2_min, scalar_min) << "d=" << d;
        EXPECT_EQ(avx2_max, scalar_max) << "d=" << d;
    }
}

TEST(RaBitQQuantization, Avx2LargeCenteredAccumulationMatchesScalar) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }

    constexpr size_t d = 40000;
    constexpr uint8_t max_code = 255;
    std::vector<float> values(d);
    for (size_t i = 0; i < d; i++) {
        values[i] = (i & 1) ? 1.0f : 0.0f;
    }

    std::vector<uint8_t> scalar_codes(d);
    size_t scalar_sum = 0;
    int64_t scalar_sum2 = 0;
    faiss::rabitq::quantize_query_values<SIMDLevel::NONE>(
            values.data(),
            d,
            0.0f,
            255.0f,
            max_code,
            true,
            scalar_codes.data(),
            scalar_sum,
            scalar_sum2);

    std::vector<uint8_t> avx2_codes(d);
    size_t avx2_sum = 0;
    int64_t avx2_sum2 = 0;
    faiss::rabitq::quantize_query_values<SIMDLevel::AVX2>(
            values.data(),
            d,
            0.0f,
            255.0f,
            max_code,
            true,
            avx2_codes.data(),
            avx2_sum,
            avx2_sum2);

    EXPECT_EQ(avx2_codes, scalar_codes);
    EXPECT_EQ(avx2_sum, scalar_sum);
    EXPECT_EQ(avx2_sum2, scalar_sum2);
    EXPECT_EQ(avx2_sum2, int64_t(d) * max_code * max_code);
}

TEST(RaBitQQuantization, Avx2MatchesScalarAcrossCodeRange) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }
    faiss_test::check_quantization_matches_scalar<SIMDLevel::AVX2>();
}

TEST(RaBitQBitwiseAndDotProductWithPopcount, Avx2MatchesScalar) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }

    for (size_t d : kDims) {
        for (size_t qb = 1; qb <= 8; qb++) {
            const size_t size = (d + 7) / 8;
            const auto data = random_bytes(size, 19717);
            const auto q = random_bytes(size * qb, 51691);

            // Compare against the scalar reference (independent oracle) rather
            // than the same-level separate kernels.
            const uint64_t expected_dot =
                    faiss::rabitq::bitwise_and_dot_product<SIMDLevel::NONE>(
                            q.data(), data.data(), size, qb);
            const uint64_t expected_pop =
                    faiss::rabitq::popcount<SIMDLevel::NONE>(data.data(), size);
            const auto fused =
                    faiss::rabitq::bitwise_and_dot_product_with_popcount<
                            SIMDLevel::AVX2>(q.data(), data.data(), size, qb);

            EXPECT_EQ(fused.dot_product, expected_dot)
                    << "d=" << d << " qb=" << qb;
            EXPECT_EQ(fused.popcount, expected_pop)
                    << "d=" << d << " qb=" << qb;
        }
    }
}
