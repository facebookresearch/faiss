/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * RaBitQ kernels that a static build does not compile, against the scalar
 * reference.
 *
 * A static x86 build holds AVX2 kernels only, so the AVX512 and
 * AVX512_VPOPCNT specializations exist in a dynamic dispatch build alone.
 * test_rabitq_simd.cpp builds in both modes and cannot reference them, so
 * they are tested here instead. The target is dynamic dispatch only.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/simd_levels.h>

#include "test_rabitq_simd_util.h"

using faiss::SIMDLevel;

namespace {

std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<uint8_t>(rng());
    }
    return v;
}

// Widths on either side of every register boundary the kernels step over.
const std::vector<size_t> kDims = {
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

} // namespace

TEST(RaBitQVpopcnt, BitwiseKernelsMatchScalarAcrossTails) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                SIMDLevel::AVX512_VPOPCNT)) {
        GTEST_SKIP() << "AVX512_VPOPCNT is not available on this CPU";
    }

    for (size_t d : kDims) {
        for (size_t qb = 1; qb <= 8; qb++) {
            const size_t size = (d + 7) / 8;
            const auto data = random_bytes(size, 19717);
            const auto q = random_bytes(size * qb, 51691);

            const uint64_t expected_and =
                    faiss::rabitq::bitwise_and_dot_product<SIMDLevel::NONE>(
                            q.data(), data.data(), size, qb);
            const uint64_t expected_xor =
                    faiss::rabitq::bitwise_xor_dot_product<SIMDLevel::NONE>(
                            q.data(), data.data(), size, qb);
            const uint64_t expected_pop =
                    faiss::rabitq::popcount<SIMDLevel::NONE>(data.data(), size);

            const uint64_t actual_and = faiss::rabitq::bitwise_and_dot_product<
                    SIMDLevel::AVX512_VPOPCNT>(q.data(), data.data(), size, qb);
            const uint64_t actual_xor = faiss::rabitq::bitwise_xor_dot_product<
                    SIMDLevel::AVX512_VPOPCNT>(q.data(), data.data(), size, qb);
            const uint64_t actual_pop =
                    faiss::rabitq::popcount<SIMDLevel::AVX512_VPOPCNT>(
                            data.data(), size);
            const auto actual_fused =
                    faiss::rabitq::bitwise_and_dot_product_with_popcount<
                            SIMDLevel::AVX512_VPOPCNT>(
                            q.data(), data.data(), size, qb);

            EXPECT_EQ(actual_and, expected_and) << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_xor, expected_xor) << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_pop, expected_pop) << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_fused.dot_product, expected_and)
                    << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_fused.popcount, expected_pop)
                    << "d=" << d << " qb=" << qb;
        }
    }
}

TEST(RaBitQQuantization, Avx512MinmaxMatchesScalar) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX512)) {
        GTEST_SKIP() << "AVX512 is not available on this CPU";
    }

    float scalar_min = 1.0f;
    float scalar_max = 2.0f;
    float avx512_min = scalar_min;
    float avx512_max = scalar_max;
    faiss::rabitq::minmax_values<SIMDLevel::NONE>(
            nullptr, 0, scalar_min, scalar_max);
    faiss::rabitq::minmax_values<SIMDLevel::AVX512>(
            nullptr, 0, avx512_min, avx512_max);
    EXPECT_EQ(avx512_min, scalar_min);
    EXPECT_EQ(avx512_max, scalar_max);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (size_t d : {1, 7, 15, 16, 17, 31, 32, 33, 255}) {
        std::vector<float> values(d);
        for (float& value : values) {
            value = dist(rng);
        }

        faiss::rabitq::minmax_values<SIMDLevel::NONE>(
                values.data(), d, scalar_min, scalar_max);
        faiss::rabitq::minmax_values<SIMDLevel::AVX512>(
                values.data(), d, avx512_min, avx512_max);
        EXPECT_EQ(avx512_min, scalar_min) << "d=" << d;
        EXPECT_EQ(avx512_max, scalar_max) << "d=" << d;
    }
}

TEST(RaBitQQuantization, Avx512LargeCenteredAccumulationMatchesScalar) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX512)) {
        GTEST_SKIP() << "AVX512 is not available on this CPU";
    }

    // 65025 * d exceeds INT32_MAX and catches 32-bit SIMD reductions.
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

    std::vector<uint8_t> avx512_codes(d);
    size_t avx512_sum = 0;
    int64_t avx512_sum2 = 0;
    faiss::rabitq::quantize_query_values<SIMDLevel::AVX512>(
            values.data(),
            d,
            0.0f,
            255.0f,
            max_code,
            true,
            avx512_codes.data(),
            avx512_sum,
            avx512_sum2);

    EXPECT_EQ(avx512_codes, scalar_codes);
    EXPECT_EQ(avx512_sum, scalar_sum);
    EXPECT_EQ(avx512_sum2, scalar_sum2);
    EXPECT_EQ(avx512_sum2, int64_t(d) * max_code * max_code);
}

TEST(RaBitQQuantization, Avx512MatchesScalarAcrossCodeRange) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX512)) {
        GTEST_SKIP() << "AVX512 is not available on this CPU";
    }
    faiss_test::check_quantization_matches_scalar<SIMDLevel::AVX512>();
}
