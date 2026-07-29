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

// 32-d chunks and chunk boundaries.
static const std::vector<size_t> kDims =
        {1, 8, 16, 31, 32, 33, 255, 256, 257, 512, 1024, 2048};

template <SIMDLevel SL>
static void check_quantization_matches_scalar() {
    constexpr size_t d = 257;
    constexpr float v_min = -2.0f;
    constexpr float inv_delta = 16.0f;

    for (uint8_t max_code : {uint8_t(1), uint8_t(15), uint8_t(255)}) {
        std::vector<float> values(d);
        for (size_t i = 0; i < d; i++) {
            const uint8_t code = i % (size_t(max_code) + 1);
            values[i] = v_min + code / inv_delta;
        }

        for (bool centered : {false, true}) {
            std::vector<uint8_t> scalar_codes(d);
            size_t scalar_sum = 0;
            int64_t scalar_sum2 = 0;
            faiss::rabitq::quantize_query_values<SIMDLevel::NONE>(
                    values.data(),
                    d,
                    v_min,
                    inv_delta,
                    max_code,
                    centered,
                    scalar_codes.data(),
                    scalar_sum,
                    scalar_sum2);

            std::vector<uint8_t> simd_codes(d);
            size_t simd_sum = 0;
            int64_t simd_sum2 = 0;
            faiss::rabitq::quantize_query_values<SL>(
                    values.data(),
                    d,
                    v_min,
                    inv_delta,
                    max_code,
                    centered,
                    simd_codes.data(),
                    simd_sum,
                    simd_sum2);

            EXPECT_EQ(simd_codes, scalar_codes)
                    << "max_code=" << int(max_code) << " centered=" << centered;
            EXPECT_EQ(simd_sum, scalar_sum);
            EXPECT_EQ(simd_sum2, scalar_sum2);
        }
    }

    constexpr float lut_min = -2.0f;
    constexpr float lut_scale = 4.0f;
    float lut[16];
    for (size_t i = 0; i < 16; i++) {
        lut[i] = lut_min + i / lut_scale;
    }
    uint8_t scalar_lut[16];
    uint8_t simd_lut[16];
    faiss::rabitq::lut_quantize_16_to_uint8<SIMDLevel::NONE>(
            lut, lut_min, lut_scale, scalar_lut);
    faiss::rabitq::lut_quantize_16_to_uint8<SL>(
            lut, lut_min, lut_scale, simd_lut);
    EXPECT_EQ(
            std::vector<uint8_t>(simd_lut, simd_lut + 16),
            std::vector<uint8_t>(scalar_lut, scalar_lut + 16));
}

// Note: scalar kernel's own correctness is covered end-to-end by
// tests/test_rabitq.py. This target is x86-only (see BUCK).
TEST(RaBitQRearrangeBitPlanes, Avx2MatchesScalar) {
#ifdef FAISS_ENABLE_DD
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }
#endif

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

#ifdef COMPILE_SIMD_AVX2

TEST(RaBitQQuantization, Avx2MinmaxMatchesScalar) {
#ifdef FAISS_ENABLE_DD
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }
#endif

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
#ifdef FAISS_ENABLE_DD
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }
#endif

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
#ifdef FAISS_ENABLE_DD
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 is not available on this CPU";
    }
#endif
    check_quantization_matches_scalar<SIMDLevel::AVX2>();
}

#endif

#ifdef COMPILE_SIMD_AVX512

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
    check_quantization_matches_scalar<SIMDLevel::AVX512>();
}

#endif
