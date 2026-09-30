/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Shared by test_rabitq_simd.cpp and test_rabitq_simd_dd.cpp. The AVX512
// kernels live in a dynamic dispatch build only, so the tests that call them
// sit in the second file, but both compare against the same scalar reference.

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/simd_levels.h>

namespace faiss_test {

using faiss::SIMDLevel;

template <SIMDLevel SL>
inline void check_quantization_matches_scalar() {
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

} // namespace faiss_test
