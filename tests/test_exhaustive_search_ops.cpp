/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file is used to test internal functions in distances.cpp that are in an
// anonymous namespace It includes distances.cpp directly to access those
// functions

#include <gtest/gtest.h>
#include <algorithm>

#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances_fused/exhaustive_l2sqr_fused_cmax_256bit.h>
#include <faiss/utils/exhaustive_search_ops.h>
#include <faiss/utils/ordered_key_value.h>

namespace faiss_test {

class ExhaustiveSearchOpsTest
        : public ::testing::TestWithParam<faiss::SIMDLevel> {
   protected:
    void SetUp() override {
        original_simd_level = faiss::SIMDConfig::get_level();

        simd_level = GetParam();

        faiss::SIMDConfig::set_level(simd_level);
    }

    void TearDown() override {
        faiss::SIMDConfig::set_level(original_simd_level);
    }

    faiss::SIMDLevel simd_level;
    faiss::SIMDLevel original_simd_level;
};

TEST_P(ExhaustiveSearchOpsTest, TestExhaustiveL2sqrBlasCMaxDirectly) {
    size_t d = 4;
    size_t nx = 3;
    size_t ny = 5;

    float x_row0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float x_row1[4] = {0.0f, 3.0f, 0.0f, 0.0f};
    float x_row2[4] = {0.0f, 0.0f, 5.0f, 0.0f};
    float* x = new float[d * nx];

    std::copy(x_row0, x_row0 + 4, x);
    std::copy(x_row1, x_row1 + 4, x + 4);
    std::copy(x_row2, x_row2 + 4, x + 8);

    float y_row0[4] = {9.0, 9.0, 9.0, 9.0};
    float y_row1[4] = {0.0f, 0.0f, 4.0f, 0.0f};
    float y_row2[4] = {0.0f, 1.5f, 0.0f, 0.0f};
    float y_row3[4] = {-0.5f, 0.0f, 0.0f, 0.0f};
    float* y = new float[d * ny];
    float* y_norms = new float[ny];

    std::copy(y_row0, y_row0 + 4, y);
    std::copy(y_row1, y_row1 + 4, y + 4);
    std::copy(y_row2, y_row2 + 4, y + 8);
    std::copy(y_row3, y_row3 + 4, y + 12);

    for (size_t i = 0; i < ny; i++) {
        y_norms[i] = 0;
        for (size_t j = 0; j < d; j++) {
            y_norms[i] += y[i * d + j] * y[i * d + j];
        }
    }

    // Create arrays to store the results
    float* distances = new float[nx];
    int64_t* indices = new int64_t[nx];

    for (size_t i = 0; i < nx; i++) {
        distances[i] = 999.0f;
        indices[i] = 999;
    }

    faiss::Top1BlockResultHandler<faiss::CMax<float, int64_t>> res(
            nx, distances, indices);

    faiss::exhaustive_L2sqr_blas<
            faiss::Top1BlockResultHandler<faiss::CMax<float, int64_t>>>(
            x, y, d, nx, ny, res, y_norms);

    EXPECT_EQ(indices[0], 3);
    EXPECT_EQ(indices[1], 2);
    EXPECT_EQ(indices[2], 1);

    EXPECT_NEAR(distances[0], 2.25, 1e-6);
    EXPECT_NEAR(distances[1], 2.25, 1e-6);
    EXPECT_NEAR(distances[2], 1.0, 1e-6);

    EXPECT_NEAR(y_norms[0], 324, 1e-6);
    EXPECT_NEAR(y_norms[1], 16, 1e-6);
    EXPECT_NEAR(y_norms[2], 2.25, 1e-6);
    EXPECT_NEAR(y_norms[3], 0.25, 1e-6);

    delete[] distances;
    delete[] indices;
    delete[] x;
    delete[] y;
    delete[] y_norms;
}

TEST_P(ExhaustiveSearchOpsTest, exhaustive_L2sqr_fused_cmax_simdlib) {
    // fused cmax simdlib is only available for AVX2, AVX512, ARM_NEON, and
    // ARM_SVE. Skip if not compiled with support for the SIMD level.
    if (simd_level == faiss::SIMDLevel::NONE) {
        return;
    }

    // Skip if the library was not compiled with support for this SIMD level
#if !defined(COMPILE_SIMD_AVX2)
    if (simd_level == faiss::SIMDLevel::AVX2) {
        return;
    }
#endif
#if !defined(COMPILE_SIMD_AVX512)
    if (simd_level == faiss::SIMDLevel::AVX512) {
        return;
    }
#endif
#if !defined(COMPILE_SIMD_ARM_NEON)
    if (simd_level == faiss::SIMDLevel::ARM_NEON) {
        return;
    }
#endif

    size_t d = 4;
    size_t nx = 3;
    size_t ny = 5;

    float x_row0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float x_row1[4] = {0.0f, 3.0f, 0.0f, 0.0f};
    float x_row2[4] = {0.0f, 0.0f, 5.0f, 0.0f};
    float* x = new float[d * nx];

    std::copy(x_row0, x_row0 + 4, x);
    std::copy(x_row1, x_row1 + 4, x + 4);
    std::copy(x_row2, x_row2 + 4, x + 8);

    float y_row0[4] = {9.0, 9.0, 9.0, 9.0};
    float y_row1[4] = {0.0f, 0.0f, 4.0f, 0.0f};
    float y_row2[4] = {0.0f, 1.5f, 0.0f, 0.0f};
    float y_row3[4] = {-0.5f, 0.0f, 0.0f, 0.0f};
    float* y = new float[d * ny];
    float* y_norms = new float[ny];

    std::copy(y_row0, y_row0 + 4, y);
    std::copy(y_row1, y_row1 + 4, y + 4);
    std::copy(y_row2, y_row2 + 4, y + 8);
    std::copy(y_row3, y_row3 + 4, y + 12);

    for (size_t i = 0; i < ny; i++) {
        y_norms[i] = 0;
        for (size_t j = 0; j < d; j++) {
            y_norms[i] += y[i * d + j] * y[i * d + j];
        }
    }

    // Create arrays to store the results
    float* distances = new float[nx];
    int64_t* indices = new int64_t[nx];

    for (size_t i = 0; i < nx; i++) {
        distances[i] = 999.0f;
        indices[i] = 999;
    }

    faiss::Top1BlockResultHandler<faiss::CMax<float, int64_t>> res(
            nx, distances, indices);

    bool actual = false;

#if defined(COMPILE_SIMD_AVX2)
    if (simd_level == faiss::SIMDLevel::AVX2) {
        actual = exhaustive_L2sqr_fused_cmax_simdlib<faiss::SIMDLevel::AVX2>(
                x, y, d, nx, ny, res, y_norms);
    }
#endif

#if defined(COMPILE_SIMD_AVX512)
    if (simd_level == faiss::SIMDLevel::AVX512) {
        actual = exhaustive_L2sqr_fused_cmax_simdlib<faiss::SIMDLevel::AVX512>(
                x, y, d, nx, ny, res, y_norms);
    }
#endif

#if defined(COMPILE_SIMD_ARM_NEON)
    if (simd_level == faiss::SIMDLevel::ARM_NEON) {
        actual =
                exhaustive_L2sqr_fused_cmax_simdlib<faiss::SIMDLevel::ARM_NEON>(
                        x, y, d, nx, ny, res, y_norms);
    }
#endif

    // #if defined(__ARM_FEATURE_SVE)
    //     if (simd_level == faiss::SIMDLevel::ARM_SVE) {
    //         actual =
    //         exhaustive_L2sqr_fused_cmax_simdlib<faiss::SIMDLevel::ARM_SVE>(
    //                 x, y, d, nx, ny, res, y_norms);
    //     }
    // #endif

    EXPECT_TRUE(actual);

    EXPECT_EQ(indices[0], 3);
    EXPECT_EQ(indices[1], 2);
    EXPECT_EQ(indices[2], 1);

    EXPECT_NEAR(distances[0], 2.25, 1e-6);
    EXPECT_NEAR(distances[1], 2.25, 1e-6);
    EXPECT_NEAR(distances[2], 1.0, 1e-6);

    EXPECT_NEAR(y_norms[0], 324, 1e-6);
    EXPECT_NEAR(y_norms[1], 16, 1e-6);
    EXPECT_NEAR(y_norms[2], 2.25, 1e-6);
    EXPECT_NEAR(y_norms[3], 0.25, 1e-6);

    delete[] distances;
    delete[] indices;
    delete[] x;
    delete[] y;
    delete[] y_norms;
}

std::vector<faiss::SIMDLevel> GetSupportedSIMDLevels() {
    std::vector<faiss::SIMDLevel> supportedSIMDLevels;
    for (int i = 0; i < static_cast<int>(faiss::SIMDLevel::COUNT); ++i) {
        auto simd_level = static_cast<faiss::SIMDLevel>(i);
        if (faiss::SIMDConfig::is_simd_level_available(simd_level)) {
            supportedSIMDLevels.push_back(static_cast<faiss::SIMDLevel>(i));
        }
    }

    EXPECT_TRUE(supportedSIMDLevels.size() > 0);
    return supportedSIMDLevels;
}

::testing::internal::ParamGenerator<faiss::SIMDLevel> SupportedSIMDLevels() {
    std::vector<faiss::SIMDLevel> levels = GetSupportedSIMDLevels();
    return ::testing::ValuesIn(levels);
}

INSTANTIATE_TEST_SUITE_P(
        SIMDLevels,
        ExhaustiveSearchOpsTest,
        SupportedSIMDLevels());

} // namespace faiss_test
