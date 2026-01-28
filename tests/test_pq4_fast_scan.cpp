/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <gtest/gtest.h>

#include <faiss/Index.h> // For idx_t definition
#include <faiss/impl/pq_4bit/pq4_fast_scan.h>
#include <faiss/utils/simd_levels.h>

class PQ4FastScanTest : public ::testing::TestWithParam<faiss::SIMDLevel> {
   protected:
    void SetUp() override {
        original_simd_level = faiss::SIMDConfig::get_level();

        simd_level = GetParam();

        faiss::SIMDConfig::set_level(simd_level);
    }

    void TearDown() override {
        faiss::SIMDConfig::set_level(original_simd_level);
    }

    faiss::SIMDLevel original_simd_level;
    faiss::SIMDLevel simd_level;
};

TEST_P(PQ4FastScanTest, some_test) {
    bool is_max = true;
    bool user_reservoir = true;
    int nq = 1;
    int k = 1;
    int ntotal = 1;

    float* distances = new float[nq * k];
    // int* ids = new int[nq * k];
    faiss::idx_t* ids = new faiss::idx_t[nq * k];
    int ns = 1;
    float* normalizers = nullptr;
    bool disable = false;

    auto pq4codescanner = faiss::pq4_make_flat_knn_handler(
            is_max,
            user_reservoir,
            nq,
            k,
            ntotal,
            distances,
            ids,
            ns,
            normalizers,
            disable);

    EXPECT_TRUE(pq4codescanner != nullptr);

    delete[] distances;
    delete[] ids;
    delete normalizers;
    delete pq4codescanner;
}

std::vector<faiss::SIMDLevel> GET_SUPPORTED_SIMD_LEVELS() {
    std::vector<faiss::SIMDLevel> levels;
    for (int i = 0; i < static_cast<int>(faiss::SIMDLevel::COUNT); ++i) {
        auto level = static_cast<faiss::SIMDLevel>(i);
        if (faiss::SIMDConfig::is_simd_level_available(level)) {
            levels.push_back(level);
        }
    }
    EXPECT_TRUE(levels.size() > 0);
    return levels;
}

::testing::internal::ParamGenerator<faiss::SIMDLevel> SupportedSIMDLevels() {
    std::vector<faiss::SIMDLevel> levels = GET_SUPPORTED_SIMD_LEVELS();
    return ::testing::ValuesIn(levels);
}

INSTANTIATE_TEST_SUITE_P(SIMDLevels, PQ4FastScanTest, SupportedSIMDLevels());
