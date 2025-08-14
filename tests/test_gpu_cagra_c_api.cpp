/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "c_api/AutoTune_c.h"
#include "c_api/Index_c.h"
#include "c_api/error_c.h"
#include "c_api/gpu/DeviceUtils_c.h"
#include "c_api/gpu/GpuAutoTune_c.h"
#include "c_api/gpu/GpuIndex_c.h"
#include "c_api/gpu/GpuResources_c.h"
#include "c_api/gpu/StandardGpuResources_c.h"
#include "c_api/index_factory_c.h"
#include "c_api/index_io_c.h"

namespace {

class GpuCagraCAPITest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Check GPU availability
        int gpus = -1;
        if (faiss_get_num_gpus(&gpus) != 0) {
            GTEST_SKIP() << "Failed to get GPU count";
        }

        if (gpus <= 0) {
            GTEST_SKIP() << "No GPUs available";
        }

        // Create GPU resources
        if (faiss_StandardGpuResources_new(&gpu_res_) != 0) {
            GTEST_SKIP() << "Failed to create GPU resources";
        }
    }

    void TearDown() override {
        if (gpu_res_) {
            faiss_StandardGpuResources_free(gpu_res_);
            gpu_res_ = nullptr;
        }
    }

    FaissStandardGpuResources* gpu_res_ = nullptr;

    // Helper function to generate random vectors
    std::vector<float> generateRandomVectors(int nb, int d) {
        std::vector<float> vectors(nb * d);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < nb * d; ++i) {
            vectors[i] = dis(gen);
        }
        return vectors;
    }
};

TEST_F(GpuCagraCAPITest, TestGpuIndexCagraCreation) {
    // Test different dimensions and graph degrees
    std::vector<int> dimensions = {64, 128, 256};
    std::vector<size_t> graph_degrees = {32, 64, 128};
    std::vector<FaissMetricType> metrics = {METRIC_L2, METRIC_INNER_PRODUCT};

    for (int d : dimensions) {
        for (size_t graph_degree : graph_degrees) {
            for (FaissMetricType metric : metrics) {
                FaissIndex* index = nullptr;

                // Test index creation
                EXPECT_EQ(
                        faiss_GpuIndexCagra_new(
                                &index, gpu_res_, d, metric, graph_degree),
                        0);
                EXPECT_NE(index, nullptr);

                // Test basic properties
                EXPECT_FALSE(faiss_Index_is_trained(index));

                // Test dimension
                EXPECT_EQ(faiss_Index_d(index), d);

                // Clean up
                faiss_Index_free(index);
            }
        }
    }
}

TEST_F(GpuCagraCAPITest, TestSearchParametersCagraCreation) {
    // Test different itopk sizes
    std::vector<size_t> itopk_sizes = {1, 5, 10, 20, 50};

    for (size_t itopk_size : itopk_sizes) {
        FaissSearchParameters* params = nullptr;

        // Test parameter creation
        EXPECT_EQ(faiss_SearchParametersCagra_new(&params, itopk_size), 0);
        EXPECT_NE(params, nullptr);

        // Clean up
        faiss_SearchParameters_free(params);
    }
}

TEST_F(GpuCagraCAPITest, TestGpuToCpuConversion) {
    // Create a GPU CAGRA index
    int d = 128;
    size_t graph_degree = 64;
    FaissIndex* gpu_index = nullptr;

    EXPECT_EQ(
            faiss_GpuIndexCagra_new(
                    &gpu_index, gpu_res_, d, METRIC_L2, graph_degree),
            0);
    EXPECT_NE(gpu_index, nullptr);

    // Add some vectors to train the index
    int nb = 100;
    auto xb = generateRandomVectors(nb, d);
    EXPECT_EQ(faiss_Index_add(gpu_index, nb, xb.data()), 0);

    // Convert GPU index to CPU
    FaissIndex* cpu_index = nullptr;
    EXPECT_EQ(faiss_index_gpu_to_cpu(gpu_index, &cpu_index), 0);
    EXPECT_NE(cpu_index, nullptr);

    // Test that both indices have the same basic properties
    EXPECT_EQ(faiss_Index_d(gpu_index), faiss_Index_d(cpu_index));
    EXPECT_EQ(
            faiss_Index_is_trained(gpu_index),
            faiss_Index_is_trained(cpu_index));

    // Clean up
    faiss_Index_free(gpu_index);
    faiss_Index_free(cpu_index);
}

TEST_F(GpuCagraCAPITest, TestCpuToGpuConversion) {
    // Create a simple CPU index first (use Flat for GPU conversion)
    int d = 128;
    FaissIndex* cpu_index = nullptr;
    EXPECT_EQ(faiss_index_factory(&cpu_index, d, "Flat", METRIC_L2), 0);
    EXPECT_NE(cpu_index, nullptr);

    // Add some vectors to the CPU index
    int nb = 100;
    auto xb = generateRandomVectors(nb, d);
    EXPECT_EQ(faiss_Index_add(cpu_index, nb, xb.data()), 0);

    // Convert CPU index to GPU
    FaissGpuIndex* gpu_index = nullptr;
    EXPECT_EQ(
            faiss_index_cpu_to_gpu(
                    reinterpret_cast<FaissGpuResourcesProvider*>(gpu_res_),
                    0,
                    cpu_index,
                    &gpu_index),
            0);
    EXPECT_NE(gpu_index, nullptr);

    // Test that both indices have the same basic properties
    EXPECT_EQ(faiss_Index_d(cpu_index), faiss_Index_d(gpu_index));
    EXPECT_EQ(
            faiss_Index_is_trained(cpu_index),
            faiss_Index_is_trained(gpu_index));

    // Clean up
    faiss_Index_free(cpu_index);
    faiss_Index_free(gpu_index);
}

TEST_F(GpuCagraCAPITest, TestEndToEndWorkflow) {
    // Generate test data
    int d = 128;
    int nb = 1000;
    int nq = 10;
    int k = 5;

    auto xb = generateRandomVectors(nb, d);
    auto xq = generateRandomVectors(nq, d);

    // Create GPU CAGRA index
    FaissIndex* gpu_index = nullptr;
    size_t graph_degree = 64;
    EXPECT_EQ(
            faiss_GpuIndexCagra_new(
                    &gpu_index, gpu_res_, d, METRIC_L2, graph_degree),
            0);
    EXPECT_NE(gpu_index, nullptr);

    // Add vectors to the index
    EXPECT_EQ(faiss_Index_add(gpu_index, nb, xb.data()), 0);

    // Create search parameters
    FaissSearchParameters* search_params = nullptr;
    size_t itopk_size = 10;
    EXPECT_EQ(faiss_SearchParametersCagra_new(&search_params, itopk_size), 0);
    EXPECT_NE(search_params, nullptr);

    // Perform search
    std::vector<idx_t> I(k * nq);
    std::vector<float> D(k * nq);

    EXPECT_EQ(
            faiss_Index_search(gpu_index, nq, xq.data(), k, D.data(), I.data()),
            0);

    // Verify search results
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < k; ++j) {
            EXPECT_GE(I[i * k + j], 0);
            EXPECT_LT(I[i * k + j], nb);
            EXPECT_GE(D[i * k + j], 0.0f);
        }
    }

    // Convert to CPU index
    FaissIndex* cpu_index = nullptr;
    EXPECT_EQ(faiss_index_gpu_to_cpu(gpu_index, &cpu_index), 0);
    EXPECT_NE(cpu_index, nullptr);

    // Search with CPU index
    std::vector<idx_t> I_cpu(k * nq);
    std::vector<float> D_cpu(k * nq);

    EXPECT_EQ(
            faiss_Index_search(
                    cpu_index, nq, xq.data(), k, D_cpu.data(), I_cpu.data()),
            0);

    // Clean up
    faiss_SearchParameters_free(search_params);
    faiss_Index_free(gpu_index);
    faiss_Index_free(cpu_index);
}
} // namespace
