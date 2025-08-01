/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Index.h>
#include <faiss/IndexSVS.h>
#include <faiss/IndexSVSFlat.h>
#include <faiss/IndexSVSLVQ.h>
#include <faiss/IndexSVSLeanVec.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>
#include <type_traits>

#include "test_util.h"

namespace {
pthread_mutex_t temp_file_mutex = PTHREAD_MUTEX_INITIALIZER;

// Test fixture class to manage shared test data
class SVSIOTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        // Generate test data once for all tests
        constexpr size_t d = 64;
        constexpr size_t n = 100;
        test_data.resize(d * n);

        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = dis(gen);
        }
    }

    static std::vector<float> test_data;
    static constexpr size_t d = 64;
    static constexpr size_t n = 100;
};

// Define static members
std::vector<float> SVSIOTest::test_data;
} // namespace

template <typename T>
void write_and_read_index(T& index, const std::vector<float>& xb, size_t n) {
    index.train(n, xb.data());
    index.add(n, xb.data());

    std::string temp_filename_template = "/tmp/faiss_svs_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);

    // Serialize
    ASSERT_NO_THROW({ faiss::write_index(&index, filename.c_str()); });

    // Deserialize
    T* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<T*>(faiss::read_index(filename.c_str()));
    });

    // Basic checks
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->d, index.d);
    EXPECT_EQ(loaded->metric_type, index.metric_type);
    EXPECT_EQ(loaded->num_threads, index.num_threads);
    EXPECT_EQ(loaded->graph_max_degree, index.graph_max_degree);
    EXPECT_EQ(loaded->alpha, index.alpha);
    EXPECT_EQ(loaded->search_window_size, index.search_window_size);
    EXPECT_EQ(loaded->search_buffer_capacity, index.search_buffer_capacity);
    EXPECT_EQ(loaded->construction_window_size, index.construction_window_size);
    EXPECT_EQ(loaded->max_candidate_pool_size, index.max_candidate_pool_size);
    EXPECT_EQ(loaded->prune_to, index.prune_to);
    EXPECT_EQ(loaded->use_full_search_history, index.use_full_search_history);
    if constexpr (std::is_same_v<std::decay_t<T>, faiss::IndexSVSLVQ>) {
        auto* lvq_loaded = dynamic_cast<faiss::IndexSVSLVQ*>(loaded);
        ASSERT_NE(lvq_loaded, nullptr);
        EXPECT_EQ(lvq_loaded->lvq_level, index.lvq_level);
    }

    delete loaded;
}

TEST_F(SVSIOTest, WriteAndReadIndexSVS) {
    faiss::IndexSVS index{d};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSIOTest, WriteAndReadIndexSVSLVQ4x0) {
    faiss::IndexSVSLVQ index{d};
    index.lvq_level = faiss::LVQLevel::LVQ_4x0;
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSIOTest, WriteAndReadIndexSVSLVQ4x4) {
    faiss::IndexSVSLVQ index{d};
    index.lvq_level = faiss::LVQLevel::LVQ_4x4;
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSIOTest, WriteAndReadIndexSVSLVQ4x8) {
    faiss::IndexSVSLVQ index{d};
    index.lvq_level = faiss::LVQLevel::LVQ_4x8;
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSIOTest, WriteAndReadIndexSVSLeanVec4x4) {
    faiss::IndexSVSLeanVec index{
            d, faiss::METRIC_L2, 0, faiss::LeanVecLevel::LeanVec_4x4};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSIOTest, WriteAndReadIndexSVSLeanVec4x8) {
    faiss::IndexSVSLeanVec index{
            d, faiss::METRIC_L2, 0, faiss::LeanVecLevel::LeanVec_4x8};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSIOTest, WriteAndReadIndexSVSLeanVec8x8) {
    faiss::IndexSVSLeanVec index{
            d, faiss::METRIC_L2, 0, faiss::LeanVecLevel::LeanVec_8x8};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSIOTest, LeanVecThrowsWithoutTraining) {
    faiss::IndexSVSLeanVec index{
            64, faiss::METRIC_L2, 0, faiss::LeanVecLevel::LeanVec_4x4};
    ASSERT_THROW(index.add(100, test_data.data()), faiss::FaissException);
}

TEST_F(SVSIOTest, WriteAndReadIndexSVSFlat) {
    faiss::IndexSVSFlat index{d};
    index.add(n, test_data.data());

    std::string temp_filename_template = "/tmp/faiss_svs_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);

    // Serialize
    ASSERT_NO_THROW({ faiss::write_index(&index, filename.c_str()); });

    // Deserialize
    faiss::IndexSVSFlat* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<faiss::IndexSVSFlat*>(
                faiss::read_index(filename.c_str()));
    });

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->d, index.d);
    EXPECT_EQ(loaded->metric_type, index.metric_type);
    EXPECT_EQ(loaded->num_threads, index.num_threads);

    delete loaded;
}
