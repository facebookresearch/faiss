/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <faiss/Index.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>
#include <faiss/svs/IndexSVSFlat.h>
#include <faiss/svs/IndexSVSVamana.h>
#include <faiss/svs/IndexSVSVamanaLVQ.h>
#include <faiss/svs/IndexSVSVamanaLeanVec.h>
#include <gtest/gtest.h>
#include <random>
#include <type_traits>

#include "test_util.h"

namespace {
pthread_mutex_t temp_file_mutex = PTHREAD_MUTEX_INITIALIZER;

// Test fixture class to manage shared test data
class SVS : public ::testing::Test {
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

// LVQ/LeanVec tests are only executed if LVQ/LeanVec support is available
class SVSLL : public SVS {
   protected:
    void SetUp() override {
        if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
            GTEST_SKIP() << "LVQ/LeanVec support not available on this "
                            "platform or build configuration";
        }
    }
};

// Consistency checks for behavior if LVQ/LeanVec are not available
// Only runs if LVQ/LeanVec is NOT available
class SVSNoLL : public SVS {
   protected:
    void SetUp() override {
        if (faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
            GTEST_SKIP() << "LVQ/LeanVec support is available; skipping "
                            "NoLL tests";
        }
    }
};

// Define static members
std::vector<float> SVS::test_data;
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
    ASSERT_NE(loaded->impl, nullptr);
    EXPECT_EQ(loaded->d, index.d);
    EXPECT_EQ(loaded->metric_type, index.metric_type);
    EXPECT_EQ(loaded->graph_max_degree, index.graph_max_degree);
    EXPECT_EQ(loaded->alpha, index.alpha);
    EXPECT_EQ(loaded->search_window_size, index.search_window_size);
    EXPECT_EQ(loaded->search_buffer_capacity, index.search_buffer_capacity);
    EXPECT_EQ(loaded->construction_window_size, index.construction_window_size);
    EXPECT_EQ(loaded->max_candidate_pool_size, index.max_candidate_pool_size);
    EXPECT_EQ(loaded->prune_to, index.prune_to);
    EXPECT_EQ(loaded->use_full_search_history, index.use_full_search_history);
    if constexpr (std::is_same_v<
                          std::decay_t<T>,
                          faiss::IndexSVSVamanaLeanVec>) {
        auto* leanvec_loaded =
                dynamic_cast<faiss::IndexSVSVamanaLeanVec*>(loaded);
        ASSERT_NE(leanvec_loaded, nullptr);
        EXPECT_EQ(leanvec_loaded->leanvec_d, index.leanvec_d);

        EXPECT_NE(leanvec_loaded->training_data, nullptr);
    }

    delete loaded;
}

template <typename T>
void train_save_load_and_add_index(
        T& index,
        const std::vector<float>& xb,
        size_t n) {
    index.train(n, xb.data());

    std::string temp_filename_template = "/tmp/faiss_svs_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);

    ASSERT_NO_THROW({ faiss::write_index(&index, filename.c_str()); });

    T* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<T*>(faiss::read_index(filename.c_str()));
    });

    ASSERT_NE(loaded, nullptr);
    ASSERT_NO_THROW({ loaded->add(n, xb.data()); });

    delete loaded;
}

template <typename T>
void save_and_load_index() {
    T index;

    std::string temp_filename_template = "/tmp/faiss_svs_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);

    faiss::write_index(&index, filename.c_str());
    T* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<T*>(faiss::read_index(filename.c_str()));
    });
    delete loaded;
}

TEST_F(SVS, WriteAndReadIndexSVS) {
    faiss::IndexSVSVamana index{d, 64ul};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVS, WriteAndReadIndexSVSFP16) {
    faiss::IndexSVSVamana index{
            d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_FP16};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVS, WriteAndReadIndexSVSSQI8) {
    faiss::IndexSVSVamana index{
            d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_SQI8};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSLL, WriteAndReadIndexSVSLVQ4x0) {
    faiss::IndexSVSVamanaLVQ index{d, 64ul};
    index.storage_kind = faiss::SVSStorageKind::SVS_LVQ4x0;
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSLL, WriteAndReadIndexSVSLVQ4x4) {
    faiss::IndexSVSVamanaLVQ index{d, 64ul};
    index.storage_kind = faiss::SVSStorageKind::SVS_LVQ4x4;
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSLL, WriteAndReadIndexSVSLVQ4x8) {
    faiss::IndexSVSVamanaLVQ index{d, 64ul};
    index.storage_kind = faiss::SVSStorageKind::SVS_LVQ4x8;
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSLL, WriteAndReadIndexSVSVamanaLeanVec4x4) {
    faiss::IndexSVSVamanaLeanVec index{
            d,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec4x4};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSLL, WriteAndReadIndexSVSVamanaLeanVec4x8) {
    faiss::IndexSVSVamanaLeanVec index{
            d,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec4x8};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSLL, WriteAndReadIndexSVSVamanaLeanVec8x8) {
    faiss::IndexSVSVamanaLeanVec index{
            d,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec8x8};
    write_and_read_index(index, test_data, n);
}

TEST_F(SVSLL, LeanVecThrowsWithoutTraining) {
    faiss::IndexSVSVamanaLeanVec index{
            64,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec4x4};
    ASSERT_THROW(index.add(100, test_data.data()), faiss::FaissException);
}

TEST_F(SVS, VamanaTrainSaveLoadAndAdd) {
    faiss::IndexSVSVamana index{d, 64ul};
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVS, VamanaFP16TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamana index{
            d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_FP16};
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVS, VamanaSQI8TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamana index{
            d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_SQI8};
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVSLL, LVQ4x0TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamanaLVQ index{d, 64ul};
    index.storage_kind = faiss::SVSStorageKind::SVS_LVQ4x0;
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVSLL, LVQ4x4TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamanaLVQ index{d, 64ul};
    index.storage_kind = faiss::SVSStorageKind::SVS_LVQ4x4;
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVSLL, LVQ4x8TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamanaLVQ index{d, 64ul};
    index.storage_kind = faiss::SVSStorageKind::SVS_LVQ4x8;
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVSLL, LeanVec4x4TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamanaLeanVec index{
            d,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec4x4};
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVSLL, LeanVec4x8TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamanaLeanVec index{
            d,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec4x8};
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVSLL, LeanVec8x8TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamanaLeanVec index{
            d,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec8x8};
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVS, SaveAndLoadIndexSVSFlat) {
    save_and_load_index<faiss::IndexSVSFlat>();
}

TEST_F(SVS, SaveAndLoadIndexSVSVamana) {
    save_and_load_index<faiss::IndexSVSVamana>();
}

TEST_F(SVSLL, SaveAndLoadIndexSVSVamanaLVQ) {
    save_and_load_index<faiss::IndexSVSVamanaLVQ>();
}

TEST_F(SVSLL, SaveAndLoadIndexSVSVamanaLeanVec) {
    save_and_load_index<faiss::IndexSVSVamanaLeanVec>();
}

TEST_F(SVS, WriteAndReadIndexSVSFlat) {
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
    ASSERT_NE(loaded->impl, nullptr);
    EXPECT_EQ(loaded->d, index.d);
    EXPECT_EQ(loaded->nlabels, index.nlabels);
    EXPECT_EQ(loaded->metric_type, index.metric_type);

    delete loaded;
}

// Test search with IDSelector filtering
TEST_F(SVS, SearchWithIDSelector) {
    faiss::IndexSVSVamana index{d, 64ul};
    index.add(n, test_data.data());

    const int nq = 8;                   // number of queries
    const float* xq = test_data.data(); // reuse first nq vectors as queries
    const int k = 10;

    size_t min_id = n / 5;     // inclusive
    size_t max_id = n * 4 / 5; // exclusive
    faiss::IDSelectorRange selector(min_id, max_id);

    faiss::SearchParameters params; // generic search parameters with selector
    params.sel = &selector;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    ASSERT_NO_THROW(
            index.search(nq, xq, k, distances.data(), labels.data(), &params));

    // All returned labels must fall inside the selected range
    for (int i = 0; i < nq * k; ++i) {
        EXPECT_GE(labels[i], (faiss::idx_t)min_id);
        EXPECT_LT(labels[i], (faiss::idx_t)max_id);
    }
}

// Basic functional test for range_search and parameter override helper
TEST_F(SVS, RangeSearchFunctional) {
    faiss::IndexSVSVamana index{d, 64ul};
    index.add(n, test_data.data());
    const int nq = 5;
    const float* xq = test_data.data();

    // Small radius
    faiss::RangeSearchResult res_small(nq);
    ASSERT_NO_THROW(index.range_search(nq, xq, 0.05f, &res_small));

    // Larger radius to exercise loop continuation
    faiss::RangeSearchResult res_big(nq);
    ASSERT_NO_THROW(index.range_search(nq, xq, 5.0f, &res_big));
    EXPECT_GE(res_big.lims[nq], res_small.lims[nq]);

    // Provide custom params to ensure branch coverage (non-null params)
    faiss::SearchParametersSVSVamana params;
    params.search_window_size = 15;
    params.search_buffer_capacity = 20;

    // Provide IDSelector to ensure branch coverage (non-null params->sel)
    size_t min_id = n / 5;     // inclusive
    size_t max_id = n * 4 / 5; // exclusive
    faiss::IDSelectorRange selector(min_id, max_id);
    params.sel = &selector;

    faiss::RangeSearchResult res_params(nq);
    ASSERT_NO_THROW(index.range_search(nq, xq, 1.0f, &res_params, &params));

    // All returned labels must fall inside the selected range
    for (size_t i = 0; i < res_params.lims[nq]; ++i) {
        EXPECT_GE(res_params.labels[i], (faiss::idx_t)min_id);
        EXPECT_LT(res_params.labels[i], (faiss::idx_t)max_id);
    }
}

TEST_F(SVSLL, LVQAndLeanVecDoNotThrowWhenEnabled) {
    // explicit constructor with LVQ dataset
    ASSERT_NO_THROW({
        faiss::IndexSVSVamanaLVQ index(
                d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_LVQ4x4);
    });

    // default constructor, will initialize dataset on first add()
    ASSERT_NO_THROW({
        faiss::IndexSVSVamanaLVQ index;
        index.add(n, test_data.data());
    });

    // explicit constructor with LeanVec dataset
    ASSERT_NO_THROW({
        faiss::IndexSVSVamanaLeanVec index(
                d,
                64ul,
                faiss::METRIC_L2,
                faiss::SVSStorageKind::SVS_LeanVec4x4);
    });

    // default constructor, will initialize dataset on first add()
    ASSERT_NO_THROW({
        faiss::IndexSVSVamanaLeanVec index;
        index.d = 64;
        index.leanvec_d = 32;
        index.train(n, test_data.data());
        index.add(n, test_data.data());
    });
}

TEST_F(SVSNoLL, LVQAndLeanVecThrowWhenNotEnabled) {
    // explicit constructor with LVQ dataset
    ASSERT_THROW(
            {
                faiss::IndexSVSVamanaLVQ index(
                        d,
                        64ul,
                        faiss::METRIC_L2,
                        faiss::SVSStorageKind::SVS_LVQ4x4);
            },
            faiss::FaissException);

    // default constructor, will initialize dataset on first add()
    ASSERT_THROW(
            {
                faiss::IndexSVSVamanaLVQ index;
                index.add(n, test_data.data());
            },
            faiss::FaissException);

    // explicit constructor with LeanVec dataset
    ASSERT_THROW(
            {
                faiss::IndexSVSVamanaLeanVec index(
                        d,
                        64ul,
                        faiss::METRIC_L2,
                        faiss::SVSStorageKind::SVS_LeanVec4x4);
            },
            faiss::FaissException);

    // default constructor, will initialize dataset on first add()
    ASSERT_THROW(
            {
                faiss::IndexSVSVamanaLeanVec index;
                index.d = 64;
                index.leanvec_d = 32;
                index.train(n, test_data.data());
            },
            faiss::FaissException);
}
