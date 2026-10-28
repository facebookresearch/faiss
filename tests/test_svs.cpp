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
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>
#include <faiss/svs/IndexSVSFlat.h>
#include <faiss/svs/IndexSVSIVF.h>
#include <faiss/svs/IndexSVSIVFLVQ.h>
#include <faiss/svs/IndexSVSIVFLeanVec.h>
#include <faiss/svs/IndexSVSVamana.h>
#include <faiss/svs/IndexSVSVamanaLVQ.h>
#include <faiss/svs/IndexSVSVamanaLeanVec.h>
#include <gtest/gtest.h>
#include <random>
#include <type_traits>

#include "test_util.h"

namespace {
pthread_mutex_t temp_file_mutex = PTHREAD_MUTEX_INITIALIZER;

class SVS : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
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

class SVSLL : public SVS {
   protected:
    void SetUp() override {
        if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
            GTEST_SKIP() << "LVQ/LeanVec support not available on this "
                            "platform or build configuration";
        }
    }
};

class SVSNoLL : public SVS {
   protected:
    void SetUp() override {
        if (faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
            GTEST_SKIP() << "LVQ/LeanVec support is available; skipping "
                            "NoLL tests";
        }
    }
};

std::vector<float> SVS::test_data;
} // namespace

template <typename T>
void write_and_read_index(T& index, const std::vector<float>& xb, size_t n) {
    index.train(n, xb.data());
    index.add(n, xb.data());

    std::string temp_filename_template = "/tmp/faiss_svs_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);

    ASSERT_NO_THROW({ faiss::write_index(&index, filename.c_str()); });

    T* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<T*>(faiss::read_index(filename.c_str()));
    });

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
    EXPECT_EQ(loaded->leanvec_d, index.leanvec_d);
    EXPECT_NE(loaded->training_data, nullptr);

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

TEST_F(SVSLL, LeanVec4x4TrainSaveLoadAndAdd) {
    faiss::IndexSVSVamanaLeanVec index{
            d,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec4x4};
    train_save_load_and_add_index(index, test_data, n);
}

TEST_F(SVS, SaveAndLoadIndexSVSFlat) {
    save_and_load_index<faiss::IndexSVSFlat>();
}

TEST_F(SVSLL, SaveAndLoadIndexSVSVamanaLeanVec) {
    save_and_load_index<faiss::IndexSVSVamanaLeanVec>();
}

TEST_F(SVS, RangeSearchFunctional) {
    faiss::IndexSVSVamana index{d, 64ul};
    index.add(n, test_data.data());
    const int nq = 5;
    const float* xq = test_data.data();

    // SearchParametersSVSVamana override and selector branches
    faiss::SearchParametersSVSVamana params;
    params.search_window_size = 15;
    params.search_buffer_capacity = 20;

    size_t min_id = n / 5;     // inclusive
    size_t max_id = n * 4 / 5; // exclusive
    faiss::IDSelectorRange selector(min_id, max_id);
    params.sel = &selector;

    // The radius covers nearly every vector, so the batch iterator loop
    // continues past its first batch.
    faiss::RangeSearchResult res_params(nq);
    ASSERT_NO_THROW(index.range_search(nq, xq, 20.0f, &res_params, &params));
    ASSERT_GT(res_params.lims[nq], 0);

    for (size_t i = 0; i < res_params.lims[nq]; ++i) {
        EXPECT_GE(res_params.labels[i], (faiss::idx_t)min_id);
        EXPECT_LT(res_params.labels[i], (faiss::idx_t)max_id);
    }
}

TEST_F(SVSLL, LVQAndLeanVecDoNotThrowWhenEnabled) {
    // default constructor, will initialize dataset on first add()
    ASSERT_NO_THROW({
        faiss::IndexSVSVamanaLVQ index;
        index.add(n, test_data.data());
    });

    ASSERT_NO_THROW({
        faiss::IndexSVSVamanaLeanVec index;
        index.d = 64;
        index.leanvec_d = 32;
        index.train(n, test_data.data());
        index.add(n, test_data.data());
    });
}

TEST_F(SVSNoLL, LVQAndLeanVecThrowWhenNotEnabled) {
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

    ASSERT_THROW(
            {
                faiss::IndexSVSVamanaLeanVec index(
                        d,
                        64ul,
                        faiss::METRIC_L2,
                        faiss::SVSStorageKind::SVS_LeanVec4x4);
            },
            faiss::FaissException);

    ASSERT_THROW(
            {
                faiss::IndexSVSVamanaLeanVec index;
                index.d = 64;
                index.leanvec_d = 32;
                index.train(n, test_data.data());
            },
            faiss::FaissException);
}

/****************************************************************
 * SVS IVF Tests
 ****************************************************************/

template <typename T>
void write_and_read_ivf_index(
        T& index,
        const std::vector<float>& xb,
        size_t n) {
    index.train(n, xb.data());
    index.add(n, xb.data());

    std::string temp_filename_template = "/tmp/faiss_svs_ivf_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);

    ASSERT_NO_THROW({ faiss::write_index(&index, filename.c_str()); });

    T* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<T*>(faiss::read_index(filename.c_str()));
    });

    ASSERT_NE(loaded, nullptr);
    ASSERT_NE(loaded->impl, nullptr);
    EXPECT_EQ(loaded->d, index.d);
    EXPECT_EQ(loaded->metric_type, index.metric_type);
    EXPECT_EQ(loaded->num_centroids, index.num_centroids);
    EXPECT_EQ(loaded->n_probes, index.n_probes);
    EXPECT_EQ(loaded->k_reorder, index.k_reorder);
    EXPECT_EQ(loaded->num_iterations, index.num_iterations);
    EXPECT_EQ(loaded->minibatch_size, index.minibatch_size);
    EXPECT_EQ(loaded->storage_kind, index.storage_kind);
    if constexpr (std::is_same_v<std::decay_t<T>, faiss::IndexSVSIVFLeanVec>) {
        auto* leanvec_loaded = dynamic_cast<faiss::IndexSVSIVFLeanVec*>(loaded);
        ASSERT_NE(leanvec_loaded, nullptr);
        EXPECT_EQ(leanvec_loaded->leanvec_d, index.leanvec_d);

        EXPECT_NE(leanvec_loaded->training_data, nullptr);
    }

    delete loaded;
}

TEST_F(SVSLL, WriteAndReadIndexSVSIVFLVQ4x0) {
    faiss::IndexSVSIVFLVQ index{d, 4ul};
    index.storage_kind = faiss::SVSStorageKind::SVS_LVQ4x0;
    write_and_read_ivf_index(index, test_data, n);
}

TEST_F(SVSLL, WriteAndReadIndexSVSIVFLeanVec4x8) {
    faiss::IndexSVSIVFLeanVec index{
            d, 4ul, faiss::METRIC_L2, 0, faiss::SVSStorageKind::SVS_LeanVec4x8};
    write_and_read_ivf_index(index, test_data, n);
}

TEST_F(SVSLL, WriteAndReadIndexSVSIVFLeanVec8x8) {
    faiss::IndexSVSIVFLeanVec index{
            d, 4ul, faiss::METRIC_L2, 0, faiss::SVSStorageKind::SVS_LeanVec8x8};
    write_and_read_ivf_index(index, test_data, n);
}

// Each SVS family streams its payload to the third-party SVS reader through a
// ReaderStreambuf, and only a streambuf carrying the deserialization vector
// byte limit rejects an oversized read. Every family must therefore build one
// the same way.
template <typename T>
void expect_byte_limit_enforced(
        T& index,
        const std::vector<float>& xb,
        size_t n) {
    index.train(n, xb.data());
    index.add(n, xb.data());

    std::string temp_filename_template = "/tmp/faiss_svs_limit_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);
    ASSERT_NO_THROW({ faiss::write_index(&index, filename.c_str()); });

    const size_t old_limit = faiss::get_deserialization_vector_byte_limit();
    faiss::set_deserialization_vector_byte_limit(64);
    EXPECT_THROW(
            { delete faiss::read_index(filename.c_str()); },
            faiss::FaissException);
    faiss::set_deserialization_vector_byte_limit(old_limit);

    faiss::Index* loaded = nullptr;
    ASSERT_NO_THROW({ loaded = faiss::read_index(filename.c_str()); });
    EXPECT_NE(loaded, nullptr);
    delete loaded;
}

TEST_F(SVS, FlatHonorsDeserializationVectorByteLimit) {
    faiss::IndexSVSFlat index{d};
    expect_byte_limit_enforced(index, test_data, n);
}

TEST_F(SVS, IVFHonorsDeserializationVectorByteLimit) {
    faiss::IndexSVSIVF index{d, 4ul};
    expect_byte_limit_enforced(index, test_data, n);
}

TEST_F(SVS, SaveAndLoadIndexSVSIVF) {
    save_and_load_index<faiss::IndexSVSIVF>();
}

TEST_F(SVSLL, SaveAndLoadIndexSVSIVFLeanVec) {
    save_and_load_index<faiss::IndexSVSIVFLeanVec>();
}

/****************************************************************
 * SVS Static IVF Tests
 ****************************************************************/

template <typename T>
void write_and_read_static_ivf_index(
        T& index,
        const std::vector<float>& xb,
        size_t n) {
    ASSERT_TRUE(index.is_static);
    index.train(n, xb.data());

    const int nq = 4;
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    ASSERT_NO_THROW(
            index.search(nq, xb.data(), k, distances.data(), labels.data()));

    std::string temp_filename_template =
            "/tmp/faiss_svs_static_ivf_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);

    ASSERT_NO_THROW({ faiss::write_index(&index, filename.c_str()); });

    T* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<T*>(faiss::read_index(filename.c_str()));
    });

    ASSERT_NE(loaded, nullptr);
    ASSERT_NE(loaded->impl, nullptr);
    EXPECT_TRUE(loaded->is_static);
    EXPECT_EQ(loaded->d, index.d);
    EXPECT_EQ(loaded->metric_type, index.metric_type);
    EXPECT_EQ(loaded->num_centroids, index.num_centroids);
    EXPECT_EQ(loaded->storage_kind, index.storage_kind);

    ASSERT_NO_THROW(
            loaded->search(nq, xb.data(), k, distances.data(), labels.data()));

    delete loaded;
}

TEST_F(SVS, WriteAndReadStaticIVFFP16) {
    faiss::IndexSVSIVF index{
            d, 4ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_FP16, true};
    write_and_read_static_ivf_index(index, test_data, n);
}

// The SVS runtime only takes intra_query_threads when the index is built or
// loaded and has no setter afterwards, so it must be set before train().
TEST_F(SVS, IVFIntraQueryThreadsSetBeforeTrain) {
    faiss::IndexSVSIVF index{d, 4ul};

    index.intra_query_threads = 2;
    index.train(n, test_data.data());
    index.add(n, test_data.data());

    const int nq = 4;
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    ASSERT_NO_THROW(index.search(
            nq, test_data.data(), k, distances.data(), labels.data()));
}

/****************************************************************
 * SVS Static Vamana Tests
 ****************************************************************/

template <typename T>
void write_and_read_static_vamana_index(
        T& index,
        const std::vector<float>& xb,
        size_t n,
        int read_io_flags = 0) {
    ASSERT_TRUE(index.is_static);

    // The static build itself happens inside add().
    index.add(n, xb.data());
    ASSERT_NE(index.impl, nullptr);
    ASSERT_TRUE(index.is_trained);

    const int nq = 4;
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    ASSERT_NO_THROW(
            index.search(nq, xb.data(), k, distances.data(), labels.data()));

    std::string temp_filename_template =
            "/tmp/faiss_svs_static_vamana_test_XXXXXX";
    Tempfilename filename(&temp_file_mutex, temp_filename_template);

    ASSERT_NO_THROW({ faiss::write_index(&index, filename.c_str()); });

    T* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<T*>(
                faiss::read_index(filename.c_str(), read_io_flags));
    });

    ASSERT_NE(loaded, nullptr);
    ASSERT_NE(loaded->impl, nullptr);
    EXPECT_TRUE(loaded->is_static);
    EXPECT_EQ(loaded->d, index.d);
    EXPECT_EQ(loaded->metric_type, index.metric_type);
    EXPECT_EQ(loaded->graph_max_degree, index.graph_max_degree);
    EXPECT_EQ(loaded->storage_kind, index.storage_kind);

    ASSERT_NO_THROW(
            loaded->search(nq, xb.data(), k, distances.data(), labels.data()));

    delete loaded;
}

TEST_F(SVS, WriteAndReadStaticVamanaFP16) {
    faiss::IndexSVSVamana index{
            d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_FP16, true};
    write_and_read_static_vamana_index(index, test_data, n);
}

TEST_F(SVS, WriteAndReadStaticVamanaSQ8) {
    faiss::IndexSVSVamana index{
            d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_SQ8, true};
    write_and_read_static_vamana_index(index, test_data, n);
}

TEST_F(SVS, VamanaStoreVectorsDisabled) {
    faiss::IndexSVSVamana index{
            d,
            64ul,
            faiss::METRIC_L2,
            faiss::SVSStorageKind::SVS_FP32,
            false,
            false};
    index.add(n, test_data.data());

    // The opt-out survives reset().
    index.reset();
    index.add(n, test_data.data());
    EXPECT_TRUE(index.stored_vectors.empty());
}

TEST_F(SVS, StaticVamanaStoreVectorsDisabled) {
    faiss::IndexSVSVamana index{
            d,
            64ul,
            faiss::METRIC_L2,
            faiss::SVSStorageKind::SVS_FP32,
            true,
            false};
    index.add(n, test_data.data());
    EXPECT_TRUE(index.stored_vectors.empty());

    std::vector<float> recons(d);
    EXPECT_THROW(index.reconstruct(0, recons.data()), faiss::FaissException);
}

// Opting out mid-stream cannot keep the remaining copy aligned with the ids,
// so it is dropped rather than left partially populated.
TEST_F(SVS, VamanaStoreVectorsDroppedOnMidStreamOptOut) {
    faiss::IndexSVSVamana index{d, 64ul};
    index.add(n / 2, test_data.data());
    EXPECT_EQ(index.stored_vectors.size(), (n / 2) * d);

    index.store_vectors = false;
    index.add(n / 2, test_data.data() + (n / 2) * d);
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(n));
    EXPECT_TRUE(index.stored_vectors.empty());

    std::vector<float> recons(d);
    EXPECT_THROW(index.reconstruct(0, recons.data()), faiss::FaissException);
}

TEST_F(SVSLL, LVQStoreVectorsDisabled) {
    faiss::IndexSVSVamanaLVQ index{
            d,
            64ul,
            faiss::METRIC_L2,
            faiss::SVSStorageKind::SVS_LVQ4x8,
            false,
            false};
    index.add(n, test_data.data());
    EXPECT_TRUE(index.stored_vectors.empty());
}

TEST_F(SVSLL, LeanVecStoreVectorsDisabled) {
    faiss::IndexSVSVamanaLeanVec index{
            d,
            64ul,
            faiss::METRIC_L2,
            0,
            faiss::SVSStorageKind::SVS_LeanVec4x8,
            false,
            false};
    index.train(n, test_data.data());
    index.add(n, test_data.data());
    EXPECT_TRUE(index.stored_vectors.empty());
}

TEST_F(SVS, WriteAndMapStaticVamanaFP16) {
    faiss::IndexSVSVamana index{
            d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_FP16, true};
    write_and_read_static_vamana_index(
            index, test_data, n, faiss::IO_FLAG_MMAP_IFC);
}

TEST_F(SVS, WriteAndMapStaticVamanaSQ8) {
    faiss::IndexSVSVamana index{
            d, 64ul, faiss::METRIC_L2, faiss::SVSStorageKind::SVS_SQ8, true};
    write_and_read_static_vamana_index(
            index, test_data, n, faiss::IO_FLAG_MMAP_IFC);
}
