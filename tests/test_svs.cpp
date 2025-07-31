/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Index.h>
#include <faiss/IndexSVS.h>
#include <faiss/IndexSVSLVQ.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>

#include "test_util.h"

namespace {
pthread_mutex_t temp_file_mutex = PTHREAD_MUTEX_INITIALIZER;
}

template <typename T>
void write_and_read_index(T& index) {
    std::vector<float> xb(index.d * 100);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < xb.size(); ++i) {
        xb[i] = dis(gen);
    }
    index.add(100, xb.data());

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

    delete loaded;
}

TEST(SVSIO, WriteAndReadIndexIndexSVS) {
    constexpr faiss::idx_t d = 64;
    faiss::IndexSVS index{d};
    write_and_read_index(index);
}

TEST(SVSIO, WriteAndReadIndexIndexSVSLVQ4x0) {
    constexpr faiss::idx_t d = 64;
    faiss::IndexSVSLVQ index{d};
    index.lvq_level = faiss::LVQLevel::LVQ_4x0;
    write_and_read_index(index);
}

TEST(SVSIO, WriteAndReadIndexIndexSVSLVQ4x4) {
    constexpr faiss::idx_t d = 64;
    faiss::IndexSVSLVQ index{d};
    index.lvq_level = faiss::LVQLevel::LVQ_4x4;
    write_and_read_index(index);
}

TEST(SVSIO, WriteAndReadIndexIndexSVSLVQ4x8) {
    constexpr faiss::idx_t d = 64;
    faiss::IndexSVSLVQ index{d};
    index.lvq_level = faiss::LVQLevel::LVQ_4x8;
    write_and_read_index(index);
}