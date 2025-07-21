/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Index.h>
#include <faiss/IndexSVSUncompressed.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>

TEST(SVSIO, WriteAndReadIndex) {
    const faiss::idx_t d = 64;
    faiss::IndexSVSUncompressed index(d);
    std::vector<float> xb(d * 100);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < xb.size(); ++i) {
        xb[i] = dis(gen);
    }
    index.add(100, xb.data());

    const std::string path = "/tmp/test_svs_index.faiss";

    // Serialize
    ASSERT_NO_THROW({ faiss::write_index(&index, path.c_str()); });

    // Deserialize
    faiss::IndexSVSUncompressed* loaded = nullptr;
    ASSERT_NO_THROW({
        loaded = dynamic_cast<faiss::IndexSVSUncompressed*>(
                faiss::read_index(path.c_str()));
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

    // Question: Save/load of SVS indices is tested within SVS. Do we still want
    // to validate `loaded->impl`?

    delete loaded;
}
