/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/Index.h>
#include <faiss/utils/utils.h>

TEST(TestUtils, get_version) {
    std::string version = std::to_string(FAISS_VERSION_MAJOR) + "." +
            std::to_string(FAISS_VERSION_MINOR) + "." +
            std::to_string(FAISS_VERSION_PATCH);

    EXPECT_EQ(version, faiss::get_version());
}

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>

TEST(IndexFlat1D, remove_ids) {
    // continuous_update = true
    faiss::IndexFlat1D index(true);
    std::vector<float> xb = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    index.add(5, xb.data());
    EXPECT_EQ(index.ntotal, 5);

    faiss::idx_t id_to_remove = 0;
    faiss::IDSelectorBatch sel(1, &id_to_remove);
    size_t removed = index.remove_ids(sel);
    EXPECT_EQ(removed, 1);
    EXPECT_EQ(index.ntotal, 4);

    // Search should succeed without throwing
    float xq = 2.5f;
    std::vector<float> dist(4);
    std::vector<faiss::idx_t> labels(4);
    EXPECT_NO_THROW(index.search(1, &xq, 4, dist.data(), labels.data()));

    // continuous_update = false
    faiss::IndexFlat1D index_manual(false);
    index_manual.add(5, xb.data());
    index_manual.update_permutation();

    removed = index_manual.remove_ids(sel);
    EXPECT_EQ(removed, 1);
    // Permutation is invalidated, search should throw
    EXPECT_THROW(
            index_manual.search(1, &xq, 4, dist.data(), labels.data()),
            faiss::FaissException);

    index_manual.update_permutation();
    EXPECT_NO_THROW(
            index_manual.search(1, &xq, 4, dist.data(), labels.data()));
}

TEST(IndexFlat1D, merge_from) {
    faiss::IndexFlat1D index1(true);
    std::vector<float> xb1 = {1.0f, 3.0f};
    index1.add(2, xb1.data());

    faiss::IndexFlat1D index2(true);
    std::vector<float> xb2 = {2.0f, 4.0f};
    index2.add(2, xb2.data());

    index1.merge_from(index2);
    EXPECT_EQ(index1.ntotal, 4);
    EXPECT_EQ(index2.ntotal, 0);

    float xq = 2.5f;
    std::vector<float> dist(4);
    std::vector<faiss::idx_t> labels(4);
    EXPECT_NO_THROW(index1.search(1, &xq, 4, dist.data(), labels.data()));

    // Merge empty index into index1: should preserve permutation
    faiss::IndexFlat1D index_empty(true);
    index1.merge_from(index_empty);
    EXPECT_EQ(index1.ntotal, 4);
    EXPECT_NO_THROW(index1.search(1, &xq, 4, dist.data(), labels.data()));
}

