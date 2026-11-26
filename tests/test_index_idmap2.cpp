/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>

// See https://github.com/facebookresearch/faiss/issues/5578

TEST(TestIDMap2, ResetClearsRevMap) {
    const int d = 4;
    const int n = 10;
    std::vector<float> x(n * d);
    for (int i = 0; i < n * d; i++) {
        x[i] = (i * 7 % 13) * 0.5f;
    }

    faiss::IndexFlatL2 index(d);
    faiss::IndexIDMap2 index_with_ids(&index);
    std::vector<faiss::idx_t> ids(n);
    for (int i = 0; i < n; i++) {
        ids[i] = 100 + i;
    }

    index_with_ids.add_with_ids(n, x.data(), ids.data());
    index_with_ids.check_consistency();

    index_with_ids.reset();
    index_with_ids.check_consistency();

    // The index must remain fully usable after a reset
    index_with_ids.add_with_ids(n, x.data(), ids.data());
    index_with_ids.check_consistency();

    std::vector<float> query(x.begin(), x.begin() + d);
    std::vector<faiss::idx_t> labels(1);
    std::vector<float> distances(1);
    index_with_ids.search(1, query.data(), 1, distances.data(), labels.data());
    EXPECT_EQ(labels[0], 100);

    std::vector<float> recons(d);
    index_with_ids.reconstruct(100, recons.data());
    EXPECT_EQ(recons[0], x[0]);
}
