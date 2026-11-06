/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <set>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

// See https://github.com/facebookresearch/faiss/issues/5580

TEST(TestIVFFlatDedup, ResetClearsInstances) {
    const int d = 8;
    // 10 unique vectors, each repeated once with a distinct id
    const int n = 20;
    std::vector<float> x(n * d);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < d; j++) {
            x[i * d + j] = (i * 7 + j * 13) % 29 * 0.25f;
            x[(i + 10) * d + j] = x[i * d + j];
        }
    }
    std::vector<faiss::idx_t> ids(n);
    for (int i = 0; i < n; i++) {
        ids[i] = i;
    }

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlatDedup index(&quantizer, d, 1);
    index.cp.min_points_per_centroid = 1; // silence the clustering warning
    index.cp.max_points_per_centroid = 100;
    index.train(n, x.data());
    index.add_with_ids(n, x.data(), ids.data());
    EXPECT_EQ(index.ntotal, n);
    EXPECT_EQ(index.instances.size(), 10u);

    index.reset();
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_TRUE(index.instances.empty());

    // The index must keep deduplicating after a reset
    index.add_with_ids(n, x.data(), ids.data());
    EXPECT_EQ(index.ntotal, n);
    EXPECT_EQ(index.instances.size(), 10u);

    std::vector<float> query(x.begin(), x.begin() + d);
    std::vector<faiss::idx_t> labels(n);
    std::vector<float> distances(n);
    index.search(1, query.data(), n, distances.data(), labels.data());
    std::set<faiss::idx_t> found(labels.begin(), labels.end());
    EXPECT_EQ(found.size(), static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(found.count(i), 1u);
    }
}
