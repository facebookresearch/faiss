/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <faiss/Clustering.h>

TEST(Clustering1D, shifted_interval_costs) {
    // The optimal contiguous groups are [64,96], [136,136], [168,176],
    // [200], [224,232,232], [296], [360,376,376,384], [408].
    const std::vector<float> values = {
            64,
            96,
            136,
            136,
            168,
            176,
            200,
            224,
            232,
            232,
            296,
            360,
            376,
            376,
            384,
            408};
    const std::vector<double> means = {
            80, 136, 172, 200, 688.0 / 3, 296, 374, 408};

    for (float offset : {0.0f, 1e8f, -1e8f}) {
        SCOPED_TRACE(offset);
        std::vector<float> x = values;
        for (float& v : x) {
            v += offset;
        }
        std::reverse(x.begin(), x.end());
        faiss::Clustering1D clustering(int(means.size()));
        clustering.train_exact(x.size(), x.data());

        ASSERT_EQ(clustering.centroids.size(), means.size());
        for (size_t i = 0; i < means.size(); ++i) {
            // Account for unavoidable rounding of the float output, rather
            // than comparing shifted-back centroids with unshifted centroids.
            EXPECT_EQ(
                    clustering.centroids[i],
                    static_cast<float>(double(offset) + means[i]));
        }
        ASSERT_EQ(clustering.iteration_stats.size(), 1);
        EXPECT_DOUBLE_EQ(clustering.iteration_stats[0].imbalance_factor, 1.25);
    }
}

TEST(Clustering1D, constant_cluster_mean) {
    for (float value : {1000001.0f, -1000001.0f}) {
        SCOPED_TRACE(value);
        std::vector<float> x(128, value);
        faiss::Clustering1D clustering(1);
        clustering.max_points_per_centroid = 128; // Keep every input point.
        clustering.train_exact(x.size(), x.data());

        ASSERT_EQ(clustering.centroids.size(), 1);
        EXPECT_EQ(clustering.centroids[0], value);
        ASSERT_EQ(clustering.iteration_stats.size(), 1);
        EXPECT_DOUBLE_EQ(clustering.iteration_stats[0].imbalance_factor, 1.0);
    }
}
