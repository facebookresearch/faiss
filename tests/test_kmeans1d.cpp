/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/impl/FaissException.h>
#include <faiss/impl/kmeans1d.h>

namespace {

using faiss::kmeans1d;

// exact optimum for reference, O(n^2 k) in double
double optimal_sse(const std::vector<float>& x, int k) {
    const int n = int(x.size());
    std::vector<double> sum(n + 1, 0.0), sum2(n + 1, 0.0);
    for (int i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + x[i];
        sum2[i + 1] = sum2[i] + double(x[i]) * double(x[i]);
    }
    auto cost = [&](int i, int j) {
        double s = sum[j + 1] - sum[i];
        double s2 = sum2[j + 1] - sum2[i];
        int m = j - i + 1;
        return s2 - s * s / m;
    };
    const double INF = std::numeric_limits<double>::max();
    std::vector<std::vector<double>> D(k + 1, std::vector<double>(n, INF));
    for (int m = 0; m < n; m++) {
        D[1][m] = cost(0, m);
    }
    for (int c = 2; c <= k; c++) {
        for (int m = c - 1; m < n; m++) {
            for (int i = c - 1; i <= m; i++) {
                D[c][m] = std::min(D[c][m], D[c - 1][i - 1] + cost(i, m));
            }
        }
    }
    return D[k][n - 1];
}

// SSE of the clustering induced by the centroids (nearest centroid)
double sse_of(const std::vector<float>& x, const std::vector<float>& c) {
    double sse = 0.0;
    for (float v : x) {
        double best = std::numeric_limits<double>::max();
        for (float q : c) {
            best = std::min(best, double(v - q) * double(v - q));
        }
        sse += best;
    }
    return sse;
}

std::vector<float> run(const std::vector<float>& x, int k) {
    std::vector<float> centroids(k);
    kmeans1d(x.data(), x.size(), k, centroids.data());
    return centroids;
}

} // namespace

TEST(Kmeans1d, EmptyClusterInfeasibleAtKEqualDistinct) {
    const int nd = 21;
    std::vector<float> x;
    for (int i = 0; i < nd; i++) {
        float v = 1000.0f + i * 1e-4f;
        x.push_back(v);
        x.push_back(v);
    }
    std::sort(x.begin(), x.end());

    std::vector<float> centroids(nd);
    ASSERT_NO_THROW(kmeans1d(x.data(), x.size(), nd, centroids.data()));

    for (float c : centroids) {
        EXPECT_TRUE(std::isfinite(c));
    }
    EXPECT_NEAR(sse_of(x, centroids), optimal_sse(x, nd), 1e-2);
}

TEST(Kmeans1d, MoreClustersThanDistinctValues) {
    std::vector<float> x = {1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f};
    const int k = 7;
    std::vector<float> centroids(k);
    ASSERT_NO_THROW(kmeans1d(x.data(), x.size(), k, centroids.data()));
    for (float c : centroids) {
        EXPECT_TRUE(std::isfinite(c));
    }
    EXPECT_NEAR(
            sse_of(x, centroids), optimal_sse(x, std::min<int>(k, 3)), 1e-3);
}

TEST(Kmeans1d, CentroidAccuracyLargeCluster) {
    const int n = 1024;
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = 1e5f + i * 1e-2f;
    }
    std::sort(x.begin(), x.end());

    std::vector<float> centroids = run(x, 1);
    double mean = 0.0;
    for (float v : x) {
        mean += v;
    }
    mean /= n;
    EXPECT_NEAR(centroids[0], mean, 0.02);
}

TEST(Kmeans1d, MatchesBruteForceOptimum) {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 60; trial++) {
        const int n = 30 + int(rng() % 40);
        const double base = (trial % 3 == 0) ? 1e2
                : (trial % 3 == 1)           ? 1e3
                                             : 1e4;
        std::vector<float> x(n);
        for (auto& v : x) {
            v = float(base + double(rng() % 40) * 1e-2);
        }
        std::sort(x.begin(), x.end());
        std::set<float> distinct(x.begin(), x.end());
        const int k = 1 + int(rng() % std::min<int>(n, int(distinct.size())));

        std::vector<float> centroids = run(x, k);
        double got = sse_of(x, centroids);
        double opt = optimal_sse(x, k);
        double scale = optimal_sse(x, 1);
        EXPECT_NEAR(got, opt, 1e-4 * (1.0 + scale))
                << "trial " << trial << " n=" << n << " k=" << k;
    }
}

TEST(Kmeans1d, CostPrecisionAtLargeOffset) {
    const float base = 1e8f;
    std::vector<float> x = {
            base,
            base,
            base + 8.f,
            base + 8.f,
            base + 16.f,
            base + 16.f,
            base + 16.f,
            base + 16.f,
            base + 16.f,
            base + 16.f,
            base + 16.f,
            base + 16.f,
            base + 24.f,
            base + 24.f,
            base + 32.f,
            base + 32.f};
    std::sort(x.begin(), x.end());
    std::set<float> distinct(x.begin(), x.end());
    const int k = int(distinct.size());

    std::vector<float> centroids = run(x, k);
    EXPECT_NEAR(sse_of(x, centroids), optimal_sse(x, k), 1e-3);
}

TEST(Kmeans1d, NearTiesWithKEqualToDistinct) {
    std::mt19937 rng(999);
    for (int trial = 0; trial < 20; trial++) {
        const int nd = 15 + int(rng() % 10);
        std::vector<float> x;
        for (int i = 0; i < nd; i++) {
            float v = 1000.0f + i * 1e-4f;
            int copies = 1 + int(rng() % 3);
            for (int c = 0; c < copies; c++) {
                x.push_back(v);
            }
        }
        std::sort(x.begin(), x.end());
        std::set<float> distinct(x.begin(), x.end());
        const int k = int(distinct.size());

        std::vector<float> centroids(k);
        ASSERT_NO_THROW(kmeans1d(x.data(), x.size(), k, centroids.data()));
        EXPECT_NEAR(sse_of(x, centroids), optimal_sse(x, k), 1e-2)
                << "trial " << trial;
    }
}

TEST(Kmeans1d, CentroidsAreConsistentMeans) {
    std::mt19937 rng(7);
    const int n = 120;
    std::vector<float> x(n);
    for (auto& v : x) {
        v = float(rng() % 1000);
    }
    std::sort(x.begin(), x.end());
    const int k = 5;

    std::vector<float> centroids = run(x, k);

    std::vector<int> count(k, 0);
    std::vector<double> assigned_sum(k, 0.0);
    for (float v : x) {
        int best = 0;
        double bestd = std::numeric_limits<double>::max();
        for (int c = 0; c < k; c++) {
            double d = std::abs(double(v) - centroids[c]);
            if (d < bestd) {
                bestd = d;
                best = c;
            }
        }
        count[best]++;
        assigned_sum[best] += v;
    }
    for (int c = 0; c < k; c++) {
        ASSERT_GT(count[c], 0) << "cluster " << c << " is empty";
        EXPECT_NEAR(centroids[c], assigned_sum[c] / count[c], 1e-3)
                << "cluster " << c;
    }
}

TEST(Kmeans1d, ExtremeMagnitudeFailsLoudly) {
    const float v = 1e25f;
    std::vector<float> x(4);
    for (int i = 0; i < 4; i++) {
        x[i] = v + i * (v * 1e-5f);
    }
    std::sort(x.begin(), x.end());
    std::vector<float> centroids(3);
    EXPECT_THROW(
            kmeans1d(x.data(), x.size(), 3, centroids.data()),
            faiss::FaissException);
}
