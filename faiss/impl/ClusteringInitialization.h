/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {

/// Initialization methods for k-means clustering centroids
enum class ClusteringInitMethod : uint8_t {
    /// Random sampling: select k random points uniformly from the dataset.
    /// Time complexity: O(k)
    RANDOM,

    /// k-means++: select centroids with probability proportional to D(x)²,
    /// where D(x) is the distance to the nearest existing centroid.
    /// Reference: Arthur, D., & Vassilvitskii, S. (2006). k-means++:
    /// The advantages of careful seeding. Stanford.
    /// Time complexity: O(nkd)
    KMEANS_PLUS_PLUS,

    /// AFK-MC²: Assumption-Free K-MC² using Markov Chain Monte Carlo.
    /// Provides theoretical guarantees without assumptions on data
    /// distribution.
    /// Uses a non-uniform proposal distribution based on D²-sampling from
    /// the first center, combined with uniform sampling for regularization.
    /// Reference: Bachem, O., Lucic, M., Hassani, H., & Krause, A. (2016).
    /// Fast and provably good seedings for k-means. Advances in neural
    /// information processing systems, 29.
    /// Time complexity: O(nd) preprocessing + O(mk²d) main loop
    AFK_MC2
};

/// Centroid initialization for k-means clustering.
///
/// This class provides different algorithms for selecting initial centroids
/// before running k-means iterations. Good initialization can significantly
/// improve clustering quality and convergence speed.
///
/// Example usage:
/// @code
///     ClusteringInitialization init(128, 1000);  // d=128, k=1000
///     init.method = ClusteringInitMethod::KMEANS_PLUS_PLUS;
///     init.seed = 42;
///
///     std::vector<float> centroids(128 * 1000);
///     init.init_centroids(n, x, centroids.data());
/// @endcode
struct ClusteringInitialization {
    size_t d; ///< vector dimension
    size_t k; ///< number of centroids to initialize

    /// Initialization method to use
    ClusteringInitMethod method = ClusteringInitMethod::RANDOM;

    /// Random seed.
    int64_t seed = 1234;

    /// Chain length for AFK-MC² (only used when method = AFK_MC2).
    /// Longer chains give better approximation to k-means++ but are slower.
    uint16_t afkmc2_chain_length = 50;

    ClusteringInitialization(size_t d, size_t k);

    /// Initialize k centroids from n input vectors.
    ///
    /// @param n          number of input vectors
    /// @param x          input vectors, size (n, d), row-major
    /// @param centroids  output centroids, size (k, d), row-major
    /// @param n_existing_centroids  number of pre-existing centroids to
    /// consider
    ///                              when computing distances (for k-means++ and
    ///                              AFK-MC²). These centroids are not modified.
    /// @param existing_centroids    pre-existing centroids, size
    ///                              (n_existing_centroids, d), row-major.
    ///                              New centroids will be selected to be far
    ///                              from these existing ones.
    void init_centroids(
            size_t n,
            const float* x,
            float* centroids,
            size_t n_existing_centroids = 0,
            const float* existing_centroids = nullptr) const;

   private:
    void init_random(size_t n, const float* x, float* centroids) const;
    void init_kmeans_plus_plus(
            size_t n,
            const float* x,
            float* centroids,
            size_t n_existing_centroids,
            const float* existing_centroids) const;
    void init_afkmc2(
            size_t n,
            const float* x,
            float* centroids,
            size_t n_existing_centroids,
            const float* existing_centroids) const;
};

} // namespace faiss
