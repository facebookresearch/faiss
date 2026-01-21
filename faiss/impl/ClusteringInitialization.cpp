/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ClusteringInitialization.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

namespace faiss {

namespace {

uint64_t get_seed(int64_t seed) {
    if (seed >= 0) {
        return static_cast<uint64_t>(seed);
    }
    return static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                                         .time_since_epoch()
                                         .count());
}

/// Compute distance from point idx to its nearest centroid.
/// Optionally checks both primary and secondary centroid sets.
float distance_to_nearest_centroid(
        size_t d,
        size_t n_centroids,
        const float* x,
        size_t idx,
        const float* centroids,
        size_t n_existing_centroids = 0,
        const float* existing_centroids = nullptr) {
    if (n_centroids == 0 && n_existing_centroids == 0) {
        return std::numeric_limits<float>::infinity();
    }

    const float* point = x + idx * d;
    float min_dist = std::numeric_limits<float>::max();

    // Check primary centroids
    for (size_t c = 0; c < n_centroids; c++) {
        float dist = fvec_L2sqr(point, centroids + c * d, d);
        min_dist = std::min(min_dist, dist);
    }

    // Check existing centroids if provided
    for (size_t c = 0; c < n_existing_centroids; c++) {
        float dist = fvec_L2sqr(point, existing_centroids + c * d, d);
        min_dist = std::min(min_dist, dist);
    }

    return min_dist;
}

/// Result of initializing distances for D² sampling
struct InitDistancesResult {
    size_t first_new_centroid_idx;
    double sum_d2;
    size_t first_selected_idx; // Only valid when first_new_centroid_idx == 1
};

/// Initialize distance array for D² sampling.
/// If existing centroids are provided, computes distances to them.
/// Otherwise, selects first centroid randomly and computes distances to it.
/// Returns first_new_centroid_idx (0 if existing, 1 if random first),
/// sum of squared distances, and the first selected index (if applicable).
InitDistancesResult init_distances_for_d2_sampling(
        size_t d,
        size_t n,
        const float* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids,
        std::vector<double>& distances,
        std::mt19937_64& rng) {
    double sum_d2 = 0.0;
    size_t first_selected_idx = 0;

    if (n_existing_centroids > 0 && existing_centroids != nullptr) {
        // Compute distances to nearest existing centroid
        for (size_t i = 0; i < n; i++) {
            distances[i] = distance_to_nearest_centroid(
                    d, n_existing_centroids, x, i, existing_centroids);
            sum_d2 += distances[i];
        }
        return {0, sum_d2, 0};
    } else {
        // Select first centroid randomly
        std::uniform_int_distribution<size_t> uniform_dist(0, n - 1);
        first_selected_idx = uniform_dist(rng);
        std::memcpy(centroids, x + first_selected_idx * d, d * sizeof(float));

        // Compute distances to first centroid
        for (size_t i = 0; i < n; i++) {
            distances[i] = fvec_L2sqr(x + i * d, centroids, d);
            sum_d2 += distances[i];
        }
        return {1, sum_d2, first_selected_idx};
    }
}

/// Sample an index from a distribution using precomputed cumulative sum.
/// Falls back to uniform sampling if total weight is zero.
size_t sample_from_cumsum(
        const std::vector<double>& q_cumsum,
        std::mt19937_64& rng) {
    size_t n = q_cumsum.size();
    if (n == 0) {
        return 0;
    }

    double total = q_cumsum[n - 1];
    if (total <= 0) {
        // Fallback to uniform sampling if all weights are zero
        std::uniform_int_distribution<size_t> uniform(0, n - 1);
        return uniform(rng);
    }

    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng);

    auto it = std::lower_bound(q_cumsum.begin(), q_cumsum.end(), r);
    size_t idx = std::distance(q_cumsum.begin(), it);
    return std::min(idx, n - 1);
}

} // namespace

ClusteringInitialization::ClusteringInitialization(size_t d, size_t k)
        : d(d), k(k) {}

void ClusteringInitialization::init_centroids(
        size_t n,
        const float* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) const {
    FAISS_THROW_IF_NOT_FMT(
            n >= k,
            "Number of points (%zu) must be >= number of centroids (%zu)",
            n,
            k);
    FAISS_THROW_IF_NOT(d > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(centroids != nullptr);
    FAISS_THROW_IF_NOT(
            n_existing_centroids == 0 || existing_centroids != nullptr);

    switch (method) {
        case ClusteringInitMethod::RANDOM:
            init_random(n, x, centroids);
            break;
        case ClusteringInitMethod::KMEANS_PLUS_PLUS:
            init_kmeans_plus_plus(
                    n, x, centroids, n_existing_centroids, existing_centroids);
            break;
        case ClusteringInitMethod::AFK_MC2:
            init_afkmc2(
                    n, x, centroids, n_existing_centroids, existing_centroids);
            break;
        default:
            FAISS_THROW_MSG("Unknown initialization method");
    }
}

void ClusteringInitialization::init_random(
        size_t n,
        const float* x,
        float* centroids) const {
    // Use rand_perm for backward compatibility with Clustering.cpp
    // This ensures the same random sequence as the original implementation
    std::vector<int> perm(n);
    rand_perm(perm.data(), n, seed);

    // Copy selected points to centroids
    for (size_t i = 0; i < k; i++) {
        std::memcpy(centroids + i * d, x + perm[i] * d, d * sizeof(float));
    }
}

void ClusteringInitialization::init_kmeans_plus_plus(
        size_t n,
        const float* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) const {
    std::mt19937_64 rng(get_seed(seed));

    std::vector<double> min_distances(n);
    auto result = init_distances_for_d2_sampling(
            d,
            n,
            x,
            centroids,
            n_existing_centroids,
            existing_centroids,
            min_distances,
            rng);

    if (result.first_new_centroid_idx == 1 && k == 1) {
        return;
    }

    // Reusable buffer for cumulative sum
    std::vector<double> cumsum(n);

    // Select remaining centroids using D² sampling
    for (size_t c = result.first_new_centroid_idx; c < k; c++) {
        // Compute cumulative sum
        cumsum[0] = min_distances[0];
        for (size_t i = 1; i < n; i++) {
            cumsum[i] = cumsum[i - 1] + min_distances[i];
        }

        // Sample using precomputed cumsum
        size_t next_idx = sample_from_cumsum(cumsum, rng);

        float* new_centroid = centroids + c * d;
        std::memcpy(new_centroid, x + next_idx * d, d * sizeof(float));

        // Update min distances incrementally
        for (size_t i = 0; i < n; i++) {
            double dist = fvec_L2sqr(x + i * d, new_centroid, d);
            min_distances[i] = std::min(min_distances[i], dist);
        }
    }
}

void ClusteringInitialization::init_afkmc2(
        size_t n,
        const float* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) const {
    // AFK-MC² (Assumption-Free K-MC²) algorithm:
    // Reference: Bachem et al., "Fast and Provably Good Seedings for k-Means"

    std::mt19937_64 rng(get_seed(seed));
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);

    // Track selected centroids to prevent duplicates
    std::unordered_set<size_t> selected_centroids;

    // Compute proposal distribution q(x)
    // If existing centroids: base q on distance to nearest existing centroid
    // Otherwise: select first centroid randomly and base q on it
    std::vector<double> dist_to_nearest(n);
    auto result = init_distances_for_d2_sampling(
            d,
            n,
            x,
            centroids,
            n_existing_centroids,
            existing_centroids,
            dist_to_nearest,
            rng);

    if (result.first_new_centroid_idx == 1) {
        selected_centroids.insert(result.first_selected_idx);
        if (k == 1) {
            return;
        }
    }

    // Compute q(x) and cumulative sum in a single pass
    std::vector<double> q(n);
    std::vector<double> q_cumsum(n);
    double uniform_term = 0.5 / static_cast<double>(n);

    for (size_t i = 0; i < n; i++) {
        double d2_term = (result.sum_d2 > 0)
                ? 0.5 * dist_to_nearest[i] / result.sum_d2
                : 0.0;
        q[i] = d2_term + uniform_term;
        q_cumsum[i] = (i > 0 ? q_cumsum[i - 1] : 0.0) + q[i];
    }

    // Main loop: Select remaining centroids using MCMC
    for (size_t c = result.first_new_centroid_idx; c < k; c++) {
        // Sample initial candidate from proposal distribution q, skip
        // duplicates
        size_t current_idx;
        do {
            current_idx = sample_from_cumsum(q_cumsum, rng);
        } while (selected_centroids.count(current_idx) > 0);

        // Compute distance to nearest centroid (existing + newly selected)
        double current_dist = distance_to_nearest_centroid(
                d,
                c,
                x,
                current_idx,
                centroids,
                n_existing_centroids,
                existing_centroids);
        double current_q = q[current_idx];

        // Run Markov chain
        for (size_t m = 0; m < afkmc2_chain_length; m++) {
            // Sample proposal from q
            size_t proposed_idx = sample_from_cumsum(q_cumsum, rng);

            // Skip duplicates before expensive distance computation
            if (selected_centroids.count(proposed_idx) > 0) {
                continue;
            }

            // Compute distance to nearest centroid (existing + newly selected)
            double proposed_dist = distance_to_nearest_centroid(
                    d,
                    c,
                    x,
                    proposed_idx,
                    centroids,
                    n_existing_centroids,
                    existing_centroids);
            double proposed_q = q[proposed_idx];

            // Metropolis-Hastings acceptance ratio:
            // accept = min(1, d(y,C)² · q(x) / (d(x,C)² · q(y)))
            double acceptance_prob = 0.0;
            if (current_dist <= 0) {
                // Current point is a centroid (distance = 0), never leave
                acceptance_prob = 0.0;
            } else if (proposed_q > 0) {
                double numerator = proposed_dist * current_q;
                double denominator = current_dist * proposed_q;
                acceptance_prob = std::min(1.0, numerator / denominator);
            }

            if (uniform_01(rng) < acceptance_prob) {
                current_idx = proposed_idx;
                current_dist = proposed_dist;
                current_q = proposed_q;
            }
        }

        // Use final chain state as new centroid
        selected_centroids.insert(current_idx);
        std::memcpy(centroids + c * d, x + current_idx * d, d * sizeof(float));
    }
}

} // namespace faiss
