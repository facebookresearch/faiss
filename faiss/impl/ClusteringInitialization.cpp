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
#include <utility>
#include <vector>

#include <faiss/impl/ClusteringHelpers.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances_dispatch.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_impl/fp16_kernels.h>

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

struct FloatRows {
    static constexpr bool needs_scratch = false;
    static constexpr bool has_fixed_simd_level = false;

    size_t d;
    const float* x;

    const float* row(size_t idx, float*) const {
        return x + idx * d;
    }

    void copy_row(size_t idx, float* out) const {
        std::memcpy(out, x + idx * d, d * sizeof(float));
    }
};

template <SIMDLevel SL>
struct Fp16Rows {
    static constexpr bool needs_scratch = true;
    static constexpr bool has_fixed_simd_level = true;
    static constexpr SIMDLevel simd_level = SL;

    size_t d;
    const uint16_t* x;

    const float* row(size_t idx, float* scratch) const {
        detail::fp16_to_fp32_kernel<SL>(d, x + idx * d, scratch);
        return scratch;
    }

    void copy_row(size_t idx, float* out) const {
        detail::fp16_to_fp32_kernel<SL>(d, x + idx * d, out);
    }

    void copy_rows(size_t idx, size_t count, float* out) const {
        detail::fp16_to_fp32_kernel<SL>(count * d, x + idx * d, out);
    }
};

template <typename Rows, typename Action>
auto with_rows_simd_level(Action&& action) {
    if constexpr (Rows::has_fixed_simd_level) {
        return action.template operator()<Rows::simd_level>();
    } else {
        return with_simd_level(std::forward<Action>(action));
    }
}

template <typename Rows>
std::vector<float> row_scratch(size_t d) {
    return std::vector<float>(Rows::needs_scratch ? d : 0);
}

/// Compute distance from point idx to its nearest centroid.
/// Optionally checks both primary and secondary centroid sets.
template <typename Rows>
float distance_to_nearest_centroid(
        const Rows& rows,
        size_t n_centroids,
        size_t idx,
        const float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids,
        float* scratch) {
    if (n_centroids == 0 && n_existing_centroids == 0) {
        return std::numeric_limits<float>::infinity();
    }

    const float* point = rows.row(idx, scratch);
    float min_dist = std::numeric_limits<float>::max();

    auto check_centroids = [&]<SIMDLevel SL>() {
        for (size_t c = 0; c < n_centroids; c++) {
            float dist = fvec_L2sqr<SL>(point, centroids + c * rows.d, rows.d);
            min_dist = std::min(min_dist, dist);
        }

        for (size_t c = 0; c < n_existing_centroids; c++) {
            float dist = fvec_L2sqr<SL>(
                    point, existing_centroids + c * rows.d, rows.d);
            min_dist = std::min(min_dist, dist);
        }
    };
    with_rows_simd_level<Rows>(check_centroids);
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
template <typename Rows>
InitDistancesResult init_distances_for_d2_sampling(
        size_t n,
        const Rows& rows,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids,
        std::vector<double>& distances,
        std::mt19937_64& rng) {
    double sum_d2 = 0.0;
    size_t first_selected_idx = 0;
    auto scratch = row_scratch<Rows>(rows.d);

    if (n_existing_centroids > 0 && existing_centroids != nullptr) {
        for (size_t i = 0; i < n; i++) {
            distances[i] = distance_to_nearest_centroid(
                    rows,
                    n_existing_centroids,
                    i,
                    existing_centroids,
                    0,
                    nullptr,
                    scratch.data());
            sum_d2 += distances[i];
        }
        return {0, sum_d2, 0};
    } else {
        std::uniform_int_distribution<size_t> uniform_dist(0, n - 1);
        first_selected_idx = uniform_dist(rng);
        rows.copy_row(first_selected_idx, centroids);

        with_rows_simd_level<Rows>([&]<SIMDLevel SL>() {
            for (size_t i = 0; i < n; i++) {
                const float* point = rows.row(i, scratch.data());
                distances[i] = fvec_L2sqr<SL>(point, centroids, rows.d);
                sum_d2 += distances[i];
            }
        });
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

template <typename Rows>
void init_kmeans_plus_plus_impl(
        const ClusteringInitialization& initializer,
        size_t n,
        const Rows& rows,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) {
    std::mt19937_64 rng(get_seed(initializer.seed));

    std::vector<double> min_distances(n);
    auto result = init_distances_for_d2_sampling(
            n,
            rows,
            centroids,
            n_existing_centroids,
            existing_centroids,
            min_distances,
            rng);

    if (result.first_new_centroid_idx == 1 && initializer.k == 1) {
        return;
    }

    std::vector<double> cumsum(n);

    with_rows_simd_level<Rows>([&]<SIMDLevel SL>() {
        for (size_t c = result.first_new_centroid_idx; c < initializer.k; c++) {
            cumsum[0] = min_distances[0];
            for (size_t i = 1; i < n; i++) {
                cumsum[i] = cumsum[i - 1] + min_distances[i];
            }

            size_t next_idx = sample_from_cumsum(cumsum, rng);

            float* new_centroid = centroids + c * rows.d;
            rows.copy_row(next_idx, new_centroid);

            if constexpr (Rows::needs_scratch) {
                constexpr size_t kRowsPerBlock = 64;
                const int64_t nblocks = static_cast<int64_t>(
                        (n + kRowsPerBlock - 1) / kRowsPerBlock);
#pragma omp parallel
                {
                    std::vector<float> block(kRowsPerBlock * rows.d);
#pragma omp for
                    for (int64_t b = 0; b < nblocks; b++) {
                        const size_t first =
                                static_cast<size_t>(b) * kRowsPerBlock;
                        const size_t count = std::min(kRowsPerBlock, n - first);
                        rows.copy_rows(first, count, block.data());
                        for (size_t j = 0; j < count; j++) {
                            double dist = fvec_L2sqr<SL>(
                                    block.data() + j * rows.d,
                                    new_centroid,
                                    rows.d);
                            min_distances[first + j] =
                                    std::min(min_distances[first + j], dist);
                        }
                    }
                }
            } else {
                const int64_t ni = static_cast<int64_t>(n);
#pragma omp parallel for
                for (int64_t i = 0; i < ni; i++) {
                    double dist = fvec_L2sqr<SL>(
                            rows.row(i, nullptr), new_centroid, rows.d);
                    min_distances[i] = std::min(min_distances[i], dist);
                }
            }
        }
    });
}

template <typename Rows>
void init_afkmc2_impl(
        const ClusteringInitialization& initializer,
        size_t n,
        const Rows& rows,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) {
    std::mt19937_64 rng(get_seed(initializer.seed));
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);

    std::unordered_set<size_t> selected_centroids;

    std::vector<double> dist_to_nearest(n);
    auto result = init_distances_for_d2_sampling(
            n,
            rows,
            centroids,
            n_existing_centroids,
            existing_centroids,
            dist_to_nearest,
            rng);

    if (result.first_new_centroid_idx == 1) {
        selected_centroids.insert(result.first_selected_idx);
        if (initializer.k == 1) {
            return;
        }
    }

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

    auto scratch = row_scratch<Rows>(rows.d);
    for (size_t c = result.first_new_centroid_idx; c < initializer.k; c++) {
        size_t current_idx;
        do {
            current_idx = sample_from_cumsum(q_cumsum, rng);
        } while (selected_centroids.count(current_idx) > 0);

        double current_dist = distance_to_nearest_centroid(
                rows,
                c,
                current_idx,
                centroids,
                n_existing_centroids,
                existing_centroids,
                scratch.data());
        double current_q = q[current_idx];

        for (size_t m = 0; m < initializer.afkmc2_chain_length; m++) {
            size_t proposed_idx = sample_from_cumsum(q_cumsum, rng);

            if (selected_centroids.count(proposed_idx) > 0) {
                continue;
            }

            double proposed_dist = distance_to_nearest_centroid(
                    rows,
                    c,
                    proposed_idx,
                    centroids,
                    n_existing_centroids,
                    existing_centroids,
                    scratch.data());
            double proposed_q = q[proposed_idx];

            double acceptance_prob = 0.0;
            if (current_dist <= 0) {
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

        selected_centroids.insert(current_idx);
        rows.copy_row(current_idx, centroids + c * rows.d);
    }
}

void validate_initialization(
        const ClusteringInitialization& initializer,
        size_t n,
        const void* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) {
    FAISS_THROW_IF_NOT_FMT(
            n >= initializer.k,
            "Number of points (%zu) must be >= number of centroids (%zu)",
            n,
            initializer.k);
    FAISS_THROW_IF_NOT(initializer.d > 0);
    FAISS_THROW_IF_NOT(x);
    FAISS_THROW_IF_NOT(centroids);
    FAISS_THROW_IF_NOT(
            n_existing_centroids == 0 || existing_centroids != nullptr);
}

} // namespace

ClusteringInitialization::ClusteringInitialization(size_t d_in, size_t k_in)
        : d(d_in), k(k_in) {}

void ClusteringInitialization::init_centroids(
        size_t n,
        const float* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) const {
    validate_initialization(
            *this, n, x, centroids, n_existing_centroids, existing_centroids);

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
    init_kmeans_plus_plus_impl(
            *this,
            n,
            FloatRows{d, x},
            centroids,
            n_existing_centroids,
            existing_centroids);
}

void ClusteringInitialization::init_afkmc2(
        size_t n,
        const float* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) const {
    init_afkmc2_impl(
            *this,
            n,
            FloatRows{d, x},
            centroids,
            n_existing_centroids,
            existing_centroids);
}

namespace detail {

void init_centroids_fp16(
        const ClusteringInitialization& initializer,
        size_t n,
        const uint16_t* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) {
    validate_initialization(
            initializer,
            n,
            x,
            centroids,
            n_existing_centroids,
            existing_centroids);

    with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_BASE>([&]<SIMDLevel SL>() {
        switch (initializer.method) {
            case ClusteringInitMethod::KMEANS_PLUS_PLUS:
                init_kmeans_plus_plus_impl(
                        initializer,
                        n,
                        Fp16Rows<SL>{initializer.d, x},
                        centroids,
                        n_existing_centroids,
                        existing_centroids);
                break;
            case ClusteringInitMethod::AFK_MC2:
                init_afkmc2_impl(
                        initializer,
                        n,
                        Fp16Rows<SL>{initializer.d, x},
                        centroids,
                        n_existing_centroids,
                        existing_centroids);
                break;
            case ClusteringInitMethod::RANDOM:
                FAISS_THROW_MSG(
                        "fp16 centroid helper requires non-random "
                        "initialization");
            default:
                FAISS_THROW_MSG("Unknown initialization method");
        }
    });
}

} // namespace detail

} // namespace faiss
