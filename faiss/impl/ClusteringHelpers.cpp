/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ClusteringHelpers.h>

#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/random.h>

namespace faiss {
namespace detail {

uint64_t get_actual_rng_seed(const int seed) {
    return (seed >= 0)
            ? seed
            : static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                                            .time_since_epoch()
                                            .count());
}

idx_t subsample_training_set(
        const Clustering& clus,
        idx_t nx,
        const uint8_t* x,
        size_t line_size,
        const float* weights,
        uint8_t** x_out,
        float** weights_out) {
    FAISS_THROW_IF_NOT(clus.k > 0 && clus.max_points_per_centroid > 0);
    if (clus.verbose) {
        printf("Sampling a subset of %zd / %" PRId64 " for training\n",
               clus.k * clus.max_points_per_centroid,
               nx);
    }

    const uint64_t actual_seed = get_actual_rng_seed(clus.seed);

    std::vector<idx_t> perm;
    if (clus.use_faster_subsampling) {
        SplitMix64RandomGenerator rng(actual_seed);

        const idx_t new_nx = clus.k * clus.max_points_per_centroid;
        perm.resize(new_nx);
        assert(!perm.empty());
        for (idx_t i = 0; i < new_nx; i++) {
            perm[i] = rng.rand_int64() % nx;
        }
    } else {
        // Partial Fisher-Yates shuffle: uniform without-replacement sampling
        // in O(target) time and O(target) memory.  The previous implementation
        // allocated two O(nx)-sized buffers (a 4-byte permutation array and an
        // 8-byte idx_t copy), causing OOM for large training sets — e.g.
        // nx=100 M yielded ~1.2 GB of temporaries for a ~2 MB output.
        // It also capped nx at INT_MAX, forcing users with large datasets onto
        // the with-replacement faster path.
        //
        // This implementation uses an unordered_map as a sparse swap table:
        // only positions touched by the partial shuffle are stored, bounding
        // memory at O(target) regardless of nx.  The statistical guarantee is
        // identical to a full Fisher-Yates: each of the C(nx, target) possible
        // subsets is equally likely, drawn in uniformly random order.
        const idx_t target = clus.k * clus.max_points_per_centroid;
        perm.resize(target);

        // sparse_swap[j] = current logical value at position j.  Positions
        // absent from the map hold their identity value (position j → value j).
        std::unordered_map<idx_t, idx_t> sparse_swap;
        sparse_swap.reserve(static_cast<size_t>(target) * 2);

        SplitMix64RandomGenerator rng(actual_seed);

        for (idx_t i = 0; i < target; i++) {
            // Pick j uniformly from [i, nx).
            const idx_t range = nx - i;
            const idx_t j = i + rng.rand_int64() % range;

            // Retrieve values at positions i and j; default to identity.
            auto it_j = sparse_swap.find(j);
            const idx_t val_j = (it_j != sparse_swap.end()) ? it_j->second : j;

            auto it_i = sparse_swap.find(i);
            const idx_t val_i = (it_i != sparse_swap.end()) ? it_i->second : i;

            // Draw the element at position j.
            perm[i] = val_j;

            // Move i's value to position j (completing the logical swap).
            if (val_i == j) {
                // Identity case — erase rather than storing the no-op j→j.
                sparse_swap.erase(j);
            } else {
                sparse_swap[j] = val_i;
            }

            // Position i is consumed; remove it to keep the map compact.
            sparse_swap.erase(i);
        }
    }

    nx = clus.k * clus.max_points_per_centroid;
    FAISS_THROW_IF_NOT_FMT(
            perm.size() >= static_cast<size_t>(nx),
            "subsample_training_set: perm size %zu < required nx %" PRId64,
            perm.size(),
            nx);
    assert(!perm.empty());

    uint8_t* x_new = new uint8_t[nx * line_size];
    *x_out = x_new;

    for (idx_t i = 0; i < nx; i++) {
        memcpy(x_new + i * line_size, x + perm[i] * line_size, line_size);
    }
    if (weights) {
        float* weights_new = new float[nx];
        for (idx_t i = 0; i < nx; i++) {
            weights_new[i] = weights[perm[i]];
        }
        *weights_out = weights_new;
    } else {
        *weights_out = nullptr;
    }
    return nx;
}

void compute_centroids(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const uint8_t* x,
        const Index* codec,
        const int64_t* assign,
        const float* weights,
        float* hassign,
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    memset(centroids, 0, sizeof(*centroids) * d * k);

    size_t line_size = codec ? codec->sa_code_size() : d * sizeof(float);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
        std::vector<float> decode_buffer(d);

        for (size_t i = 0; i < n; i++) {
            int64_t ci = assign[i];
            FAISS_THROW_IF_NOT_MSG(
                    ci >= 0 && ci < k + k_frozen, "invalid cluster assignment");
            ci -= k_frozen;
            if (ci >= static_cast<int64_t>(c0) &&
                ci < static_cast<int64_t>(c1)) {
                float* c = centroids + ci * d;
                const float* xi;
                if (!codec) {
                    xi = reinterpret_cast<const float*>(x + i * line_size);
                } else {
                    float* xif = decode_buffer.data();
                    codec->sa_decode(1, x + i * line_size, xif);
                    xi = xif;
                }
                if (weights) {
                    float w = weights[i];
                    hassign[ci] += w;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j] * w;
                    }
                } else {
                    hassign[ci] += 1.0;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j];
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (idx_t ci = 0; ci < static_cast<idx_t>(k); ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / hassign[ci];
        float* c = centroids + ci * d;
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}

// a bit above machine epsilon for float16
static constexpr float EPS = 1.f / 1024.f;

int split_clusters(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        float* hassign,
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;
    FAISS_THROW_IF_NOT_MSG(
            n > k,
            "split_clusters: n must exceed k to find a non-empty donor centroid");

    size_t nsplit = 0;
    RandomGenerator rng(1234);
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            // Probabilistic donor pick weighted by hassign; deterministic
            // fallback to the largest cluster if too many iterations pass.
            size_t cj;
            size_t max_tries = 10 * k;
            size_t n_tries = 0;
            bool found = false;
            for (cj = 0; n_tries < max_tries; cj = (cj + 1) % k) {
                float p = (hassign[cj] - 1.0) / (float)(n - k);
                float r = rng.rand_float();
                if (r < p) {
                    found = true;
                    break;
                }
                n_tries++;
            }
            if (!found) {
                // Deterministic fallback: split the largest cluster.
                cj = 0;
                for (size_t j = 1; j < k; j++) {
                    if (hassign[j] > hassign[cj]) {
                        cj = j;
                    }
                }
            }
            memcpy(centroids + ci * d,
                   centroids + cj * d,
                   sizeof(*centroids) * d);

            /* small symmetric perturbation */
            for (size_t j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }

    return static_cast<int>(nsplit);
}

} // namespace detail
} // namespace faiss
