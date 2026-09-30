/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ClusteringHelpers.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_impl/fp16_kernels.h>

namespace faiss {
namespace detail {

namespace {

/// Below this many values, the fp16 finiteness check stays on one thread.
constexpr size_t fp16_parallel_threshold = size_t(1) << 16;

/** Shared body of compute_centroids and compute_centroids_fp16.
 *
 * `add_row(i, w, c, scratch)` adds training vector i to the fp32 vector c,
 * scaled by *w when w is not null. `scratch` is a per-thread buffer of d
 * floats.
 */
template <class AddRow>
void compute_centroids_impl(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const int64_t* assign,
        const float* weights,
        float* hassign,
        float* centroids,
        const AddRow& add_row) {
    k -= k_frozen;
    centroids += k_frozen * d;

    memset(centroids, 0, sizeof(*centroids) * d * k);

    std::atomic<bool> invalid_assignment{false};
    const int64_t num_centroids = static_cast<int64_t>(k + k_frozen);
#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
        std::vector<float> scratch(d);

        for (size_t i = 0; i < n; i++) {
            int64_t ci = assign[i];
            if (ci < 0 || ci >= num_centroids) {
                invalid_assignment.store(true, std::memory_order_relaxed);
                continue;
            }
            ci -= k_frozen;
            if (ci >= static_cast<int64_t>(c0) &&
                ci < static_cast<int64_t>(c1)) {
                const float* w = weights ? weights + i : nullptr;
                hassign[ci] += w ? *w : 1.0f;
                add_row(i, w, centroids + ci * d, scratch.data());
            }
        }
    }
    FAISS_THROW_IF_MSG(
            invalid_assignment.load(std::memory_order_relaxed),
            "invalid cluster assignment");

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

} // namespace

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
        FAISS_THROW_IF_NOT_FMT(
                nx <= static_cast<idx_t>(std::numeric_limits<int>::max()),
                "Dataset too large (%" PRId64
                ") for standard subsampling; "
                "set use_faster_subsampling=true",
                nx);
        std::vector<int> int_perm(nx);
        rand_perm(int_perm.data(), nx, actual_seed);
        perm.assign(int_perm.begin(), int_perm.end());
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
    const size_t line_size = codec ? codec->sa_code_size() : d * sizeof(float);
    compute_centroids_impl(
            d,
            k,
            n,
            k_frozen,
            assign,
            weights,
            hassign,
            centroids,
            [&](size_t i, const float* w, float* c, float* scratch) {
                const float* xi;
                if (!codec) {
                    xi = reinterpret_cast<const float*>(x + i * line_size);
                } else {
                    codec->sa_decode(1, x + i * line_size, scratch);
                    xi = scratch;
                }
                if (w) {
                    const float wi = *w;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j] * wi;
                    }
                } else {
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j];
                    }
                }
            });
}

void compute_centroids_fp16(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const uint16_t* x,
        const int64_t* assign,
        const float* weights,
        float* hassign,
        float* centroids) {
    with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_AVX2_NEON>(
            [&]<SIMDLevel SL>() {
                compute_centroids_impl(
                        d,
                        k,
                        n,
                        k_frozen,
                        assign,
                        weights,
                        hassign,
                        centroids,
                        [&](size_t i,
                            const float* w,
                            float* c,
                            float* /*scratch*/) {
                            // x * 1.0f is exact, so unweighted sums match the
                            // fp32 path.
                            fp16_madd<SL>(d, x + i * d, w ? *w : 1.0f, c);
                        });
            });
}

bool fp16_all_finite(size_t n, const uint16_t* x) {
    // binary16 NaNs and infinities are the values with all exponent bits set
    constexpr uint16_t exponent_mask = 0x7c00;
    std::atomic<bool> all_finite{true};
#pragma omp parallel for if (n > fp16_parallel_threshold)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        if ((x[i] & exponent_mask) == exponent_mask) {
            all_finite.store(false, std::memory_order_relaxed);
        }
    }
    return all_finite.load(std::memory_order_relaxed);
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
