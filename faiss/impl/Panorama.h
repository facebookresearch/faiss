/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_PANORAMA_H
#define FAISS_PANORAMA_H

#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(COMPILE_SIMD_AVX2) && defined(__AVX2__) && defined(__BMI2__)
#include <immintrin.h>
#endif

namespace faiss {

#ifndef SWIG

/// Compute dot products between query_level and active vectors.
///
/// @tparam AllActive  If true, vectors are at sequential positions 0..N-1
///                    (first level, full batch). If false, positions come
///                    from active_indices (subsequent levels after pruning).
/// @tparam LevelWidth Compile-time level width in floats (0 = use runtime
///                    level_width_dims). Enables full loop unrolling.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <bool AllActive = false, size_t LevelWidth = 0>
static inline void compute_level_dot_kernel(
        const float* FAISS_RESTRICT query_level,
        const float* FAISS_RESTRICT level_storage,
        const uint32_t* active_indices,
        const size_t num_active,
        const size_t level_width_dims,
        float* FAISS_RESTRICT dot_products) {
    const size_t width = LevelWidth > 0 ? LevelWidth : level_width_dims;
    size_t i = 0;
    for (; i + 4 <= num_active; i += 4) {
        const float* y0 = level_storage +
                (AllActive ? (i + 0) : active_indices[i + 0]) * width;
        const float* y1 = level_storage +
                (AllActive ? (i + 1) : active_indices[i + 1]) * width;
        const float* y2 = level_storage +
                (AllActive ? (i + 2) : active_indices[i + 2]) * width;
        const float* y3 = level_storage +
                (AllActive ? (i + 3) : active_indices[i + 3]) * width;

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        FAISS_PRAGMA_IMPRECISE_LOOP
        for (size_t j = 0; j < width; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
            dp2 += q * y2[j];
            dp3 += q * y3[j];
        }

        dot_products[i + 0] = dp0;
        dot_products[i + 1] = dp1;
        dot_products[i + 2] = dp2;
        dot_products[i + 3] = dp3;
    }
    for (; i < num_active; i++) {
        const float* yj =
                level_storage + (AllActive ? i : active_indices[i]) * width;
        float dp = 0;
        FAISS_PRAGMA_IMPRECISE_LOOP
        for (size_t j = 0; j < width; j++) {
            dp += query_level[j] * yj[j];
        }
        dot_products[i] = dp;
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/// Update exact distances with the current level's dot products, then apply
/// Panorama pruning: for each active vector, compute a lower bound on
/// the final distance and mark it for removal if it cannot beat the current
/// threshold. Writes 0/1 into active_byteset for subsequent compaction.
///
/// Uses `if constexpr` on C::is_max rather than C::cmp() to ensure the
/// comparison autovectorizes (C::cmp generates scalar function calls).
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <bool AllActive, typename C, MetricType M>
static inline void prune_kernel(
        float* FAISS_RESTRICT exact_distances,
        const float* FAISS_RESTRICT dot_buffer,
        const float* FAISS_RESTRICT level_cum_sums,
        uint8_t* FAISS_RESTRICT active_byteset,
        const uint32_t* FAISS_RESTRICT active_indices,
        const uint32_t num_active,
        const float query_cum_norm,
        const float threshold) {
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (uint32_t i = 0; i < num_active; i++) {
        uint32_t idx = AllActive ? i : active_indices[i];
        if constexpr (M == METRIC_INNER_PRODUCT) {
            exact_distances[idx] += dot_buffer[i];
        } else {
            exact_distances[idx] -= 2.0f * dot_buffer[i];
        }

        float cum_sum = level_cum_sums[idx];
        float cauchy_schwarz_bound;
        if constexpr (M == METRIC_INNER_PRODUCT) {
            cauchy_schwarz_bound = -cum_sum * query_cum_norm;
        } else {
            cauchy_schwarz_bound = 2.0f * cum_sum * query_cum_norm;
        }

        float lower_bound = exact_distances[idx] - cauchy_schwarz_bound;
        if constexpr (C::is_max) {
            active_byteset[i] = (threshold > lower_bound) ? 1 : 0;
        } else {
            active_byteset[i] = (threshold < lower_bound) ? 1 : 0;
        }
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/// Compact active_indices in-place, removing entries where active_byteset[i]
/// is zero. Returns the new count of active elements. Uses a branchless BMI2 +
/// AVX2 fast path (8 elements/iteration via _pext_u64 permutation) with a
/// scalar fallback for the tail and non-x86 platforms.
inline size_t compact_active_kernel(
        uint32_t* active_indices,
        const uint8_t* FAISS_RESTRICT active_byteset,
        const size_t num_active) {
    size_t next_active = 0;
    size_t i = 0;

#if defined(COMPILE_SIMD_AVX2) && defined(__AVX2__) && defined(__BMI2__)
    for (; i + 8 <= num_active; i += 8) {
        uint64_t bytes;
        memcpy(&bytes, &active_byteset[i], 8);

        uint64_t expanded = bytes * 0xFFULL;
        uint64_t packed = _pext_u64(0x0706050403020100ULL, expanded);

        __m256i perm = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((int64_t)packed));
        __m256i data = _mm256_loadu_si256((const __m256i*)&active_indices[i]);
        __m256i compacted = _mm256_permutevar8x32_epi32(data, perm);
        _mm256_storeu_si256((__m256i*)&active_indices[next_active], compacted);

        next_active += __builtin_popcountll(bytes);
    }
#endif

    for (; i < num_active; i++) {
        active_indices[next_active] = active_indices[i];
        next_active += active_byteset[i] ? 1 : 0;
    }

    return next_active;
}

/// Compile-time dispatch: converts a runtime `width` value into a template
/// parameter by generating an if-else chain over [Lo, Hi] in steps of Step.
/// Falls through to LevelWidth=0 (runtime path) if no specialization matches.
/// Allows for specialization of common level widths.
namespace detail {
template <size_t Lo, size_t Hi, size_t Step, typename Lambda>
inline auto dispatch_width(size_t width, Lambda&& fn) {
    if constexpr (Lo > Hi) {
        return fn.template operator()<0>();
    } else {
        if (width == Lo) {
            return fn.template operator()<Lo>();
        }
        return dispatch_width<Lo + Step, Hi, Step>(
                width, std::forward<Lambda>(fn));
    }
}
} // namespace detail

/// Specialize for common float level widths (multiples of 8 up to 128).
template <typename LambdaType>
inline auto with_level_width(size_t width, LambdaType&& action) {
    return detail::dispatch_width<8, 128, 8>(
            width, std::forward<LambdaType>(action));
}

template <typename Lambda>
inline auto with_bool(bool value, Lambda&& fn) {
    if (value) {
        return fn.template operator()<true>();
    } else {
        return fn.template operator()<false>();
    }
}
#endif // SWIG

/**
 * Implements the core logic of Panorama-based refinement.
 * arXiv: https://arxiv.org/abs/2510.00566
 *
 * Panorama partitions the dimensions of all vectors into L contiguous levels.
 * During the refinement stage of ANNS, it computes distances between the query
 * and its candidates level-by-level. After processing each level, it prunes the
 * candidates whose lower bound exceeds the k-th best distance.
 *
 * In order to enable speedups, the dimensions (or codes) of each vector are
 * stored in a batched, level-major manner. Within each batch of b vectors, the
 * dimensions corresponding to level 1 will be stored first (for all elements in
 * that batch), followed by level 2, and so on. This allows for efficient memory
 * access patterns.
 *
 * Coupled with the appropriate orthogonal PreTransform (e.g. PCA, Cayley,
 * etc.), Panorama can prune the vast majority of dimensions, greatly
 * accelerating the refinement stage.
 */
struct Panorama {
    static constexpr size_t kDefaultBatchSize = 128;

    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width = 0;
    size_t level_width_floats = 0;
    size_t batch_size = 0;

    explicit Panorama(size_t code_size, size_t n_levels, size_t batch_size);

    void set_derived_values();

    /// Helper method to copy codes into level-oriented batch layout at a given
    /// offset in the list.
    void copy_codes_to_level_layout(
            uint8_t* codes,
            size_t offset,
            size_t n_entry,
            const uint8_t* code);

    /// Helper method to compute the cumulative sums of the codes.
    /// The cumsums also follow the level-oriented batch layout to minimize the
    /// number of random memory accesses.
    void compute_cumulative_sums(
            float* cumsum_base,
            size_t offset,
            size_t n_entry,
            const float* vectors) const;

    /// Compute the cumulative sums of the query vector.
    void compute_query_cum_sums(const float* query, float* query_cum_sums)
            const;

    /// Copy single entry (code and cum_sum) from one location to another.
    void copy_entry(
            uint8_t* dest_codes,
            uint8_t* src_codes,
            float* dest_cum_sums,
            float* src_cum_sums,
            size_t dest_idx,
            size_t src_idx) const;

    /// Panorama's core progressive filtering algorithm:
    /// Process vectors in batches for cache efficiency. For each batch:
    /// 1. Apply ID selection filter and initialize distances
    /// (||y||^2 + ||x||^2).
    /// 2. Maintain an "active set" of candidate indices that haven't been
    /// pruned yet.
    /// 3. For each level, refine distances incrementally and compact the active
    /// set:
    ///    - Compute dot product for current level: exact_dist -= 2*<x,y>.
    ///    - Use Cauchy-Schwarz bound on remaining levels to get lower bound
    ///    - Prune candidates whose lower bound exceeds k-th best distance.
    ///    - Compact active_indices to remove pruned candidates (branchless)
    /// 4. After all levels, survivors are exact distances; update heap.
    /// This achieves early termination while maintaining SIMD-friendly
    /// sequential access patterns in the level-oriented storage layout.
#ifndef SWIG
    template <typename C, MetricType M>
    size_t progressive_filter_batch(
            const uint8_t* codes_base,
            const float* cum_sums,
            const float* query,
            const float* query_cum_sums,
            size_t batch_no,
            size_t list_size,
            const IDSelector* sel,
            const idx_t* ids,
            bool use_sel,
            std::vector<uint32_t>& active_indices,
            std::vector<uint8_t>& active_byteset,
            std::vector<float>& exact_distances,
            std::vector<float>& dot_buffer,
            float threshold,
            PanoramaStats& local_stats) const {
        size_t batch_start = batch_no * batch_size;
        size_t curr_batch_size = std::min(list_size - batch_start, batch_size);

        size_t cumsum_batch_offset = batch_no * batch_size * (n_levels + 1);
        const float* batch_cum_sums = cum_sums + cumsum_batch_offset;
        const float* level_cum_sums = batch_cum_sums + batch_size;
        float q_norm = query_cum_sums[0] * query_cum_sums[0];

        size_t batch_offset = batch_no * batch_size * code_size;
        const uint8_t* storage_base = codes_base + batch_offset;

        // Initialize active set with ID-filtered vectors.
        size_t num_active = 0;
        for (size_t i = 0; i < curr_batch_size; i++) {
            size_t global_idx = batch_start + i;
            idx_t id = (ids == nullptr) ? global_idx : ids[global_idx];
            bool include = !use_sel || sel->is_member(id);

            active_indices[num_active] = i;
            float cum_sum = batch_cum_sums[i];

            if constexpr (M == METRIC_INNER_PRODUCT) {
                exact_distances[i] = 0.0f;
            } else {
                exact_distances[i] = cum_sum * cum_sum + q_norm;
            }

            num_active += include;
        }

        size_t total_active = num_active;
        const bool first_level_full = (num_active == curr_batch_size);

        local_stats.total_dims += total_active * n_levels;

        for (size_t level = 0; (level < n_levels) && (num_active > 0);
             level++) {
            local_stats.total_dims_scanned += num_active;

            float query_cum_norm = query_cum_sums[level + 1];

            size_t level_offset = level * level_width * batch_size;
            const float* level_storage =
                    (const float*)(storage_base + level_offset);
            const float* query_level = query + level * level_width_floats;
            size_t actual_level_width = std::min(
                    level_width_floats, d - level * level_width_floats);

            num_active = with_bool(
                    level == 0 && first_level_full, [&]<bool AllActive>() {
                        with_level_width(
                                actual_level_width, [&]<size_t LevelWidth>() {
                                    compute_level_dot_kernel<
                                            AllActive,
                                            LevelWidth>(
                                            query_level,
                                            level_storage,
                                            active_indices.data(),
                                            num_active,
                                            actual_level_width,
                                            dot_buffer.data());
                                });

                        prune_kernel<AllActive, C, M>(
                                exact_distances.data(),
                                dot_buffer.data(),
                                level_cum_sums,
                                active_byteset.data(),
                                active_indices.data(),
                                (uint32_t)num_active,
                                query_cum_norm,
                                threshold);

                        return compact_active_kernel(
                                active_indices.data(),
                                active_byteset.data(),
                                num_active);
                    });

            level_cum_sums += batch_size;
        }

        return num_active;
    }
#endif // SWIG

    void reconstruct(idx_t key, float* recons, const uint8_t* codes_base) const;
};
} // namespace faiss

#endif
