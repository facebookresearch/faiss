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
/// @tparam AllActive  If true, vectors are at sequential row positions
///                    0..num_active-1 (first level, full batch, no prior
///                    pruning); the compiler then sees a simple strided
///                    access pattern and can vectorize aggressively.
///                    If false, row positions are gathered via
///                    `active_indices`.
/// @tparam LevelWidth Compile-time level width in floats (0 = use runtime
///                    `level_width_dims`). Enables full loop unrolling
///                    and permanent query-register allocation
///                    (e.g. for LevelWidth=32, the query stays in 2 zmm
///                    registers across all vectors regardless of dataset).
///
/// `stride` is the per-row stride in floats inside `level_storage`. Pass 0
/// to use a tightly-packed level layout where the row stride equals the
/// level width (legacy chunked Panorama). Pass a larger value (e.g. the
/// full row stride for a row-interleaved `[cum_sums | xb]` storage) to
/// stride past the cum-sums prefix and any other per-row payload.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <bool AllActive = false, size_t LevelWidth = 0>
static inline void compute_level_dot_kernel(
        const float* FAISS_RESTRICT query_level,
        const float* FAISS_RESTRICT level_storage,
        const uint32_t* active_indices,
        const size_t num_active,
        const size_t level_width_dims,
        float* FAISS_RESTRICT dot_products,
        size_t stride = 0) {
    const size_t width = LevelWidth > 0 ? LevelWidth : level_width_dims;
    const size_t row_stride = stride == 0 ? width : stride;
    size_t i = 0;
    for (; i + 4 <= num_active; i += 4) {
        const float* y0 = level_storage +
                (AllActive ? (i + 0) : active_indices[i + 0]) * row_stride;
        const float* y1 = level_storage +
                (AllActive ? (i + 1) : active_indices[i + 1]) * row_stride;
        const float* y2 = level_storage +
                (AllActive ? (i + 2) : active_indices[i + 2]) * row_stride;
        const float* y3 = level_storage +
                (AllActive ? (i + 3) : active_indices[i + 3]) * row_stride;

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
        const float* yj = level_storage +
                (AllActive ? i : active_indices[i]) * row_stride;
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

    /// When true, each batch in the `codes` buffer is prefixed with its
    /// own (n_levels + 1) * batch_size cum-sums block (laid out
    /// `[cs[0]_0..cs[0]_{B-1}, cs[1]_0..cs[1]_{B-1}, ..., cs[L]_0..cs[L]_{B-1}]`)
    /// followed by the same `n_levels` feature blocks the legacy chunked
    /// layout uses. Eliminates the separate `cum_sums` vector while
    /// keeping the feature region byte-identical to the legacy layout, so
    /// the SIMD kernels (`compute_level_dot_kernel`, `prune_kernel`)
    /// don't need a second variant.
    ///
    /// With `batch_size = 1` the per-row layout becomes
    /// `[cs[0], cs[1], ..., cs[L], feat[0], feat[1], ..., feat[L-1]]`,
    /// which packs each row's cum-sums at the start of the same cache
    /// line as its level-0 features \u2014 a 1 cache-line-fetch hot path for
    /// random-access workloads (e.g. HNSW).
    ///
    /// When false (legacy), `codes` holds only feature levels in the
    /// classic level-major batched layout and a separate `cum_sums` array
    /// is required.
    bool inline_layout = false;

    Panorama(size_t code_size,
             size_t n_levels,
             size_t batch_size,
             bool inline_layout = false);

    void set_derived_values();

    /// Number of cum-sum floats per batch (= (n_levels + 1) * batch_size).
    /// In inline mode this prefix sits at the start of every batch in
    /// `codes`; in chunked mode it lives at the same offset inside the
    /// separate `cum_sums` vector.
    size_t cs_floats_per_batch() const {
        return (n_levels + 1) * batch_size;
    }

    /// Byte size of a batch in inline-layout `codes`:
    ///   cum-sums prefix (cs_floats_per_batch * 4 bytes) + feature
    ///   region (n_levels * level_width * batch_size bytes).
    /// Returns 0 when inline_layout is false (codes holds only the
    /// feature region in that case, cum_sums lives in a separate vector).
    size_t inline_batch_bytes() const {
        return inline_layout
                ? cs_floats_per_batch() * sizeof(float) +
                        n_levels * level_width * batch_size
                : 0;
    }

    /// Byte offset of batch `batch_no`'s start within the `codes`
    /// buffer. Inline mode includes the cum-sums prefix; chunked mode
    /// does not.
    size_t batch_byte_offset(size_t batch_no) const {
        return inline_layout ? batch_no * inline_batch_bytes()
                             : batch_no * batch_size * code_size;
    }

    /// Byte offset of the feature region within a batch in `codes`.
    /// Inline mode skips the cum-sums prefix; chunked mode is 0.
    size_t feat_region_byte_offset() const {
        return inline_layout ? cs_floats_per_batch() * sizeof(float) : 0;
    }

    /// Helper method to copy codes into level-oriented batch layout at a given
    /// offset in the list.
    /// In inline mode, `codes` is expected to be sized so that each batch
    /// is `inline_batch_bytes()` long; this helper writes to the feature
    /// region, leaving the cum-sums prefix to `compute_cumulative_sums`.
    void copy_codes_to_level_layout(
            uint8_t* codes,
            size_t offset,
            size_t n_entry,
            const uint8_t* code);

    /// Helper method to compute the cumulative sums of the codes.
    /// The cumsums also follow the level-oriented batch layout to minimize the
    /// number of random memory accesses.
    /// `cumsum_base` points either at the start of a separate cum_sums
    /// vector (chunked layout) or at the start of the inline `codes`
    /// buffer (inline layout); the per-batch offset is the same in both
    /// cases since the layouts share the same per-batch cum-sums shape.
    void compute_cumulative_sums(
            float* cumsum_base,
            size_t offset,
            size_t n_entry,
            const float* vectors) const;

    /// Compute the cumulative sums of the query vector.
    void compute_query_cum_sums(const float* query, float* query_cum_sums)
            const;

    /// Copy single entry (code and cum_sum) from one location to another.
    /// In inline mode the caller passes `codes + batch0` for both codes
    /// pointers and `codes + 0` (reinterpreted as float*) for both
    /// cum_sums pointers \u2014 the per-batch offset math is identical to
    /// the chunked layout because the cs prefix shape didn't change.
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

        // cs prefix shape (per-batch (n_levels+1)*B floats) is identical
        // for both layouts, but the per-batch *stride* differs:
        //   chunked: cum_sums is a separate buffer packed tightly at
        //     stride cs_floats_per_batch() floats per batch.
        //   inline: cum_sums points into `codes_base` reinterpreted as
        //     float*, where each batch is inline_batch_bytes() long
        //     (cum-sums prefix + feature region), so the stride between
        //     batch starts is the larger inline batch width.
        const size_t cs_batch_stride = inline_layout
                ? (inline_batch_bytes() / sizeof(float))
                : cs_floats_per_batch();
        const size_t cumsum_batch_offset = batch_no * cs_batch_stride;
        const float* batch_cum_sums = cum_sums + cumsum_batch_offset;
        const float* level_cum_sums = batch_cum_sums + batch_size;
        float q_norm = query_cum_sums[0] * query_cum_sums[0];

        // In inline mode, the feature region sits past the cum-sums
        // prefix inside `codes_base`'s batch; in chunked mode it
        // starts at the batch boundary. `batch_byte_offset` and
        // `feat_region_byte_offset` collapse the two cases.
        const uint8_t* storage_base = codes_base + batch_byte_offset(batch_no) +
                feat_region_byte_offset();

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

    /// Reconstruct the raw float vector for `key` from the codes buffer.
    /// Works for both layouts: the only difference is the per-batch
    /// stride into `codes_base`, which `batch_byte_offset()` /
    /// `feat_region_byte_offset()` handle uniformly.
    void reconstruct(idx_t key, float* recons, const uint8_t* codes_base) const;
};
} // namespace faiss

#endif
