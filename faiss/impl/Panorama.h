/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_PANORAMA_H
#define FAISS_PANORAMA_H

#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace faiss {

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
            const float* vectors);

    /// Compute the cumulative sums of the query vector.
    void compute_query_cum_sums(const float* query, float* query_cum_sums)
            const;

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
    template <typename C>
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
            std::vector<float>& exact_distances,
            float threshold,
            PanoramaStats& local_stats) const;

    void reconstruct(idx_t key, float* recons, const uint8_t* codes_base) const;
};

template <typename C>
size_t Panorama::progressive_filter_batch(
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
        std::vector<float>& exact_distances,
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
        exact_distances[i] = cum_sum * cum_sum + q_norm;

        num_active += include;
    }

    if (num_active == 0) {
        return 0;
    }

    size_t total_active = num_active;
    for (size_t level = 0; level < n_levels; level++) {
        local_stats.total_dims_scanned += num_active;
        local_stats.total_dims += total_active;

        float query_cum_norm = query_cum_sums[level + 1];

        size_t level_offset = level * level_width * batch_size;
        const float* level_storage =
                (const float*)(storage_base + level_offset);

        size_t next_active = 0;
        for (size_t i = 0; i < num_active; i++) {
            uint32_t idx = active_indices[i];
            size_t actual_level_width = std::min(
                    level_width_floats, d - level * level_width_floats);

            const float* yj = level_storage + idx * actual_level_width;
            const float* query_level = query + level * level_width_floats;

            float dot_product =
                    fvec_inner_product(query_level, yj, actual_level_width);

            exact_distances[idx] -= 2.0f * dot_product;

            float cum_sum = level_cum_sums[idx];
            float cauchy_schwarz_bound = 2.0f * cum_sum * query_cum_norm;
            float lower_bound = exact_distances[idx] - cauchy_schwarz_bound;

            active_indices[next_active] = idx;
            next_active += C::cmp(threshold, lower_bound) ? 1 : 0;
        }

        num_active = next_active;
        level_cum_sums += batch_size;
    }

    return num_active;
}
} // namespace faiss

#endif
