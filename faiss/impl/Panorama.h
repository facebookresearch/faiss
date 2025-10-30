/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_PANORAMA_H
#define FAISS_PANORAMA_H

#include <faiss/impl/PanoramaStats.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace faiss {

struct Panorama {
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

    /// Processes a batch of vectors through all levels,
    /// computing exact distances and pruning based on a threshold.
    /// Returns the number of active survivors after all levels.
    template <typename C>
    size_t progressive_filter_batch(
            const uint8_t* storage_base,
            const float* level_cum_sums,
            const float* query,
            const float* query_cum_sums,
            std::vector<uint32_t>& active_indices,
            std::vector<float>& exact_distances,
            size_t num_active,
            float threshold,
            PanoramaStats& local_stats) const;

    //    private:
    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width = 0;
    size_t level_width_floats = 0;
    size_t batch_size = 0;
};

template <typename C>
size_t Panorama::progressive_filter_batch(
        const uint8_t* storage_base,
        const float* level_cum_sums,
        const float* query,
        const float* query_cum_sums,
        std::vector<uint32_t>& active_indices,
        std::vector<float>& exact_distances,
        size_t num_active,
        float threshold,
        PanoramaStats& local_stats) const {
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
            const float* yj = level_storage + idx * level_width_floats;
            const float* query_level = query + level * level_width_floats;

            size_t actual_level_width = std::min(
                    level_width_floats, d - level * level_width_floats);
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
