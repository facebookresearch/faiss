/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/impl/Panorama.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/panorama_kernels/panorama_kernels.h>
#include <faiss/utils/Heap.h>
#include <cmath>

#include <numeric>

namespace faiss {

/**
 * Panorama for PQ-compressed vectors.
 *
 * Codes are PQ codes (code_size = M bytes for 8-bit PQ).
 * compute_cumulative_sums decodes via PQ then computes suffix norms.
 * progressive_filter_batch uses LUT accumulation with panorama_kernels.
 */
struct PanoramaPQ : Panorama {
    const ProductQuantizer* pq = nullptr;
    const Index* quantizer = nullptr;
    size_t chunk_size = 0;
    size_t levels_size = 0;

    PanoramaPQ() = default;
    PanoramaPQ(
            size_t d,
            size_t code_size,
            size_t n_levels,
            size_t batch_size,
            const ProductQuantizer* pq,
            const Index* quantizer = nullptr);

    void copy_codes_to_level_layout(
            uint8_t* codes,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) override;

    void reconstruct(idx_t key, float* recons, const uint8_t* codes_base)
            const override;

    void compute_cumulative_sums(
            float* cumsum_base,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) const override;

    /// Precompute per-point init distances: ||r||^2 + 2<r, c>.
    /// Requires quantizer to be set. Layout is flat per-list,
    /// padded to batch_size boundaries.
    void compute_init_distances(
            float* init_dists_base,
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) const;

    /// Progressive filtering for PQ codes: processes one batch.
    ///
    /// Initializes exact_distances from precomputed init_dists
    /// (||r||^2 + 2<r, c>), then refines with the query-specific
    /// sim_table_2 level-by-level with Cauchy-Schwarz pruning.
    ///
    /// @param col_codes       Column-major codes for this inverted list.
    /// @param list_cum_sums   Cumulative sums for this inverted list.
    /// @param init_dists      Precomputed init distances for this list.
    /// @param sim_table_2     -2 * inner_prod_table (query-specific LUT).
    /// @param query_cum_norms Query suffix norms per level.
    /// @param coarse_dis      Coarse distance (dis0) for this list.
    /// @param list_size       Total number of vectors in this list.
    /// @param batch_no        Which batch to process.
    /// @param exact_distances [out] Scratch buffer for partial distances.
    /// @param active_indices  [out] Scratch buffer for survivor indices.
    /// @param bitset          Scratch buffer for code compression.
    /// @param compressed_codes Scratch buffer for compressed codes.
    /// @param threshold       Current heap threshold for pruning.
    /// @param local_stats     [out] Accumulated pruning statistics.
    /// @return Number of surviving candidates in active_indices.
    template <typename C>
    size_t progressive_filter_batch(
            const uint8_t* col_codes,
            const float* list_cum_sums,
            const float* init_dists,
            const float* sim_table_2,
            const float* query_cum_norms,
            float coarse_dis,
            size_t list_size,
            size_t batch_no,
            std::vector<float>& exact_distances,
            std::vector<uint32_t>& active_indices,
            std::vector<uint8_t>& bitset,
            std::vector<uint8_t>& compressed_codes,
            float threshold,
            PanoramaStats& local_stats) const;
};

template <typename C>
size_t PanoramaPQ::progressive_filter_batch(
        const uint8_t* col_codes,
        const float* list_cum_sums,
        const float* init_dists,
        const float* sim_table_2,
        const float* query_cum_norms,
        float coarse_dis,
        size_t list_size,
        size_t batch_no,
        std::vector<float>& exact_distances,
        std::vector<uint32_t>& active_indices,
        std::vector<uint8_t>& bitset,
        std::vector<uint8_t>& compressed_codes,
        float threshold,
        PanoramaStats& local_stats) const {
    const size_t bs = batch_size;
    const size_t cs = chunk_size;
    const size_t ksub = pq->ksub;

    size_t curr_batch_size = std::min(list_size - batch_no * bs, bs);
    size_t b_offset = batch_no * bs;

    // Initialize active set.
    std::iota(
            active_indices.begin(),
            active_indices.begin() + curr_batch_size,
            b_offset);
    std::fill(bitset.begin(), bitset.begin() + curr_batch_size, 1);
    std::fill(bitset.begin() + curr_batch_size, bitset.end(), 0);

    const uint8_t* batch_codes = col_codes + b_offset * code_size;

    // Load precomputed init distances (||r||^2 + 2<r, c>).
    const float* batch_init = init_dists + b_offset;
    std::copy(batch_init, batch_init + curr_batch_size, exact_distances.begin());

    const float* batch_cums = list_cum_sums + b_offset * (n_levels + 1);

    size_t next_num_active = curr_batch_size;
    size_t batch_offset = batch_no * bs;
    const size_t total_active = next_num_active;

    for (size_t level = 0; level < n_levels && next_num_active > 0; level++) {
        local_stats.total_dims_scanned += next_num_active;
        local_stats.total_dims += total_active;

        size_t level_sim_offset = level * ksub * cs;

        float query_cum_norm = 2 * query_cum_norms[level + 1];

        const float* cum_sums_level = batch_cums + bs * (level + 1);
        const uint8_t* codes_level = batch_codes + bs * cs * level;

        const float* sim_table_level = sim_table_2 + level_sim_offset;

        bool is_sparse = next_num_active < bs / 16;

        size_t num_active_for_filtering = 0;
        if (is_sparse) {
            for (size_t ci = 0; ci < cs; ci++) {
                size_t chunk_off = ci * bs;
                const float* chunk_sim = sim_table_level + ci * ksub;
                for (size_t i = 0; i < next_num_active; i++) {
                    size_t real_idx = active_indices[i] - batch_offset;
                    exact_distances[i] +=
                            chunk_sim[codes_level[chunk_off + real_idx]];
                }
            }
            num_active_for_filtering = next_num_active;
        } else {
            auto [cc, na] = panorama_kernels::process_code_compression(
                    next_num_active,
                    bs,
                    cs,
                    compressed_codes.data(),
                    bitset.data(),
                    codes_level);

            panorama_kernels::process_chunks(
                    cs,
                    bs,
                    na,
                    const_cast<float*>(sim_table_level),
                    cc,
                    exact_distances.data());
            num_active_for_filtering = na;
        }

        next_num_active = panorama_kernels::process_filtering(
                num_active_for_filtering,
                exact_distances.data(),
                active_indices.data(),
                const_cast<float*>(cum_sums_level),
                bitset.data(),
                batch_offset,
                coarse_dis,
                query_cum_norm,
                threshold);
    }

    return next_num_active;
}

} // namespace faiss
