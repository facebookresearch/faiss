#ifndef FAISS_INDEX_IVFPQ_PANORAMA_H
#define FAISS_INDEX_IVFPQ_PANORAMA_H

#include <vector>

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct IndexIVFPQPanorama : public IndexIVFPQ {
    const int n_levels;
    uint8_t* column_storage;

    size_t* column_offsets;
    float* cum_sums;
    size_t* cum_sum_offsets;

    float* init_exact_distances;
    size_t* init_exact_distances_offsets;

    const size_t chunk_size;
    const size_t levels_size;
    bool added;
    size_t num_points;
    size_t batch_size;
    size_t nbits_per_idx;
    size_t m_level_width;

    float epsilon;

    IndexIVFPQPanorama(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            int n_levels,
            float epsilon,
            size_t batch_size = 128,
            MetricType metric = METRIC_L2,
            bool own_invlists = true);

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params_in) const;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* keys,
            const float* coarse_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params,
            IndexIVFStats* ivf_stats) const override;
};

} // namespace faiss

#endif
