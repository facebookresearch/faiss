/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFPQPanorama.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/PanoramaPQ.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>

namespace faiss {

/*****************************************
 * Constructor
 ******************************************/

IndexIVFPQPanorama::IndexIVFPQPanorama(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        int n_levels,
        size_t batch_size,
        MetricType metric,
        bool own_invlists)
        : IndexIVFPQ(quantizer, d, nlist, M, nbits_per_idx, metric, false),
          n_levels(n_levels),
          batch_size(batch_size),
          chunk_size(code_size / n_levels),
          levels_size(d / n_levels) {
    FAISS_THROW_IF_NOT_MSG(
            M % n_levels == 0, "M must be divisible by n_levels");
    FAISS_THROW_IF_NOT_MSG(
            batch_size % 64 == 0, "batch_size must be multiple of 64");
    FAISS_THROW_IF_NOT_MSG(nbits_per_idx == 8, "only 8-bit PQ codes supported");
    FAISS_THROW_IF_NOT_MSG(
            M == code_size, "M must equal code_size for 8-bit PQ");
    FAISS_THROW_IF_NOT_MSG(metric == METRIC_L2, "only L2 metric supported");

    auto* pano = new PanoramaPQ(d, code_size, n_levels, batch_size, &pq, quantizer);
    this->invlists = new ArrayInvertedListsPanorama(nlist, code_size, pano);
    this->own_invlists = own_invlists;
}

/*****************************************
 * Panorama scanner — overrides scan_codes with batch processing
 ******************************************/

namespace {

using idx_t = faiss::idx_t;

template <class C, bool use_sel>
struct IVFPQScannerPanorama : InvertedListScanner {
    const IndexIVFPQPanorama& index;
    const ProductQuantizer& pq;
    const ArrayInvertedListsPanorama* storage;
    const PanoramaPQ* pano_pq;

    // Query state
    const float* qi = nullptr;
    std::vector<float> query_cum_norms;
    std::vector<float> sim_table_2;

    // Per-list state
    float coarse_dis = 0;

    IVFPQScannerPanorama(
            const IndexIVFPQPanorama& index,
            const ArrayInvertedListsPanorama* storage,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel),
              index(index),
              pq(index.pq),
              storage(storage),
              pano_pq(dynamic_cast<const PanoramaPQ*>(storage->pano.get())) {
        FAISS_THROW_IF_NOT(pano_pq);
        this->keep_max = is_similarity_metric(index.metric_type);
        this->code_size = pq.code_size;
        query_cum_norms.resize(index.n_levels + 1);
        sim_table_2.resize(pq.M * pq.ksub);
    }

    void set_query(const float* query) override {
        this->qi = query;

        FAISS_ASSERT(index.by_residual);
        FAISS_ASSERT(index.use_precomputed_table == 1);

        pq.compute_inner_prod_table(qi, sim_table_2.data());

        // The PQ distance LUT is -2 * inner_prod_table; apply in-place
        // so scan_codes() can use sim_table_2 directly.
        const size_t n = pq.M * pq.ksub;
        for (size_t i = 0; i < n; i++) {
            sim_table_2[i] *= -2.0f;
        }

        pano_pq->compute_query_cum_sums(qi, query_cum_norms.data());
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        this->coarse_dis = coarse_dis;
    }

    float distance_to_code(const uint8_t* code) const override {
        FAISS_THROW_MSG("IndexIVFPQPanorama does not support distance_to_code");
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* /* codes (column-major in storage) */,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const override {
        size_t nup = 0;

        const size_t bs = index.batch_size;
        const size_t cs = index.chunk_size;

        const size_t n_batches = (list_size + bs - 1) / bs;
        const uint8_t* col_codes = storage->get_codes(list_no);
        const float* list_cum_sums = storage->get_cum_sums(list_no);
        const float* list_init_dists = storage->get_init_dists(list_no);

        // Scratch buffers.
        std::vector<float> exact_distances(bs);
        std::vector<uint8_t> bitset(bs);
        std::vector<uint32_t> active_indices(bs);
        std::vector<uint8_t> compressed_codes(bs * cs);
        float dis0 = coarse_dis;

        PanoramaStats local_stats;
        local_stats.reset();

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t num_active = pano_pq->progressive_filter_batch<C>(
                    col_codes,
                    list_cum_sums,
                    list_init_dists,
                    sim_table_2.data(),
                    query_cum_norms.data(),
                    dis0,
                    list_size,
                    batch_no,
                    exact_distances,
                    active_indices,
                    bitset,
                    compressed_codes,
                    distances[0],
                    local_stats);

            // Insert surviving candidates into heap.
            for (size_t i = 0; i < num_active; i++) {
                float dis = dis0 + exact_distances[i];
                if (C::cmp(distances[0], dis)) {
                    idx_t id = store_pairs
                            ? lo_build(list_no, active_indices[i])
                            : ids[active_indices[i]];
                    heap_replace_top<C>(k, distances, labels, dis, id);
                    nup++;
                }
            }
        }

        indexPanorama_stats.add(local_stats);
        return nup;
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        FAISS_THROW_MSG(
                "IndexIVFPQPanorama: ResultHandler scan_codes not supported");
    }
};

} // anonymous namespace

/*****************************************
 * get_InvertedListScanner
 ******************************************/

InvertedListScanner* IndexIVFPQPanorama::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2, "only L2 metric supported");
    FAISS_THROW_IF_NOT_MSG(
            use_precomputed_table == 1,
            "Panorama PQ requires use_precomputed_table == 1");
    FAISS_THROW_IF_NOT_MSG(pq.nbits == 8, "only 8-bit PQ codes supported");
    FAISS_THROW_IF_NOT_MSG(by_residual, "Panorama PQ requires by_residual");
    FAISS_THROW_IF_NOT_MSG(
            polysemous_ht == 0, "Panorama PQ does not support polysemous");

    const auto* storage =
            dynamic_cast<const ArrayInvertedListsPanorama*>(invlists);
    FAISS_THROW_IF_NOT_MSG(
            storage, "IndexIVFPQPanorama requires ArrayInvertedListsPanorama");

    if (sel) {
        return new IVFPQScannerPanorama<CMax<float, idx_t>, true>(
                *this, storage, store_pairs, sel);
    } else {
        return new IVFPQScannerPanorama<CMax<float, idx_t>, false>(
                *this, storage, store_pairs, sel);
    }
}

} // namespace faiss
