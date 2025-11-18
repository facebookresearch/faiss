/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlatPanorama.h>

#include <cstdio>

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

IndexIVFFlatPanorama::IndexIVFFlatPanorama(
        Index* quantizer,
        size_t d,
        size_t nlist,
        int n_levels,
        MetricType metric,
        bool own_invlists)
        : IndexIVFFlat(quantizer, d, nlist, metric, false), n_levels(n_levels) {
    // For now, we only support L2 distance.
    // Supporting dot product and cosine distance is a trivial addition
    // left for future work.
    FAISS_THROW_IF_NOT(metric == METRIC_L2);

    // We construct the inverted lists here so that we can use the
    // level-oriented storage. This does not cause a leak as we constructed
    // IndexIVF first, with own_invlists set to false.
    this->invlists = new ArrayInvertedListsPanorama(nlist, code_size, n_levels);
    this->own_invlists = own_invlists;
}

IndexIVFFlatPanorama::IndexIVFFlatPanorama() : n_levels(0) {}

namespace {

template <typename VectorDistance, bool use_sel>
struct IVFFlatScannerPanorama : InvertedListScanner {
    VectorDistance vd;
    const ArrayInvertedListsPanorama* storage;
    using C = typename VectorDistance::C;

    IVFFlatScannerPanorama(
            const VectorDistance& vd,
            const ArrayInvertedListsPanorama* storage,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd), storage(storage) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
        cum_sums.resize(storage->n_levels + 1);
    }

    const float* xi = nullptr;
    std::vector<float> cum_sums;
    float q_norm = 0.0f;
    void set_query(const float* query) override {
        this->xi = query;
        this->storage->pano.compute_query_cum_sums(query, cum_sums.data());
        q_norm = cum_sums[0] * cum_sums[0];
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    /// This function is unreachable as `IndexIVF` only calls this within
    /// iterators, which are not supported by `IndexIVFFlatPanorama`.
    /// To avoid undefined behavior, we throw an error here.
    float distance_to_code(const uint8_t* /* code */) const override {
        FAISS_THROW_MSG(
                "IndexIVFFlatPanorama does not support distance_to_code");
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;

        const size_t n_batches =
                (list_size + storage->kBatchSize - 1) / storage->kBatchSize;

        const float* cum_sums_data = storage->get_cum_sums(list_no);

        std::vector<float> exact_distances(storage->kBatchSize);
        std::vector<uint32_t> active_indices(storage->kBatchSize);

        PanoramaStats local_stats;
        local_stats.reset();

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * storage->kBatchSize;

            size_t num_active =
                    storage->pano
                            .progressive_filter_batch<CMax<float, int64_t>>(
                                    codes,
                                    cum_sums_data,
                                    xi,
                                    cum_sums.data(),
                                    batch_no,
                                    list_size,
                                    sel,
                                    ids,
                                    use_sel,
                                    active_indices,
                                    exact_distances,
                                    simi[0],
                                    local_stats);

            // Add batch survivors to heap.
            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];
                size_t global_idx = batch_start + idx;
                float dis = exact_distances[idx];

                if (C::cmp(simi[0], dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, global_idx)
                                             : ids[global_idx];
                    heap_replace_top<C>(k, simi, idxi, dis, id);
                    nup++;
                }
            }
        }

        indexPanorama_stats.add(local_stats);
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const size_t n_batches =
                (list_size + storage->kBatchSize - 1) / storage->kBatchSize;

        const float* cum_sums_data = storage->get_cum_sums(list_no);

        std::vector<float> exact_distances(storage->kBatchSize);
        std::vector<uint32_t> active_indices(storage->kBatchSize);

        PanoramaStats local_stats;
        local_stats.reset();

        // Same progressive filtering as scan_codes, but with fixed radius
        // threshold instead of dynamic heap threshold.
        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * storage->kBatchSize;

            size_t num_active =
                    storage->pano
                            .progressive_filter_batch<CMax<float, int64_t>>(
                                    codes,
                                    cum_sums_data,
                                    xi,
                                    cum_sums.data(),
                                    batch_no,
                                    list_size,
                                    sel,
                                    ids,
                                    use_sel,
                                    active_indices,
                                    exact_distances,
                                    radius,
                                    local_stats);

            // Add batch survivors to range result.
            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];
                size_t global_idx = batch_start + idx;
                float dis = exact_distances[idx];

                if (C::cmp(radius, dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, global_idx)
                                             : ids[global_idx];
                    res.add(dis, id);
                }
            }
        }

        indexPanorama_stats.add(local_stats);
    }
};

struct Run_get_InvertedListScanner {
    using T = InvertedListScanner*;

    template <class VD>
    InvertedListScanner* f(
            VD& vd,
            const IndexIVFFlatPanorama* ivf,
            bool store_pairs,
            const IDSelector* sel) {
        // Safely cast to ArrayInvertedListsPanorama to access cumulative sums.
        const ArrayInvertedListsPanorama* storage =
                dynamic_cast<const ArrayInvertedListsPanorama*>(ivf->invlists);
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "IndexIVFFlatPanorama requires ArrayInvertedListsPanorama");

        if (sel) {
            return new IVFFlatScannerPanorama<VD, true>(
                    vd, storage, store_pairs, sel);
        } else {
            return new IVFFlatScannerPanorama<VD, false>(
                    vd, storage, store_pairs, sel);
        }
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFFlatPanorama::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    Run_get_InvertedListScanner run;
    return dispatch_VectorDistance(
            d, metric_type, metric_arg, run, this, store_pairs, sel);
}

void IndexIVFFlatPanorama::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);
    memcpy(recons, code, code_size);
    invlists->release_codes(list_no, code);
}

} // namespace faiss
