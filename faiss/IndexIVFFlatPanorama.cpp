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
#include <faiss/MetricType.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/impl/ResultHandler.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances_dispatch.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

IndexIVFFlatPanorama::IndexIVFFlatPanorama(
        Index* quantizer_in,
        size_t d_in,
        size_t nlist_in,
        int n_levels_in,
        MetricType metric,
        bool own_invlists_in,
        size_t batch_size_in)
        : IndexIVFFlat(quantizer_in, d_in, nlist_in, metric, false),
          n_levels(n_levels_in),
          batch_size(batch_size_in) {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);

    // We construct the inverted lists here so that we can use the
    // level-oriented storage. This does not cause a leak as we constructed
    // IndexIVF first, with own_invlists set to false.
    this->invlists = new ArrayInvertedListsPanorama(
            nlist, code_size, n_levels, batch_size);
    this->own_invlists = own_invlists_in;
}

IndexIVFFlatPanorama::IndexIVFFlatPanorama()
        : n_levels(0), batch_size(Panorama::kDefaultBatchSize) {}

namespace {

template <typename VectorDistance, bool use_sel>
struct IVFFlatScannerPanorama : InvertedListScanner {
    VectorDistance vd;
    const ArrayInvertedListsPanorama* storage;
    using C = typename VectorDistance::C;
    static constexpr MetricType metric = VectorDistance::metric;

    mutable std::vector<uint32_t> active_indices_;
    mutable std::vector<uint8_t> active_byteset_;
    mutable std::vector<float> exact_distances_;
    mutable std::vector<float> dot_buffer_;

    IVFFlatScannerPanorama(
            const VectorDistance& vd_in,
            const ArrayInvertedListsPanorama* storage_in,
            bool store_pairs_in,
            const IDSelector* sel_in)
            : InvertedListScanner(store_pairs_in, sel_in),
              vd(vd_in),
              storage(storage_in) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
        cum_sums.resize(storage->pano.n_levels + 1);
        active_indices_.resize(storage->pano.batch_size);
        active_byteset_.resize(storage->pano.batch_size);
        exact_distances_.resize(storage->pano.batch_size);
        dot_buffer_.resize(storage->pano.batch_size);
    }

    const float* xi = nullptr;
    std::vector<float> cum_sums;
    float q_norm = 0.0f;
    void set_query(const float* query) override {
        this->xi = query;
        this->storage->pano.compute_query_cum_sums(query, cum_sums.data());
        q_norm = cum_sums[0] * cum_sums[0];
    }

    void set_list(idx_t list_no_in, float /* coarse_dis */) override {
        this->list_no = list_no_in;
    }

    /// This function is unreachable as `IndexIVF` only calls this within
    /// iterators, which are not supported by `IndexIVFFlatPanorama`.
    /// To avoid undefined behavior, we throw an error here.
    float distance_to_code(const uint8_t* /* code */) const override {
        FAISS_THROW_MSG(
                "IndexIVFFlatPanorama does not support distance_to_code");
    }

    using InvertedListScanner::scan_codes;

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        size_t nup = 0;

        const size_t bs = storage->pano.batch_size;
        const size_t n_batches = (list_size + bs - 1) / bs;

        const float* cum_sums_data = storage->get_cum_sums(list_no);

        PanoramaStats local_stats;
        local_stats.reset();

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * bs;
            size_t num_active = with_metric_type(metric, [&]<MetricType M>() {
                return storage->pano.progressive_filter_batch<C, M>(
                        codes,
                        cum_sums_data,
                        xi,
                        cum_sums.data(),
                        batch_no,
                        list_size,
                        sel,
                        ids,
                        use_sel,
                        active_indices_,
                        active_byteset_,
                        exact_distances_,
                        dot_buffer_,
                        handler.threshold,
                        local_stats);
            });

            // num_active is the count of codes for which exact distance
            // was computed in this batch (post-filter, post-pruning).
            handler.stats.scan_cnt += num_active;

            // Add batch survivors to heap.
            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices_[i];
                size_t global_idx = batch_start + idx;
                float dis = exact_distances_[idx];

                if (C::cmp(handler.threshold, dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, global_idx)
                                             : ids[global_idx];
                    if (handler.add_result(dis, id)) {
                        handler.stats.nheap_updates++;
                        nup++;
                    }
                }
            }
        }

        indexPanorama_stats.add(local_stats);
        return nup;
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFFlatPanorama::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    const ArrayInvertedListsPanorama* storage =
            dynamic_cast<const ArrayInvertedListsPanorama*>(invlists);
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "IndexIVFFlatPanorama requires ArrayInvertedListsPanorama");

    return with_VectorDistance(
            d, metric_type, metric_arg, [&](auto vd) -> InvertedListScanner* {
                if (sel) {
                    return new IVFFlatScannerPanorama<decltype(vd), true>(
                            vd, storage, store_pairs, sel);
                } else {
                    return new IVFFlatScannerPanorama<decltype(vd), false>(
                            vd, storage, store_pairs, sel);
                }
            });
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
