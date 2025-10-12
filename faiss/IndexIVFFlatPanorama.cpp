/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlatPanorama.h>

#include <omp.h>

#include <cstdio>

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexIVFFlatPanorama::IndexIVFFlatPanorama(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        bool own_invlists,
        int n_levels)
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

    const float* xi;
    std::vector<float> cum_sums;
    void set_query(const float* query) override {
        this->xi = query;

        const size_t d = vd.d;
        const size_t level_width = d / storage->n_levels;

        std::vector<float> suffix_sums(d + 1);
        suffix_sums[d] = 0.0f;

        for (int j = d - 1; j >= 0; j--) {
            float squared_val = query[j] * query[j];
            suffix_sums[j] = suffix_sums[j + 1] + squared_val;
        }

        for (size_t level = 0; level < storage->n_levels; level++) {
            size_t start_idx = level * level_width;
            if (start_idx < d) {
                cum_sums[level] = sqrt(suffix_sums[start_idx]);
            } else {
                cum_sums[level] = 0.0f;
            }
        }

        cum_sums[storage->n_levels] = 0.0f;
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    /// This function is unreachable as `IndexIVF` only calls this within
    /// iterators, which are not supported by `IndexIVFFlatPanorama`.
    /// To avoid undefined behavior, we throw an error here.
    float distance_to_code(const uint8_t* code) const override {
        FAISS_THROW_MSG(
                "IndexIVFFlatPanorama does not support distance_to_code");
    }

    // TODO(Alexis): Implement this!
    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        const float* list_vecs = (const float*)codes;
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + vd.d * j;
            if (use_sel && !sel->is_member(ids[j])) {
                continue;
            }
            float dis = vd(xi, yj);
            if (C::cmp(simi[0], dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                heap_replace_top<C>(k, simi, idxi, dis, id);
                nup++;
            }
        }
        return nup;
    }

    // TODO(Alexis): Implement this!
    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + vd.d * j;
            if (use_sel && !sel->is_member(ids[j])) {
                continue;
            }
            float dis = vd(xi, yj);
            if (C::cmp(radius, dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
            }
        }
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

} // namespace faiss