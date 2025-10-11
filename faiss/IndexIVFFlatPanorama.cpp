/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlatPanorama.h>

#include <omp.h>

#include <cinttypes>
#include <cstdio>
#include <numeric>

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

// TODO(Alexis): Take into account the level-oriented storage.
void IndexIVFFlatPanorama::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(!by_residual);
    if (!include_listnos) {
        memcpy(codes, x, code_size * n);
    } else {
        size_t coarse_size = coarse_code_size();
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            uint8_t* code = codes + i * (code_size + coarse_size);
            const float* xi = x + i * d;
            if (list_no >= 0) {
                encode_listno(list_no, code);
                memcpy(code + coarse_size, xi, code_size);
            } else {
                memset(code, 0, code_size + coarse_size);
            }
        }
    }
}

// TODO(Alexis): Take into account the level-oriented storage.
void IndexIVFFlatPanorama::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* /*listnos*/,
        float* x) const {
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = codes + i * code_size;
        float* xi = x + i * d;
        memcpy(xi, code, code_size);
    }
}

// TODO(Alexis): idk what this is yet.
void IndexIVFFlatPanorama::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    size_t coarse_size = coarse_code_size();
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = bytes + i * (code_size + coarse_size);
        float* xi = x + i * d;
        memcpy(xi, code + coarse_size, code_size);
    }
}

namespace {

// TODO(Alexis): We should have a reference to the ArrayInvertedListsPanorama
// here. It does not seem to unreasonable to adapt scan_codes to achieve this?
// Or perhaps just pass it into the constructor?
// NEVERMIND: We have access to the `this` reference. This is amazing!
// We can access the state :-)
template <typename VectorDistance, bool use_sel>
struct IVFFlatScannerPanorama : InvertedListScanner {
    VectorDistance vd;
    using C = typename VectorDistance::C;

    IVFFlatScannerPanorama(
            const VectorDistance& vd,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
    }

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
        // TODO(Alexis): Compute the cumulative sums of the query.
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* code) const override {
        const float* yj = (float*)code;
        return vd(xi, yj);
    }

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

    // TODO(Alexis): We never tested this in Panorama!
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

    // TODO(Alexis): See how we dispatch to ivf here!
    template <class VD>
    InvertedListScanner* f(
            VD& vd,
            const IndexIVFFlatPanorama* ivf,
            bool store_pairs,
            const IDSelector* sel) {
        if (sel) {
            return new IVFFlatScannerPanorama<VD, true>(vd, store_pairs, sel);
        } else {
            return new IVFFlatScannerPanorama<VD, false>(vd, store_pairs, sel);
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
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

} // namespace faiss