/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFSpectralHash.h>

#include <algorithm>
#include <cstdint>
#include <memory>

#include <faiss/IndexLSH.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>

namespace faiss {

IndexIVFSpectralHash::IndexIVFSpectralHash(
        Index* quantizer,
        size_t d,
        size_t nlist,
        int nbit,
        float period)
        : IndexIVF(quantizer, d, nlist, (nbit + 7) / 8, METRIC_L2),
          nbit(nbit),
          period(period) {
    RandomRotationMatrix* rr = new RandomRotationMatrix(d, nbit);
    rr->init(1234);
    vt = rr;
    is_trained = false;
    by_residual = false;
}

IndexIVFSpectralHash::IndexIVFSpectralHash() : IndexIVF() {
    by_residual = false;
}

IndexIVFSpectralHash::~IndexIVFSpectralHash() {
    if (own_fields) {
        delete vt;
    }
}

namespace {

float median(size_t n, float* x) {
    std::sort(x, x + n);
    if (n % 2 == 1) {
        return x[n / 2];
    } else {
        return (x[n / 2 - 1] + x[n / 2]) / 2;
    }
}

} // namespace

void IndexIVFSpectralHash::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* assign) {
    if (!vt->is_trained) {
        vt->train(n, x);
    }
    FAISS_THROW_IF_NOT(!by_residual);

    if (threshold_type == Thresh_global) {
        // nothing to do
        return;
    } else if (
            threshold_type == Thresh_centroid ||
            threshold_type == Thresh_centroid_half) {
        // convert all centroids with vt
        std::vector<float> centroids(nlist * d);
        quantizer->reconstruct_n(0, nlist, centroids.data());
        trained.resize(nlist * nbit);
        vt->apply_noalloc(nlist, centroids.data(), trained.data());
        if (threshold_type == Thresh_centroid_half) {
            for (size_t i = 0; i < nlist * nbit; i++) {
                trained[i] -= 0.25 * period;
            }
        }
        return;
    }
    // otherwise train medians

    // assign
    std::unique_ptr<idx_t[]> idx(new idx_t[n]);
    quantizer->assign(n, x, idx.get());

    std::vector<size_t> sizes(nlist + 1);
    for (size_t i = 0; i < n; i++) {
        FAISS_THROW_IF_NOT(idx[i] >= 0);
        sizes[idx[i]]++;
    }

    size_t ofs = 0;
    for (int j = 0; j < nlist; j++) {
        size_t o0 = ofs;
        ofs += sizes[j];
        sizes[j] = o0;
    }

    // transform
    std::unique_ptr<float[]> xt(vt->apply(n, x));

    // transpose + reorder
    std::unique_ptr<float[]> xo(new float[n * nbit]);

    for (size_t i = 0; i < n; i++) {
        size_t idest = sizes[idx[i]]++;
        for (size_t j = 0; j < nbit; j++) {
            xo[idest + n * j] = xt[i * nbit + j];
        }
    }

    trained.resize(n * nbit);
    // compute medians
#pragma omp for
    for (int i = 0; i < nlist; i++) {
        size_t i0 = i == 0 ? 0 : sizes[i - 1];
        size_t i1 = sizes[i];
        for (int j = 0; j < nbit; j++) {
            float* xoi = xo.get() + i0 + n * j;
            if (i0 == i1) { // nothing to train
                trained[i * nbit + j] = 0.0;
            } else if (i1 == i0 + 1) {
                trained[i * nbit + j] = xoi[0];
            } else {
                trained[i * nbit + j] = median(i1 - i0, xoi);
            }
        }
    }
}

namespace {

void binarize_with_freq(
        size_t nbit,
        float freq,
        const float* x,
        const float* c,
        uint8_t* codes) {
    memset(codes, 0, (nbit + 7) / 8);
    for (size_t i = 0; i < nbit; i++) {
        float xf = (x[i] - c[i]);
        int64_t xi = int64_t(floor(xf * freq));
        int64_t bit = xi & 1;
        codes[i >> 3] |= bit << (i & 7);
    }
}

} // namespace

void IndexIVFSpectralHash::encode_vectors(
        idx_t n,
        const float* x_in,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(!by_residual);
    float freq = 2.0 / period;
    size_t coarse_size = include_listnos ? coarse_code_size() : 0;

    // transform with vt
    std::unique_ptr<float[]> x(vt->apply(n, x_in));

    std::vector<float> zero(nbit);

#pragma omp for
    for (idx_t i = 0; i < n; i++) {
        int64_t list_no = list_nos[i];
        uint8_t* code = codes + i * (code_size + coarse_size);

        if (list_no >= 0) {
            if (coarse_size) {
                encode_listno(list_no, code);
            }
            const float* c;

            if (threshold_type == Thresh_global) {
                c = zero.data();
            } else {
                c = trained.data() + list_no * nbit;
            }
            binarize_with_freq(
                    nbit, freq, x.get() + i * nbit, c, code + coarse_size);
        } else {
            memset(code, 0, code_size + coarse_size);
        }
    }
}

namespace {

template <class HammingComputer>
struct IVFScanner : InvertedListScanner {
    // copied from index structure
    const IndexIVFSpectralHash* index;
    size_t nbit;

    float period, freq;
    std::vector<float> q;
    std::vector<float> zero;
    std::vector<uint8_t> qcode;
    HammingComputer hc;

    IVFScanner(const IndexIVFSpectralHash* index, bool store_pairs)
            : index(index),
              nbit(index->nbit),
              period(index->period),
              freq(2.0 / index->period),
              q(nbit),
              zero(nbit),
              qcode(index->code_size),
              hc(qcode.data(), index->code_size) {
        this->store_pairs = store_pairs;
        this->code_size = index->code_size;
        this->keep_max = is_similarity_metric(index->metric_type);
    }

    void set_query(const float* query) override {
        FAISS_THROW_IF_NOT(query);
        FAISS_THROW_IF_NOT(q.size() == nbit);
        index->vt->apply_noalloc(1, query, q.data());

        if (index->threshold_type == IndexIVFSpectralHash::Thresh_global) {
            binarize_with_freq(nbit, freq, q.data(), zero.data(), qcode.data());
            hc.set(qcode.data(), code_size);
        }
    }

    void set_list(idx_t list_no, float /*coarse_dis*/) override {
        this->list_no = list_no;
        if (index->threshold_type != IndexIVFSpectralHash::Thresh_global) {
            const float* c = index->trained.data() + list_no * nbit;
            binarize_with_freq(nbit, freq, q.data(), c, qcode.data());
            hc.set(qcode.data(), code_size);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return hc.hamming(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            float dis = hc.hamming(codes);

            if (dis < simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++) {
            float dis = hc.hamming(codes);
            if (dis < radius) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
            }
            codes += code_size;
        }
    }
};

struct BuildScanner {
    using T = InvertedListScanner*;

    template <class HammingComputer>
    static T f(const IndexIVFSpectralHash* index, bool store_pairs) {
        return new IVFScanner<HammingComputer>(index, store_pairs);
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFSpectralHash::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    FAISS_THROW_IF_NOT(!sel);
    BuildScanner bs;
    return dispatch_HammingComputer(code_size, bs, this, store_pairs);
}

void IndexIVFSpectralHash::replace_vt(VectorTransform* vt_in, bool own) {
    FAISS_THROW_IF_NOT(vt_in->d_out == nbit);
    FAISS_THROW_IF_NOT(vt_in->d_in == d);
    if (own_fields) {
        delete vt;
    }
    vt = vt_in;
    threshold_type = Thresh_global;
    is_trained = quantizer->is_trained && quantizer->ntotal == nlist &&
            vt->is_trained;
    own_fields = own;
}

/*
    Check that the encoder is a single vector transform followed by a LSH
    that just does thresholding.
    If this is not the case, the linear transform + threhsolds of the IndexLSH
    should be merged into the VectorTransform (which is feasible).
*/

void IndexIVFSpectralHash::replace_vt(IndexPreTransform* encoder, bool own) {
    FAISS_THROW_IF_NOT(encoder->chain.size() == 1);
    auto sub_index = dynamic_cast<IndexLSH*>(encoder->index);
    FAISS_THROW_IF_NOT_MSG(sub_index, "final index should be LSH");
    FAISS_THROW_IF_NOT(sub_index->nbits == nbit);
    FAISS_THROW_IF_NOT(!sub_index->rotate_data);
    FAISS_THROW_IF_NOT(!sub_index->train_thresholds);
    replace_vt(encoder->chain[0], own);
}

} // namespace faiss
