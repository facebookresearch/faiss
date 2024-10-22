/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFIndependentQuantizer.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>

namespace faiss {

IndexIVFIndependentQuantizer::IndexIVFIndependentQuantizer(
        Index* quantizer,
        IndexIVF* index_ivf,
        VectorTransform* vt)
        : Index(quantizer->d, index_ivf->metric_type),
          quantizer(quantizer),
          vt(vt),
          index_ivf(index_ivf) {
    if (vt) {
        FAISS_THROW_IF_NOT_MSG(
                vt->d_in == d && vt->d_out == index_ivf->d,
                "invalid vector dimensions");
    } else {
        FAISS_THROW_IF_NOT_MSG(index_ivf->d == d, "invalid vector dimensions");
    }

    if (quantizer->is_trained && quantizer->ntotal != 0) {
        FAISS_THROW_IF_NOT(quantizer->ntotal == index_ivf->nlist);
    }
    if (index_ivf->is_trained && vt) {
        FAISS_THROW_IF_NOT(vt->is_trained);
    }
    ntotal = index_ivf->ntotal;
    is_trained =
            (quantizer->is_trained && quantizer->ntotal == index_ivf->nlist &&
             (!vt || vt->is_trained) && index_ivf->is_trained);

    // disable precomputed tables because they use the distances that are
    // provided by the coarse quantizer (that are out of sync with the IVFPQ)
    if (auto index_ivfpq = dynamic_cast<IndexIVFPQ*>(index_ivf)) {
        index_ivfpq->use_precomputed_table = -1;
    }
}

IndexIVFIndependentQuantizer::~IndexIVFIndependentQuantizer() {
    if (own_fields) {
        delete quantizer;
        delete index_ivf;
        delete vt;
    }
}

namespace {

struct VTransformedVectors : TransformedVectors {
    VTransformedVectors(const VectorTransform* vt, idx_t n, const float* x)
            : TransformedVectors(x, vt ? vt->apply(n, x) : x) {}
};

struct SubsampledVectors : TransformedVectors {
    SubsampledVectors(int d, idx_t* n, idx_t max_n, const float* x)
            : TransformedVectors(
                      x,
                      fvecs_maybe_subsample(d, (size_t*)n, max_n, x, true)) {}
};

} // anonymous namespace

void IndexIVFIndependentQuantizer::add(idx_t n, const float* x) {
    std::vector<float> D(n);
    std::vector<idx_t> I(n);
    quantizer->search(n, x, 1, D.data(), I.data());

    VTransformedVectors tv(vt, n, x);

    index_ivf->add_core(n, tv.x, nullptr, I.data());
}

void IndexIVFIndependentQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(!params, "search parameters not supported");
    int nprobe = index_ivf->nprobe;
    std::vector<float> D(n * nprobe);
    std::vector<idx_t> I(n * nprobe);
    quantizer->search(n, x, nprobe, D.data(), I.data());

    VTransformedVectors tv(vt, n, x);

    index_ivf->search_preassigned(
            n, tv.x, k, I.data(), D.data(), distances, labels, false);
}

void IndexIVFIndependentQuantizer::reset() {
    index_ivf->reset();
    ntotal = 0;
}

void IndexIVFIndependentQuantizer::train(idx_t n, const float* x) {
    // quantizer training
    size_t nlist = index_ivf->nlist;
    Level1Quantizer l1(quantizer, nlist);
    l1.train_q1(n, x, verbose, metric_type);

    // train the VectorTransform
    if (vt && !vt->is_trained) {
        if (verbose) {
            printf("IndexIVFIndependentQuantizer: train the VectorTransform\n");
        }
        vt->train(n, x);
    }

    // get the centroids from the quantizer, transform them and
    // add them to the index_ivf's quantizer
    if (verbose) {
        printf("IndexIVFIndependentQuantizer: extract the main quantizer centroids\n");
    }
    std::vector<float> centroids(nlist * d);
    quantizer->reconstruct_n(0, nlist, centroids.data());
    VTransformedVectors tcent(vt, nlist, centroids.data());

    if (verbose) {
        printf("IndexIVFIndependentQuantizer: add centroids to the secondary quantizer\n");
    }
    if (!index_ivf->quantizer->is_trained) {
        index_ivf->quantizer->train(nlist, tcent.x);
    }
    index_ivf->quantizer->add(nlist, tcent.x);

    // train the payload

    // optional subsampling
    idx_t max_nt = index_ivf->train_encoder_num_vectors();
    if (max_nt <= 0) {
        max_nt = (size_t)1 << 35;
    }
    SubsampledVectors sv(index_ivf->d, &n, max_nt, x);

    // transform subsampled vectors
    VTransformedVectors tv(vt, n, sv.x);

    if (verbose) {
        printf("IndexIVFIndependentQuantizer: train encoder\n");
    }

    if (index_ivf->by_residual) {
        // assign with quantizer
        std::vector<idx_t> assign(n);
        quantizer->assign(n, sv.x, assign.data());

        // compute residual with IVF quantizer
        std::vector<float> residuals(n * index_ivf->d);
        index_ivf->quantizer->compute_residual_n(
                n, tv.x, residuals.data(), assign.data());

        index_ivf->train_encoder(n, residuals.data(), assign.data());
    } else {
        index_ivf->train_encoder(n, tv.x, nullptr);
    }
    index_ivf->is_trained = true;
    is_trained = true;
}

} // namespace faiss
