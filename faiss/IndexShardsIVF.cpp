/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexShardsIVF.h>

#include <cinttypes>
#include <cstdio>
#include <functional>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/WorkerThread.h>
#include <faiss/utils/utils.h>

namespace faiss {

// subroutines
namespace {

// add translation to all valid labels
void translate_labels(int64_t n, idx_t* labels, int64_t translation) {
    if (translation == 0) {
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        if (labels[i] < 0) {
            continue;
        }
        labels[i] += translation;
    }
}

} // anonymous namespace

/************************************************************
 * IndexShardsIVF
 ************************************************************/

IndexShardsIVF::IndexShardsIVF(
        Index* quantizer,
        size_t nlist,
        bool threaded,
        bool successive_ids)
        : IndexShardsTemplate<Index>(quantizer->d, threaded, successive_ids),
          Level1Quantizer(quantizer, nlist) {
    is_trained = quantizer->is_trained && quantizer->ntotal == nlist;
}

void IndexShardsIVF::addIndex(Index* index) {
    auto index_ivf = dynamic_cast<IndexIVFInterface*>(index);
    FAISS_THROW_IF_NOT_MSG(index_ivf, "can only add IndexIVFs");
    FAISS_THROW_IF_NOT(index_ivf->nlist == nlist);
    IndexShardsTemplate<Index>::addIndex(index);
}

void IndexShardsIVF::train(idx_t n, const component_t* x) {
    if (verbose) {
        printf("Training level-1 quantizer\n");
    }
    train_q1(n, x, verbose, metric_type);

    // set the sub-quantizer codebooks
    std::vector<float> centroids(nlist * d);
    quantizer->reconstruct_n(0, nlist, centroids.data());

    // probably not worth running in parallel
    for (size_t i = 0; i < indices_.size(); i++) {
        Index* index = indices_[i].first;
        auto index_ivf = dynamic_cast<IndexIVFInterface*>(index);
        Index* quantizer = index_ivf->quantizer;
        if (!quantizer->is_trained) {
            quantizer->train(nlist, centroids.data());
        }
        quantizer->add(nlist, centroids.data());
        // finish training
        index->train(n, x);
    }

    is_trained = true;
}

void IndexShardsIVF::add_with_ids(
        idx_t n,
        const component_t* x,
        const idx_t* xids) {
    // IndexIVF exposes add_core that we can use to factorize the
    bool all_index_ivf = true;
    for (size_t i = 0; i < indices_.size(); i++) {
        Index* index = indices_[i].first;
        all_index_ivf = all_index_ivf && dynamic_cast<IndexIVF*>(index);
    }
    if (!all_index_ivf) {
        IndexShardsTemplate<Index>::add_with_ids(n, x, xids);
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            !(successive_ids && xids),
            "It makes no sense to pass in ids and "
            "request them to be shifted");

    if (successive_ids) {
        FAISS_THROW_IF_NOT_MSG(
                !xids,
                "It makes no sense to pass in ids and "
                "request them to be shifted");
        FAISS_THROW_IF_NOT_MSG(
                this->ntotal == 0,
                "when adding to IndexShards with successive_ids, "
                "only add() in a single pass is supported");
    }

    // perform coarse quantization
    std::vector<idx_t> Iq(n);
    std::vector<float> Dq(n);
    quantizer->search(n, x, 1, Dq.data(), Iq.data());

    // possibly shift ids
    idx_t nshard = this->count();
    const idx_t* ids = xids;
    std::vector<idx_t> aids;
    if (!ids && !successive_ids) {
        aids.resize(n);

        for (idx_t i = 0; i < n; i++) {
            aids[i] = this->ntotal + i;
        }
        ids = aids.data();
    }
    idx_t d = this->d;

    auto fn = [n, ids, x, nshard, d, Iq](int no, Index* index) {
        idx_t i0 = (idx_t)no * n / nshard;
        idx_t i1 = ((idx_t)no + 1) * n / nshard;
        auto index_ivf = dynamic_cast<IndexIVF*>(index);

        if (index->verbose) {
            printf("begin add shard %d on %" PRId64 " points\n", no, n);
        }

        index_ivf->add_core(
                i1 - i0, x + i0 * d, ids ? ids + i0 : nullptr, Iq.data() + i0);

        if (index->verbose) {
            printf("end add shard %d on %" PRId64 " points\n", no, i1 - i0);
        }
    };

    this->runOnIndex(fn);
    syncWithSubIndexes();
}

void IndexShardsIVF::search(
        idx_t n,
        const component_t* x,
        idx_t k,
        distance_t* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(count() > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }

    auto index0 = dynamic_cast<const IndexIVFInterface*>(at(0));
    idx_t nprobe = params ? params->nprobe : index0->nprobe;

    // coarse quantization (TODO: support tiling with search_precomputed)
    std::vector<distance_t> Dq(n * nprobe);
    std::vector<idx_t> Iq(n * nprobe);

    quantizer->search(n, x, nprobe, Dq.data(), Iq.data());

    int64_t nshard = this->count();

    std::vector<distance_t> all_distances(nshard * k * n);
    std::vector<idx_t> all_labels(nshard * k * n);
    std::vector<int64_t> translations(nshard, 0);

    if (successive_ids) {
        translations[0] = 0;
        for (int s = 0; s + 1 < nshard; s++) {
            translations[s + 1] = translations[s] + this->at(s)->ntotal;
        }
    }

    auto fn = [&](int no, const Index* indexIn) {
        if (indexIn->verbose) {
            printf("begin query shard %d on %" PRId64 " points\n", no, n);
        }

        auto index = dynamic_cast<const IndexIVFInterface*>(indexIn);

        FAISS_THROW_IF_NOT_MSG(index->nprobe == nprobe, "inconsistent nprobe");

        index->search_preassigned(
                n,
                x,
                k,
                Iq.data(),
                Dq.data(),
                all_distances.data() + no * k * n,
                all_labels.data() + no * k * n,
                false);

        translate_labels(
                n * k, all_labels.data() + no * k * n, translations[no]);

        if (indexIn->verbose) {
            printf("end query shard %d\n", no);
        }
    };

    this->runOnIndex(fn);

    if (this->metric_type == METRIC_L2) {
        merge_knn_results<idx_t, CMin<distance_t, int>>(
                n,
                k,
                nshard,
                all_distances.data(),
                all_labels.data(),
                distances,
                labels);
    } else {
        merge_knn_results<idx_t, CMax<distance_t, int>>(
                n,
                k,
                nshard,
                all_distances.data(),
                all_labels.data(),
                distances,
                labels);
    }
}

} // namespace faiss
