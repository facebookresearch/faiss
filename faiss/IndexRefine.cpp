/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRefine.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/***************************************************
 * IndexRefine
 ***************************************************/

IndexRefine::IndexRefine(Index* base_index, Index* refine_index)
        : Index(base_index->d, base_index->metric_type),
          base_index(base_index),
          refine_index(refine_index) {
    own_fields = own_refine_index = false;
    if (refine_index != nullptr) {
        FAISS_THROW_IF_NOT(base_index->d == refine_index->d);
        FAISS_THROW_IF_NOT(
                base_index->metric_type == refine_index->metric_type);
        is_trained = base_index->is_trained && refine_index->is_trained;
        FAISS_THROW_IF_NOT(base_index->ntotal == refine_index->ntotal);
    } // other case is useful only to construct an IndexRefineFlat
    ntotal = base_index->ntotal;
}

IndexRefine::IndexRefine()
        : base_index(nullptr),
          refine_index(nullptr),
          own_fields(false),
          own_refine_index(false) {}

void IndexRefine::train(idx_t n, const float* x) {
    base_index->train(n, x);
    refine_index->train(n, x);
    is_trained = true;
}

void IndexRefine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    base_index->add(n, x);
    refine_index->add(n, x);
    ntotal = refine_index->ntotal;
}

void IndexRefine::reset() {
    base_index->reset();
    refine_index->reset();
    ntotal = 0;
}

namespace {

typedef faiss::Index::idx_t idx_t;

template <class C>
static void reorder_2_heaps(
        idx_t n,
        idx_t k,
        idx_t* labels,
        float* distances,
        idx_t k_base,
        const idx_t* base_labels,
        const float* base_distances) {
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        idx_t* idxo = labels + i * k;
        float* diso = distances + i * k;
        const idx_t* idxi = base_labels + i * k_base;
        const float* disi = base_distances + i * k_base;

        heap_heapify<C>(k, diso, idxo, disi, idxi, k);
        if (k_base != k) { // add remaining elements
            heap_addn<C>(k, diso, idxo, disi + k, idxi + k, k_base - k);
        }
        heap_reorder<C>(k, diso, idxo);
    }
}

} // anonymous namespace

void IndexRefine::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT(is_trained);
    idx_t k_base = idx_t(k * k_factor);
    idx_t* base_labels = labels;
    float* base_distances = distances;
    ScopeDeleter<idx_t> del1;
    ScopeDeleter<float> del2;

    if (k != k_base) {
        base_labels = new idx_t[n * k_base];
        del1.set(base_labels);
        base_distances = new float[n * k_base];
        del2.set(base_distances);
    }

    base_index->search(n, x, k_base, base_distances, base_labels);

    for (int i = 0; i < n * k_base; i++)
        assert(base_labels[i] >= -1 && base_labels[i] < ntotal);

        // parallelize over queries
#pragma omp parallel if (n > 1)
    {
        std::unique_ptr<DistanceComputer> dc(
                refine_index->get_distance_computer());
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            dc->set_query(x + i * d);
            idx_t ij = i * k_base;
            for (idx_t j = 0; j < k_base; j++) {
                idx_t idx = base_labels[ij];
                if (idx < 0)
                    break;
                base_distances[ij] = (*dc)(idx);
                ij++;
            }
        }
    }

    // sort and store result
    if (metric_type == METRIC_L2) {
        typedef CMax<float, idx_t> C;
        reorder_2_heaps<C>(
                n, k, labels, distances, k_base, base_labels, base_distances);

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        typedef CMin<float, idx_t> C;
        reorder_2_heaps<C>(
                n, k, labels, distances, k_base, base_labels, base_distances);
    } else {
        FAISS_THROW_MSG("Metric type not supported");
    }
}

void IndexRefine::reconstruct(idx_t key, float* recons) const {
    refine_index->reconstruct(key, recons);
}

IndexRefine::~IndexRefine() {
    if (own_fields)
        delete base_index;
    if (own_refine_index)
        delete refine_index;
}

/***************************************************
 * IndexRefineFlat
 ***************************************************/

IndexRefineFlat::IndexRefineFlat(Index* base_index)
        : IndexRefine(
                  base_index,
                  new IndexFlat(base_index->d, base_index->metric_type)) {
    is_trained = base_index->is_trained;
    own_refine_index = true;
    FAISS_THROW_IF_NOT_MSG(
            base_index->ntotal == 0,
            "base_index should be empty in the beginning");
}

IndexRefineFlat::IndexRefineFlat(Index* base_index, const float* xb)
        : IndexRefine(base_index, nullptr) {
    is_trained = base_index->is_trained;
    refine_index = new IndexFlat(base_index->d, base_index->metric_type);
    own_refine_index = true;
    refine_index->add(base_index->ntotal, xb);
}

IndexRefineFlat::IndexRefineFlat() : IndexRefine() {
    own_refine_index = true;
}

void IndexRefineFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT(is_trained);
    idx_t k_base = idx_t(k * k_factor);
    idx_t* base_labels = labels;
    float* base_distances = distances;
    ScopeDeleter<idx_t> del1;
    ScopeDeleter<float> del2;

    if (k != k_base) {
        base_labels = new idx_t[n * k_base];
        del1.set(base_labels);
        base_distances = new float[n * k_base];
        del2.set(base_distances);
    }

    base_index->search(n, x, k_base, base_distances, base_labels);

    for (int i = 0; i < n * k_base; i++)
        assert(base_labels[i] >= -1 && base_labels[i] < ntotal);

    // compute refined distances
    auto rf = dynamic_cast<const IndexFlat*>(refine_index);
    FAISS_THROW_IF_NOT(rf);

    rf->compute_distance_subset(n, x, k_base, base_distances, base_labels);

    // sort and store result
    if (metric_type == METRIC_L2) {
        typedef CMax<float, idx_t> C;
        reorder_2_heaps<C>(
                n, k, labels, distances, k_base, base_labels, base_distances);

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        typedef CMin<float, idx_t> C;
        reorder_2_heaps<C>(
                n, k, labels, distances, k_base, base_labels, base_distances);
    } else {
        FAISS_THROW_MSG("Metric type not supported");
    }
}

} // namespace faiss
