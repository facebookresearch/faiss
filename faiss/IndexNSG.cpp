/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexNSG.h>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

namespace faiss {

using idx_t = Index::idx_t;
using storage_idx_t = NSG::storage_idx_t;
using namespace nsg;

/**************************************************************
 * IndexNSG implementation
 **************************************************************/

IndexNSG::IndexNSG(int d, int R, MetricType metric)
    : Index(d, metric), nsg(R), own_fields(false), storage(nullptr) {}

IndexNSG::IndexNSG(Index *storage, int R)
    : Index(storage->d, storage->metric_type), nsg(R), own_fields(false),
      storage(storage), is_built(false) {}

IndexNSG::~IndexNSG() {
  if (own_fields) {
    delete storage;
  }
}

void IndexNSG::train(idx_t n, const float *x) {
  FAISS_THROW_IF_NOT_MSG(
      storage,
      "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
  // nsg structure does not require training
  storage->train(n, x);
  is_trained = true;
}

void IndexNSG::search(idx_t n, const float *x, idx_t k, float *distances,
                      idx_t *labels) const

{
  FAISS_THROW_IF_NOT_MSG(
      storage,
      "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");

  idx_t check_period = InterruptCallback::get_period_hint(d * nsg.search_L);

  for (idx_t i0 = 0; i0 < n; i0 += check_period) {
    idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
    {
      VisitedTable vt(ntotal);

      DistanceComputer *dis = storage_distance_computer(storage);
      ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
      for (idx_t i = i0; i < i1; i++) {
        idx_t *idxi = labels + i * k;
        float *simi = distances + i * k;
        dis->set_query(x + i * d);

        maxheap_heapify(k, simi, idxi);
        nsg.search(*dis, k, idxi, simi, vt);
        maxheap_reorder(k, simi, idxi);

        vt.advance();
      }
    }
    InterruptCallback::check();
  }

  if (metric_type == METRIC_INNER_PRODUCT) {
    // we need to revert the negated distances
    for (size_t i = 0; i < k * n; i++) {
      distances[i] = -distances[i];
    }
  }
}

void IndexNSG::build(idx_t n, const float *x, idx_t *knn_graph, int GK) {
  FAISS_THROW_IF_NOT_MSG(
      storage,
      "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
  FAISS_THROW_IF_NOT_MSG(!is_built, "The IndexNSG is already built");
  storage->add(n, x);
  ntotal = storage->ntotal;

  nsg::Graph<idx_t> knng(knn_graph, n, GK);
  nsg.build(storage, n, knng, verbose);
  is_built = true;
}

void IndexNSG::add(idx_t n, const float *x) {
  // FAISS_THROW_IF_NOT_MSG(storage,
  //    "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
  // FAISS_THROW_IF_NOT(is_trained && is_built);
  // int n0 = ntotal;
  // storage->add(n, x);
  // ntotal = storage->ntotal;

  // nsg.add (n, x, verbose);
  FAISS_THROW_MSG("add() not implemented for IndexNSG");
}

void IndexNSG::reset() {
  nsg.reset();
  storage->reset();
  ntotal = 0;
}

void IndexNSG::reconstruct(idx_t key, float *recons) const {
  storage->reconstruct(key, recons);
}

/**************************************************************
 * IndexNSGFlat implementation
 **************************************************************/

IndexNSGFlat::IndexNSGFlat() { is_trained = true; }

IndexNSGFlat::IndexNSGFlat(int d, int R, MetricType metric)
    : IndexNSG(new IndexFlat(d, metric), R) {
  own_fields = true;
  is_trained = true;
}

} // namespace faiss
