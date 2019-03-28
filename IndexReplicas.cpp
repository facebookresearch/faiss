/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "IndexReplicas.h"
#include "FaissAssert.h"

namespace faiss {

IndexReplicas::IndexReplicas()
    : own_fields(false) {
}

IndexReplicas::~IndexReplicas() {
  if (own_fields) {
    for (auto& index : indices_)
      delete index.first;
  }
}

void
IndexReplicas::addIndex(faiss::Index* index) {
  // Make sure that the parameters are the same for all prior indices
  if (!indices_.empty()) {
    auto& existing = indices_.front().first;

    FAISS_THROW_IF_NOT_FMT(index->d == existing->d,
                           "IndexReplicas::addIndex: dimension mismatch for "
                           "newly added index; prior index has dim %d, "
                           "new index has %d",
                           existing->d, index->d);

    FAISS_THROW_IF_NOT_FMT(index->ntotal == existing->ntotal,
                           "IndexReplicas::addIndex: newly added index does "
                           "not have same number of vectors as prior index; "
                           "prior index has %ld vectors, new index has %ld",
                           existing->ntotal, index->ntotal);

    FAISS_THROW_IF_NOT_MSG(index->metric_type == existing->metric_type,
                           "IndexReplicas::addIndex: newly added index is "
                           "of different metric type than old index");
  } else {
    // Set our parameters
    // FIXME: this is a little bit weird
    this->d = index->d;
    this->ntotal = index->ntotal;
    this->verbose = index->verbose;
    this->is_trained = index->is_trained;
    this->metric_type = index->metric_type;
  }

  indices_.emplace_back(
    std::make_pair(index,
                   std::unique_ptr<WorkerThread>(new WorkerThread)));
}

void
IndexReplicas::removeIndex(faiss::Index* index) {
  for (auto it = indices_.begin(); it != indices_.end(); ++it) {
    if (it->first == index) {
      // This is our index; stop the worker thread before removing it,
      // to ensure that it has finished before function exit
      it->second->stop();
      it->second->waitForThreadExit();

      indices_.erase(it);
      return;
    }
  }

  // could not find our index
  FAISS_THROW_MSG("IndexReplicas::removeIndex: index not found");
}

void
IndexReplicas::runOnIndex(std::function<void(faiss::Index*)> f) {
  FAISS_THROW_IF_NOT_MSG(!indices_.empty(), "no replicas in index");

  std::vector<std::future<bool>> v;

  for (auto& index : indices_) {
    auto indexPtr = index.first;
    v.emplace_back(index.second->add([indexPtr, f](){ f(indexPtr); }));
  }

  // Blocking wait for completion
  for (auto& func : v) {
    func.get();
  }
}

void
IndexReplicas::reset() {
  runOnIndex([](faiss::Index* index){ index->reset(); });
  ntotal = 0;
}

void
IndexReplicas::train(Index::idx_t n, const float* x) {
  runOnIndex([n, x](faiss::Index* index){ index->train(n, x); });
}

void
IndexReplicas::add(Index::idx_t n, const float* x) {
  runOnIndex([n, x](faiss::Index* index){ index->add(n, x); });
  ntotal += n;
}

void
IndexReplicas::reconstruct(Index::idx_t n, float* x) const {
  FAISS_THROW_IF_NOT_MSG(!indices_.empty(), "no replicas in index");
  indices_[0].first->reconstruct (n, x);
}

void
IndexReplicas::search(faiss::Index::idx_t n,
                      const float* x,
                      faiss::Index::idx_t k,
                      float* distances,
                      faiss::Index::idx_t* labels) const {
  FAISS_THROW_IF_NOT_MSG(!indices_.empty(), "no replicas in index");

  if (n == 0) {
    return;
  }

  auto dim = indices_.front().first->d;

  std::vector<std::future<bool>> v;

  // Partition the query by the number of indices we have
  auto queriesPerIndex =
    (faiss::Index::idx_t) (n + indices_.size() - 1) / indices_.size();
  FAISS_ASSERT(n / queriesPerIndex <= indices_.size());

  for (faiss::Index::idx_t i = 0; i < indices_.size(); ++i) {
    auto base = i * queriesPerIndex;
    if (base >= n) {
      break;
    }

    auto numForIndex = std::min(queriesPerIndex, n - base);
    auto queryStart = x + base * dim;
    auto distancesStart = distances + base * k;
    auto labelsStart = labels + base * k;

    auto indexPtr = indices_[i].first;
    auto fn =
      [indexPtr, numForIndex, queryStart, k, distancesStart, labelsStart]() {
        indexPtr->search(numForIndex, queryStart,
                         k, distancesStart, labelsStart);
      };

    v.emplace_back(indices_[i].second->add(std::move(fn)));
  }

  // Blocking wait for completion
  for (auto& f : v) {
    f.get();
  }
}

} // namespace
