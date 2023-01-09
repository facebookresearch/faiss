/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>

#include <faiss/IndexReplicas.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

template <typename IndexT>
IndexReplicasTemplate<IndexT>::IndexReplicasTemplate(bool threaded)
        : ThreadedIndex<IndexT>(threaded) {}

template <typename IndexT>
IndexReplicasTemplate<IndexT>::IndexReplicasTemplate(idx_t d, bool threaded)
        : ThreadedIndex<IndexT>(d, threaded) {}

template <typename IndexT>
IndexReplicasTemplate<IndexT>::IndexReplicasTemplate(int d, bool threaded)
        : ThreadedIndex<IndexT>(d, threaded) {}

template <typename IndexT>
void IndexReplicasTemplate<IndexT>::onAfterAddIndex(IndexT* index) {
    // Make sure that the parameters are the same for all prior indices, unless
    // we're the first index to be added
    if (this->count() > 0 && this->at(0) != index) {
        auto existing = this->at(0);

        FAISS_THROW_IF_NOT_FMT(
                index->ntotal == existing->ntotal,
                "IndexReplicas: newly added index does "
                "not have same number of vectors as prior index; "
                "prior index has %" PRId64 " vectors, new index has %" PRId64,
                existing->ntotal,
                index->ntotal);

        FAISS_THROW_IF_NOT_MSG(
                index->is_trained == existing->is_trained,
                "IndexReplicas: newly added index does "
                "not have same train status as prior index");

        FAISS_THROW_IF_NOT_MSG(
                index->d == existing->d,
                "IndexReplicas: newly added index does "
                "not have same dimension as prior index");
    } else {
        syncWithSubIndexes();
    }
}

template <typename IndexT>
void IndexReplicasTemplate<IndexT>::onAfterRemoveIndex(IndexT* index) {
    syncWithSubIndexes();
}

template <typename IndexT>
void IndexReplicasTemplate<IndexT>::train(idx_t n, const component_t* x) {
    auto fn = [n, x](int i, IndexT* index) {
        if (index->verbose) {
            printf("begin train replica %d on %" PRId64 " points\n", i, n);
        }

        index->train(n, x);

        if (index->verbose) {
            printf("end train replica %d\n", i);
        }
    };

    this->runOnIndex(fn);
    syncWithSubIndexes();
}

template <typename IndexT>
void IndexReplicasTemplate<IndexT>::add(idx_t n, const component_t* x) {
    auto fn = [n, x](int i, IndexT* index) {
        if (index->verbose) {
            printf("begin add replica %d on %" PRId64 " points\n", i, n);
        }

        index->add(n, x);

        if (index->verbose) {
            printf("end add replica %d\n", i);
        }
    };

    this->runOnIndex(fn);
    syncWithSubIndexes();
}

template <typename IndexT>
void IndexReplicasTemplate<IndexT>::reconstruct(idx_t n, component_t* x) const {
    FAISS_THROW_IF_NOT_MSG(this->count() > 0, "no replicas in index");

    // Just pass to the first replica
    this->at(0)->reconstruct(n, x);
}

template <typename IndexT>
void IndexReplicasTemplate<IndexT>::search(
        idx_t n,
        const component_t* x,
        idx_t k,
        distance_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(this->count() > 0, "no replicas in index");

    if (n == 0) {
        return;
    }

    auto dim = this->d;
    size_t componentsPerVec = sizeof(component_t) == 1 ? (dim + 7) / 8 : dim;

    // Partition the query by the number of indices we have
    faiss::idx_t queriesPerIndex =
            (faiss::idx_t)(n + this->count() - 1) / (faiss::idx_t)this->count();
    FAISS_ASSERT(n / queriesPerIndex <= this->count());

    auto fn = [queriesPerIndex, componentsPerVec, n, x, k, distances, labels](
                      int i, const IndexT* index) {
        faiss::idx_t base = (faiss::idx_t)i * queriesPerIndex;

        if (base < n) {
            auto numForIndex = std::min(queriesPerIndex, n - base);

            if (index->verbose) {
                printf("begin search replica %d on %" PRId64 " points\n",
                       i,
                       numForIndex);
            }

            index->search(
                    numForIndex,
                    x + base * componentsPerVec,
                    k,
                    distances + base * k,
                    labels + base * k);

            if (index->verbose) {
                printf("end search replica %d\n", i);
            }
        }
    };

    this->runOnIndex(fn);
}

// FIXME: assumes that nothing is currently running on the sub-indexes, which is
// true with the normal API, but should use the runOnIndex API instead
template <typename IndexT>
void IndexReplicasTemplate<IndexT>::syncWithSubIndexes() {
    if (!this->count()) {
        this->is_trained = false;
        this->ntotal = 0;

        return;
    }

    auto firstIndex = this->at(0);
    this->metric_type = firstIndex->metric_type;
    this->is_trained = firstIndex->is_trained;
    this->ntotal = firstIndex->ntotal;

    for (int i = 1; i < this->count(); ++i) {
        auto index = this->at(i);
        FAISS_THROW_IF_NOT(this->metric_type == index->metric_type);
        FAISS_THROW_IF_NOT(this->d == index->d);
        FAISS_THROW_IF_NOT(this->is_trained == index->is_trained);
        FAISS_THROW_IF_NOT(this->ntotal == index->ntotal);
    }
}

// No metric_type for IndexBinary
template <>
void IndexReplicasTemplate<IndexBinary>::syncWithSubIndexes() {
    if (!this->count()) {
        this->is_trained = false;
        this->ntotal = 0;

        return;
    }

    auto firstIndex = this->at(0);
    this->is_trained = firstIndex->is_trained;
    this->ntotal = firstIndex->ntotal;

    for (int i = 1; i < this->count(); ++i) {
        auto index = this->at(i);
        FAISS_THROW_IF_NOT(this->d == index->d);
        FAISS_THROW_IF_NOT(this->is_trained == index->is_trained);
        FAISS_THROW_IF_NOT(this->ntotal == index->ntotal);
    }
}

// explicit instantiations
template struct IndexReplicasTemplate<Index>;
template struct IndexReplicasTemplate<IndexBinary>;

} // namespace faiss
