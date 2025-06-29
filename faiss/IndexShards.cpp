/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexShards.h>

#include <cinttypes>
#include <cstdio>
#include <functional>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/WorkerThread.h>

namespace faiss {

// subroutines
namespace {

// IndexBinary needs to update the code_size when d is set...

void sync_d(Index* index) {}

void sync_d(IndexBinary* index) {
    FAISS_THROW_IF_NOT(index->d % 8 == 0);
    index->code_size = index->d / 8;
}

// add translation to all valid labels
void translate_labels(int64_t n, idx_t* labels, int64_t translation) {
    if (translation == 0)
        return;
    for (int64_t i = 0; i < n; i++) {
        if (labels[i] < 0)
            continue;
        labels[i] += translation;
    }
}

} // anonymous namespace

template <typename IndexT>
IndexShardsTemplate<IndexT>::IndexShardsTemplate(
        idx_t d,
        bool threaded,
        bool successive_ids)
        : ThreadedIndex<IndexT>(d, threaded), successive_ids(successive_ids) {
    sync_d(this);
}

template <typename IndexT>
IndexShardsTemplate<IndexT>::IndexShardsTemplate(
        int d,
        bool threaded,
        bool successive_ids)
        : ThreadedIndex<IndexT>(d, threaded), successive_ids(successive_ids) {
    sync_d(this);
}

template <typename IndexT>
IndexShardsTemplate<IndexT>::IndexShardsTemplate(
        bool threaded,
        bool successive_ids)
        : ThreadedIndex<IndexT>(threaded), successive_ids(successive_ids) {
    sync_d(this);
}

template <typename IndexT>
void IndexShardsTemplate<IndexT>::onAfterAddIndex(IndexT* index /* unused */) {
    syncWithSubIndexes();
}

template <typename IndexT>
void IndexShardsTemplate<IndexT>::onAfterRemoveIndex(
        IndexT* index /* unused */) {
    syncWithSubIndexes();
}

// FIXME: assumes that nothing is currently running on the sub-indexes, which is
// true with the normal API, but should use the runOnIndex API instead
template <typename IndexT>
void IndexShardsTemplate<IndexT>::syncWithSubIndexes() {
    if (!this->count()) {
        this->is_trained = false;
        this->ntotal = 0;

        return;
    }

    auto firstIndex = this->at(0);
    this->d = firstIndex->d;
    sync_d(this);
    this->metric_type = firstIndex->metric_type;
    this->is_trained = firstIndex->is_trained;
    this->ntotal = firstIndex->ntotal;

    for (int i = 1; i < this->count(); ++i) {
        auto index = this->at(i);
        FAISS_THROW_IF_NOT(this->metric_type == index->metric_type);
        FAISS_THROW_IF_NOT(this->d == index->d);
        FAISS_THROW_IF_NOT(this->is_trained == index->is_trained);

        this->ntotal += index->ntotal;
    }
}

template <typename IndexT>
void IndexShardsTemplate<IndexT>::train(idx_t n, const component_t* x) {
    auto fn = [n, x](int no, IndexT* index) {
        if (index->verbose) {
            printf("begin train shard %d on %" PRId64 " points\n", no, n);
        }

        index->train(n, x);

        if (index->verbose) {
            printf("end train shard %d\n", no);
        }
    };

    this->runOnIndex(fn);
    syncWithSubIndexes();
}

template <typename IndexT>
void IndexShardsTemplate<IndexT>::add(idx_t n, const component_t* x) {
    add_with_ids(n, x, nullptr);
}

template <typename IndexT>
void IndexShardsTemplate<IndexT>::add_with_ids(
        idx_t n,
        const component_t* x,
        const idx_t* xids) {
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

    size_t components_per_vec =
            sizeof(component_t) == 1 ? (this->d + 7) / 8 : this->d;

    auto fn = [n, ids, x, nshard, components_per_vec](int no, IndexT* index) {
        idx_t i0 = (idx_t)no * n / nshard;
        idx_t i1 = ((idx_t)no + 1) * n / nshard;
        auto x0 = x + i0 * components_per_vec;

        if (index->verbose) {
            printf("begin add shard %d on %" PRId64 " points\n", no, n);
        }

        if (ids) {
            index->add_with_ids(i1 - i0, x0, ids + i0);
        } else {
            index->add(i1 - i0, x0);
        }

        if (index->verbose) {
            printf("end add shard %d on %" PRId64 " points\n", no, i1 - i0);
        }
    };

    this->runOnIndex(fn);
    syncWithSubIndexes();
}

template <typename IndexT>
void IndexShardsTemplate<IndexT>::search(
        idx_t n,
        const component_t* x,
        idx_t k,
        distance_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    int64_t nshard = this->count();

    std::vector<distance_t> all_distances(nshard * k * n);
    std::vector<idx_t> all_labels(nshard * k * n);
    std::vector<int64_t> translations(nshard, 0);

    // Because we just called runOnIndex above, it is safe to access the
    // sub-index ntotal here
    if (successive_ids) {
        translations[0] = 0;

        for (int s = 0; s + 1 < nshard; s++) {
            translations[s + 1] = translations[s] + this->at(s)->ntotal;
        }
    }

    auto fn = [n, k, x, params, &all_distances, &all_labels, &translations](
                      int no, const IndexT* index) {
        if (index->verbose) {
            printf("begin query shard %d on %" PRId64 " points\n", no, n);
        }

        index->search(
                n,
                x,
                k,
                all_distances.data() + no * k * n,
                all_labels.data() + no * k * n,
                params);

        translate_labels(
                n * k, all_labels.data() + no * k * n, translations[no]);

        if (index->verbose) {
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

// explicit instanciations
template struct IndexShardsTemplate<Index>;
template struct IndexShardsTemplate<IndexBinary>;

} // namespace faiss
