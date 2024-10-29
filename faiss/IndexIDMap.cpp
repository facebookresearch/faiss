/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIDMap.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/WorkerThread.h>

namespace faiss {

namespace {

// IndexBinary needs to update the code_size when d is set...

void sync_d(Index* index) {}

void sync_d(IndexBinary* index) {
    FAISS_THROW_IF_NOT(index->d % 8 == 0);
    index->code_size = index->d / 8;
}

} // anonymous namespace

/*****************************************************
 * IndexIDMap implementation
 *******************************************************/

template <typename IndexT>
IndexIDMapTemplate<IndexT>::IndexIDMapTemplate(IndexT* index) : index(index) {
    FAISS_THROW_IF_NOT_MSG(index->ntotal == 0, "index must be empty on input");
    this->is_trained = index->is_trained;
    this->metric_type = index->metric_type;
    this->verbose = index->verbose;
    this->d = index->d;
    sync_d(this);
}

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::add(
        idx_t,
        const typename IndexT::component_t*) {
    FAISS_THROW_MSG(
            "add does not make sense with IndexIDMap, "
            "use add_with_ids");
}

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::train(
        idx_t n,
        const typename IndexT::component_t* x) {
    index->train(n, x);
    this->is_trained = index->is_trained;
}

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::reset() {
    index->reset();
    id_map.clear();
    this->ntotal = 0;
}

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::add_with_ids(
        idx_t n,
        const typename IndexT::component_t* x,
        const idx_t* xids) {
    index->add(n, x);
    for (idx_t i = 0; i < n; i++)
        id_map.push_back(xids[i]);
    this->ntotal = index->ntotal;
}

template <typename IndexT>
size_t IndexIDMapTemplate<IndexT>::sa_code_size() const {
    return index->sa_code_size();
}

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::add_sa_codes(
        idx_t n,
        const uint8_t* codes,
        const idx_t* xids) {
    index->add_sa_codes(n, codes, xids);
    for (idx_t i = 0; i < n; i++) {
        id_map.push_back(xids[i]);
    }
    this->ntotal = index->ntotal;
}

namespace {

/// RAII object to reset the IDSelector in the params object
struct ScopedSelChange {
    SearchParameters* params = nullptr;
    IDSelector* old_sel = nullptr;

    void set(SearchParameters* params_2, IDSelector* new_sel) {
        this->params = params_2;
        old_sel = params_2->sel;
        params_2->sel = new_sel;
    }
    ~ScopedSelChange() {
        if (params) {
            params->sel = old_sel;
        }
    }
};

} // namespace

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::search(
        idx_t n,
        const typename IndexT::component_t* x,
        idx_t k,
        typename IndexT::distance_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IDSelectorTranslated this_idtrans(this->id_map, nullptr);
    ScopedSelChange sel_change;

    if (params && params->sel) {
        auto idtrans = dynamic_cast<const IDSelectorTranslated*>(params->sel);

        if (!idtrans) {
            /*
            FAISS_THROW_IF_NOT_MSG(
                    idtrans,
                    "IndexIDMap requires an IDSelectorTranslated on input");
            */
            // then make an idtrans and force it into the SearchParameters
            // (hence the const_cast)
            auto params_non_const = const_cast<SearchParameters*>(params);
            this_idtrans.sel = params->sel;
            sel_change.set(params_non_const, &this_idtrans);
        }
    }
    index->search(n, x, k, distances, labels, params);
    idx_t* li = labels;
#pragma omp parallel for
    for (idx_t i = 0; i < n * k; i++) {
        li[i] = li[i] < 0 ? li[i] : id_map[li[i]];
    }
}

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::range_search(
        idx_t n,
        const typename IndexT::component_t* x,
        typename IndexT::distance_t radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    if (params) {
        SearchParameters internal_search_parameters;
        IDSelectorTranslated id_selector_translated(id_map, params->sel);
        internal_search_parameters.sel = &id_selector_translated;

        index->range_search(n, x, radius, result, &internal_search_parameters);
    } else {
        index->range_search(n, x, radius, result);
    }

#pragma omp parallel for
    for (idx_t i = 0; i < result->lims[result->nq]; i++) {
        result->labels[i] = result->labels[i] < 0 ? result->labels[i]
                                                  : id_map[result->labels[i]];
    }
}

template <typename IndexT>
size_t IndexIDMapTemplate<IndexT>::remove_ids(const IDSelector& sel) {
    // remove in sub-index first
    IDSelectorTranslated sel2(id_map, &sel);
    size_t nremove = index->remove_ids(sel2);

    int64_t j = 0;
    for (idx_t i = 0; i < this->ntotal; i++) {
        if (sel.is_member(id_map[i])) {
            // remove
        } else {
            id_map[j] = id_map[i];
            j++;
        }
    }
    FAISS_ASSERT(j == index->ntotal);
    this->ntotal = j;
    id_map.resize(this->ntotal);
    return nremove;
}

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::check_compatible_for_merge(
        const IndexT& otherIndex) const {
    auto other = dynamic_cast<const IndexIDMapTemplate<IndexT>*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    index->check_compatible_for_merge(*other->index);
}

template <typename IndexT>
void IndexIDMapTemplate<IndexT>::merge_from(IndexT& otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    auto other = static_cast<IndexIDMapTemplate<IndexT>*>(&otherIndex);
    index->merge_from(*other->index);
    for (size_t i = 0; i < other->id_map.size(); i++) {
        id_map.push_back(other->id_map[i] + add_id);
    }
    other->id_map.resize(0);
    this->ntotal = index->ntotal;
    other->ntotal = 0;
}

template <typename IndexT>
IndexIDMapTemplate<IndexT>::~IndexIDMapTemplate() {
    if (own_fields)
        delete index;
}

/*****************************************************
 * IndexIDMap2 implementation
 *******************************************************/

template <typename IndexT>
IndexIDMap2Template<IndexT>::IndexIDMap2Template(IndexT* index)
        : IndexIDMapTemplate<IndexT>(index) {}

template <typename IndexT>
void IndexIDMap2Template<IndexT>::add_with_ids(
        idx_t n,
        const typename IndexT::component_t* x,
        const idx_t* xids) {
    size_t prev_ntotal = this->ntotal;
    IndexIDMapTemplate<IndexT>::add_with_ids(n, x, xids);
    for (size_t i = prev_ntotal; i < this->ntotal; i++) {
        rev_map[this->id_map[i]] = i;
    }
}

template <typename IndexT>
void IndexIDMap2Template<IndexT>::check_consistency() const {
    FAISS_THROW_IF_NOT(rev_map.size() == this->id_map.size());
    FAISS_THROW_IF_NOT(this->id_map.size() == this->ntotal);
    for (size_t i = 0; i < this->ntotal; i++) {
        idx_t ii = rev_map.at(this->id_map[i]);
        FAISS_THROW_IF_NOT(ii == i);
    }
}

template <typename IndexT>
void IndexIDMap2Template<IndexT>::merge_from(IndexT& otherIndex, idx_t add_id) {
    size_t prev_ntotal = this->ntotal;
    IndexIDMapTemplate<IndexT>::merge_from(otherIndex, add_id);
    for (size_t i = prev_ntotal; i < this->ntotal; i++) {
        rev_map[this->id_map[i]] = i;
    }
    static_cast<IndexIDMap2Template<IndexT>&>(otherIndex).rev_map.clear();
}

template <typename IndexT>
void IndexIDMap2Template<IndexT>::construct_rev_map() {
    rev_map.clear();
    for (size_t i = 0; i < this->ntotal; i++) {
        rev_map[this->id_map[i]] = i;
    }
}

template <typename IndexT>
size_t IndexIDMap2Template<IndexT>::remove_ids(const IDSelector& sel) {
    // This is quite inefficient
    size_t nremove = IndexIDMapTemplate<IndexT>::remove_ids(sel);
    construct_rev_map();
    return nremove;
}

template <typename IndexT>
void IndexIDMap2Template<IndexT>::reconstruct(
        idx_t key,
        typename IndexT::component_t* recons) const {
    try {
        this->index->reconstruct(rev_map.at(key), recons);
    } catch (const std::out_of_range&) {
        FAISS_THROW_FMT("key %" PRId64 " not found", key);
    }
}

// explicit template instantiations

template struct IndexIDMapTemplate<Index>;
template struct IndexIDMapTemplate<IndexBinary>;
template struct IndexIDMap2Template<Index>;
template struct IndexIDMap2Template<IndexBinary>;

} // namespace faiss
