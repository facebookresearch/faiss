/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include "IndexBinaryIVF_c.h"
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexIVF.h>
#include "macros_impl.h"

extern "C" {

using faiss::IndexBinaryIVF;

DEFINE_DESTRUCTOR(IndexBinaryIVF)
DEFINE_INDEX_BINARY_DOWNCAST(IndexBinaryIVF)

/// number of possible key values
DEFINE_GETTER(IndexBinaryIVF, size_t, nlist)

/// number of probes at query time
DEFINE_GETTER(IndexBinaryIVF, size_t, nprobe)
DEFINE_SETTER(IndexBinaryIVF, size_t, nprobe)

/// quantizer that maps vectors to inverted lists
DEFINE_GETTER_PERMISSIVE(IndexBinaryIVF, FaissIndexBinary*, quantizer)

/// whether object owns the quantizer
DEFINE_GETTER(IndexBinaryIVF, int, own_fields)
DEFINE_SETTER(IndexBinaryIVF, int, own_fields)

/// max nb of codes to visit to do a query
DEFINE_GETTER(IndexBinaryIVF, size_t, max_codes)
DEFINE_SETTER(IndexBinaryIVF, size_t, max_codes)

/** Select between using a heap or counting to select the k smallest values
 * when scanning inverted lists.
 */
DEFINE_GETTER(IndexBinaryIVF, int, use_heap)
DEFINE_SETTER(IndexBinaryIVF, int, use_heap)

/// collect computations per batch
DEFINE_GETTER(IndexBinaryIVF, int, per_invlist_search)
DEFINE_SETTER(IndexBinaryIVF, int, per_invlist_search)

int faiss_IndexBinaryIVF_merge_from(
        FaissIndexBinaryIVF* index,
        FaissIndexBinaryIVF* other,
        idx_t add_id) {
    try {
        reinterpret_cast<IndexBinaryIVF*>(index)->merge_from(
                *reinterpret_cast<IndexBinaryIVF*>(other), add_id);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinaryIVF_search_preassigned(
        const FaissIndexBinaryIVF* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        const idx_t* cidx,
        const int32_t* cdis,
        int32_t* dis,
        idx_t* idx,
        int store_pairs,
        const FaissSearchParametersIVF* params) {
    try {
        const faiss::SearchParametersIVF* sp =
                reinterpret_cast<const faiss::SearchParametersIVF*>(params);
        reinterpret_cast<const IndexBinaryIVF*>(index)->search_preassigned(
                n, x, k, cidx, cdis, dis, idx, store_pairs, sp);
    }
    CATCH_AND_HANDLE
}

size_t faiss_IndexBinaryIVF_get_list_size(
        const FaissIndexBinaryIVF* index,
        size_t list_no) {
    return reinterpret_cast<const IndexBinaryIVF*>(index)->get_list_size(
            list_no);
}

int faiss_IndexBinaryIVF_make_direct_map(
        FaissIndexBinaryIVF* index,
        int new_maintain_direct_map) {
    try {
        reinterpret_cast<IndexBinaryIVF*>(index)->make_direct_map(
                static_cast<bool>(new_maintain_direct_map));
    }
    CATCH_AND_HANDLE
}

double faiss_IndexBinaryIVF_imbalance_factor(const FaissIndexBinaryIVF* index) {
    return reinterpret_cast<const IndexBinaryIVF*>(index)
            ->invlists->imbalance_factor();
}

/// display some stats about the inverted lists
void faiss_IndexBinaryIVF_print_stats(const FaissIndexBinaryIVF* index) {
    reinterpret_cast<const IndexBinaryIVF*>(index)->invlists->print_stats();
}

/// get inverted lists ids
void faiss_IndexBinaryIVF_invlists_get_ids(
        const FaissIndexBinaryIVF* index,
        size_t list_no,
        idx_t* invlist) {
    const idx_t* list =
            reinterpret_cast<const IndexBinaryIVF*>(index)->invlists->get_ids(
                    list_no);
    size_t list_size =
            reinterpret_cast<const IndexBinaryIVF*>(index)->get_list_size(
                    list_no);
    memcpy(invlist, list, list_size * sizeof(idx_t));
}
}
