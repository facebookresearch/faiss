/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_INDEX_IVF_C_H
#define FAISS_INDEX_IVF_C_H

#include "faiss_c.h"
#include "Index_c.h"
#include "Clustering_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Index based on a inverted file (IVF)
 *
 * In the inverted file, the quantizer (an Index instance) provides a
 * quantization index for each vector to be added. The quantization
 * index maps to a list (aka inverted list or posting list), where the
 * id of the vector is then stored.
 *
 * At search time, the vector to be searched is also quantized, and
 * only the list corresponding to the quantization index is
 * searched. This speeds up the search by making it
 * non-exhaustive. This can be relaxed using multi-probe search: a few
 * (nprobe) quantization indices are selected and several inverted
 * lists are visited.
 *
 * Sub-classes implement a post-filtering of the index that refines
 * the distance estimation from the query to databse vectors.
 */
FAISS_DECLARE_CLASS_INHERITED(IndexIVF, Index)
FAISS_DECLARE_DESTRUCTOR(IndexIVF)
FAISS_DECLARE_INDEX_DOWNCAST(IndexIVF)

/// number of possible key values
FAISS_DECLARE_GETTER(IndexIVF, size_t, nlist)
/// number of probes at query time
FAISS_DECLARE_GETTER(IndexIVF, size_t, nprobe)
/// quantizer that maps vectors to inverted lists
FAISS_DECLARE_GETTER(IndexIVF, FaissIndex*, quantizer)
/**
 * = 0: use the quantizer as index in a kmeans training
 * = 1: just pass on the training set to the train() of the quantizer
 * = 2: kmeans training on a flat index + add the centroids to the quantizer
 */
FAISS_DECLARE_GETTER(IndexIVF, char, quantizer_trains_alone)

/// whether object owns the quantizer
FAISS_DECLARE_GETTER(IndexIVF, int, own_fields)

/** moves the entries from another dataset to self. On output,
 * other is empty. add_id is added to all moved ids (for
 * sequential ids, this would be this->ntotal */
int faiss_IndexIVF_merge_from(
    FaissIndexIVF* index, FaissIndexIVF* other, idx_t add_id);

/** copy a subset of the entries index to the other index
 *
 * if subset_type == 0: copies ids in [a1, a2)
 * if subset_type == 1: copies ids if id % a1 == a2
 * if subset_type == 2: copies inverted lists such that a1
 *                      elements are left before and a2 elements are after
 */
int faiss_IndexIVF_copy_subset_to(
    const FaissIndexIVF* index, FaissIndexIVF* other, int subset_type, idx_t a1,
    idx_t a2);

/** search a set of vectors, that are pre-quantized by the IVF
 *  quantizer. Fill in the corresponding heaps with the query
 *  results. search() calls this.
 *
 * @param n      nb of vectors to query
 * @param x      query vectors, size nx * d
 * @param assign coarse quantization indices, size nx * nprobe
 * @param centroid_dis
 *               distances to coarse centroids, size nx * nprobe
 * @param distance
 *               output distances, size n * k
 * @param labels output labels, size n * k
 * @param store_pairs store inv list index + inv list offset
 *                     instead in upper/lower 32 bit of result,
 *                     instead of ids (used for reranking).
 */
int faiss_IndexIVF_search_preassigned (const FaissIndexIVF* index,
    idx_t n, const float *x, idx_t k, const idx_t *assign,
    const float *centroid_dis, float *distances, idx_t *labels,
    int store_pairs);

size_t faiss_IndexIVF_get_list_size(const FaissIndexIVF* index,
    size_t list_no);

/** intialize a direct map
 *
 * @param new_maintain_direct_map    if true, create a direct map,
 *                                   else clear it
 */
int faiss_IndexIVF_make_direct_map(FaissIndexIVF* index,
    int new_maintain_direct_map);

/** Check the inverted lists' imbalance factor.
 *
 * 1= perfectly balanced, >1: imbalanced
 */
double faiss_IndexIVF_imbalance_factor (const FaissIndexIVF* index);

/// display some stats about the inverted lists of the index
void faiss_IndexIVF_print_stats (const FaissIndexIVF* index);

/// Get the IDs in an inverted list. IDs are written to `invlist`, which must be large enough
//// to accommodate the full list.
///
/// @param list_no the list ID
/// @param invlist output pointer to a slice of memory, at least as long as the list's size
/// @see faiss_IndexIVF_get_list_size(size_t) 
void faiss_IndexIVF_invlists_get_ids (const FaissIndexIVF* index, size_t list_no, idx_t* invlist);

typedef struct FaissIndexIVFStats {
    size_t nq;       // nb of queries run
    size_t nlist;    // nb of inverted lists scanned
    size_t ndis;     // nb of distancs computed
} FaissIndexIVFStats;

void faiss_IndexIVFStats_reset(FaissIndexIVFStats* stats);

inline void faiss_IndexIVFStats_init(FaissIndexIVFStats* stats) {
    faiss_IndexIVFStats_reset(stats);
}

#ifdef __cplusplus
}
#endif


#endif
