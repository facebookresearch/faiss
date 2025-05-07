/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_BINARY_IVF_C_H
#define FAISS_INDEX_BINARY_IVF_C_H

#include "IndexBinary_c.h"
#include "IndexIVF_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Index based on a inverted file (IVF)
 *
 * In the inverted file, the quantizer (an IndexBinary instance) provides a
 * quantization index for each vector to be added. The quantization
 * index maps to a list (aka inverted list or posting list), where the
 * id of the vector is stored.
 *
 * Otherwise the object is similar to the IndexIVF
 */
FAISS_DECLARE_CLASS_INHERITED(IndexBinaryIVF, IndexBinary)
FAISS_DECLARE_DESTRUCTOR(IndexBinaryIVF)
FAISS_DECLARE_INDEX_BINARY_DOWNCAST(IndexBinaryIVF)

/// number of possible key values
FAISS_DECLARE_GETTER(IndexBinaryIVF, size_t, nlist)
/// number of probes at query time
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, size_t, nprobe)
/// quantizer that maps vectors to inverted lists
FAISS_DECLARE_GETTER(IndexBinaryIVF, FaissIndexBinary*, quantizer)

/// whether object owns the quantizer
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, int, own_fields)

/// max nb of codes to visit to do a query
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, size_t, max_codes)

/** Select between using a heap or counting to select the k smallest values
 * when scanning inverted lists.
 */
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, int, use_heap)

/// collect computations per batch
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, int, per_invlist_search)

/** moves the entries from another dataset to self. On output,
 * other is empty. add_id is added to all moved ids (for
 * sequential ids, this would be this->ntotal */
int faiss_IndexBinaryIVF_merge_from(
        FaissIndexBinaryIVF* index,
        FaissIndexBinaryIVF* other,
        idx_t add_id);

/** Search a set of vectors, that are pre-quantized by the IVF
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
 * @param params used to override the object's search parameters
 */
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
        const FaissSearchParametersIVF* params);

size_t faiss_IndexBinaryIVF_get_list_size(
        const FaissIndexBinaryIVF* index,
        size_t list_no);

/** initialize a direct map
 *
 * @param new_maintain_direct_map    if true, create a direct map,
 *                                   else clear it
 */
int faiss_IndexBinaryIVF_make_direct_map(
        FaissIndexBinaryIVF* index,
        int new_maintain_direct_map);

/** Check the inverted lists' imbalance factor.
 *
 * 1= perfectly balanced, >1: imbalanced
 */
double faiss_IndexBinaryIVF_imbalance_factor(const FaissIndexBinaryIVF* index);

/// display some stats about the inverted lists of the index
void faiss_IndexBinaryIVF_print_stats(const FaissIndexBinaryIVF* index);

#ifdef __cplusplus
}
#endif

#endif
