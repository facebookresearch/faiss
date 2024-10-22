/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_BINARY_C_H
#define FAISS_INDEX_BINARY_C_H

#include <stddef.h>
#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration required here
FAISS_DECLARE_CLASS(RangeSearchResult)

// typedef struct FaissRangeSearchResult_H FaissRangeSearchResult;
typedef struct FaissIDSelector_H FaissIDSelector;

/// Opaque type for referencing to a binary index object
FAISS_DECLARE_CLASS(IndexBinary)
FAISS_DECLARE_DESTRUCTOR(IndexBinary)

/// Getter for d
FAISS_DECLARE_GETTER(IndexBinary, int, d)

/// Getter for is_trained
FAISS_DECLARE_GETTER(IndexBinary, int, is_trained)

/// Getter for ntotal
FAISS_DECLARE_GETTER(IndexBinary, idx_t, ntotal)

/// Getter for metric_type
FAISS_DECLARE_GETTER(IndexBinary, FaissMetricType, metric_type)

FAISS_DECLARE_GETTER_SETTER(IndexBinary, int, verbose)

/** Perform training on a representative set of vectors
 *
 * @param index  opaque pointer to index object
 * @param n      nb of training vectors
 * @param x      training vectors, size n * d
 */
int faiss_IndexBinary_train(FaissIndexBinary* index, idx_t n, const uint8_t* x);

/** Add n vectors of dimension d to the index.
 *
 * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
 * This function slices the input vectors in chunks smaller than
 * blocksize_add and calls add_core.
 * @param index  opaque pointer to index object
 * @param x      input matrix, size n * d
 */
int faiss_IndexBinary_add(FaissIndexBinary* index, idx_t n, const uint8_t* x);

/** Same as add, but stores xids instead of sequential ids.
 *
 * The default implementation fails with an assertion, as it is
 * not supported by all indexes.
 *
 * @param index  opaque pointer to index object
 * @param xids   if non-null, ids to store for the vectors (size n)
 */
int faiss_IndexBinary_add_with_ids(
        FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        const idx_t* xids);

/** query n vectors of dimension d to the index.
 *
 * return at most k vectors. If there are not enough results for a
 * query, the result array is padded with -1s.
 *
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d
 * @param labels      output labels of the NNs, size n*k
 * @param distances   output pairwise distances, size n*k
 */
int faiss_IndexBinary_search(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels);

/** query n vectors of dimension d to the index.
 *
 * return all vectors with distance < radius. Note that many
 * indexes do not implement the range_search (only the k-NN search
 * is mandatory).
 *
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d
 * @param radius      search radius
 * @param result      result table
 */
int faiss_IndexBinary_range_search(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        int radius,
        FaissRangeSearchResult* result);

/** return the indexes of the k vectors closest to the query x.
 *
 * This function is identical as search but only return labels of neighbors.
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d
 * @param labels      output labels of the NNs, size n*k
 */
int faiss_IndexBinary_assign(
        FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t* labels,
        idx_t k);

/** removes all elements from the database.
 * @param index       opaque pointer to index object
 */
int faiss_IndexBinary_reset(FaissIndexBinary* index);

/** removes IDs from the index. Not supported by all indexes
 * @param index       opaque pointer to index object
 * @param nremove     output for the number of IDs removed
 */
int faiss_IndexBinary_remove_ids(
        FaissIndexBinary* index,
        const FaissIDSelector* sel,
        size_t* n_removed);

/** Reconstruct a stored vector (or an approximation if lossy coding)
 *
 * this function may not be defined for some indexes
 * @param index       opaque pointer to index object
 * @param key         id of the vector to reconstruct
 * @param recons      reconstructed vector (size d)
 */
int faiss_IndexBinary_reconstruct(
        const FaissIndexBinary* index,
        idx_t key,
        uint8_t* recons);

/** Reconstruct vectors i0 to i0 + ni - 1
 *
 * this function may not be defined for some indexes
 * @param index       opaque pointer to index object
 * @param recons      reconstructed vector (size ni * d)
 */
int faiss_IndexBinary_reconstruct_n(
        const FaissIndexBinary* index,
        idx_t i0,
        idx_t ni,
        uint8_t* recons);

#ifdef __cplusplus
}
#endif

#endif
