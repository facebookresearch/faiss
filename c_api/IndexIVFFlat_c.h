/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_IVF_FLAT_C_H
#define FAISS_INDEX_IVF_FLAT_C_H

#include "Clustering_c.h"
#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
FAISS_DECLARE_CLASS_INHERITED(IndexIVFFlat, Index)
FAISS_DECLARE_DESTRUCTOR(IndexIVFFlat)
FAISS_DECLARE_INDEX_DOWNCAST(IndexIVFFlat)

/// number of possible key values
FAISS_DECLARE_GETTER(IndexIVFFlat, size_t, nlist)
/// number of probes at query time
FAISS_DECLARE_GETTER_SETTER(IndexIVFFlat, size_t, nprobe)
/// quantizer that maps vectors to inverted lists
FAISS_DECLARE_GETTER(IndexIVFFlat, FaissIndex*, quantizer)
/**
 * = 0: use the quantizer as index in a kmeans training
 * = 1: just pass on the training set to the train() of the quantizer
 * = 2: kmeans training on a flat index + add the centroids to the quantizer
 */
FAISS_DECLARE_GETTER(IndexIVFFlat, char, quantizer_trains_alone)

/// whether object owns the quantizer
FAISS_DECLARE_GETTER_SETTER(IndexIVFFlat, int, own_fields)

int faiss_IndexIVFFlat_new(FaissIndexIVFFlat** p_index);

int faiss_IndexIVFFlat_new_with(
        FaissIndexIVFFlat** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist);

int faiss_IndexIVFFlat_new_with_metric(
        FaissIndexIVFFlat** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        FaissMetricType metric);

int faiss_IndexIVFFlat_add_core(
        FaissIndexIVFFlat* index,
        idx_t n,
        const float* x,
        const idx_t* xids,
        const int64_t* precomputed_idx);

/** Update a subset of vectors.
 *
 * The index must have a direct_map
 *
 * @param nv     nb of vectors to update
 * @param idx    vector indices to update, size nv
 * @param v      vectors of new values, size nv*d
 */
int faiss_IndexIVFFlat_update_vectors(
        FaissIndexIVFFlat* index,
        int nv,
        idx_t* idx,
        const float* v);

#ifdef __cplusplus
}
#endif

#endif
