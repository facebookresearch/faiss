/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "Index_c.h"
#include "Clustering_c.h"
#include "IndexIVF_c.h"
#include "IndexIVF.h"
#include "macros_impl.h"

using faiss::IndexIVF;
using faiss::IndexIVFFlat;
using faiss::IndexIVFStats;

DEFINE_DESTRUCTOR(IndexIVF)

/// number of possible key values
DEFINE_GETTER(IndexIVF, size_t, nlist)
/// number of probes at query time
DEFINE_GETTER(IndexIVF, size_t, nprobe)
/// quantizer that maps vectors to inverted lists
DEFINE_GETTER_PERMISSIVE(IndexIVF, FaissIndex*, quantizer)

/**
 * = 0: use the quantizer as index in a kmeans training
 * = 1: just pass on the training set to the train() of the quantizer
 * = 2: kmeans training on a flat index + add the centroids to the quantizer
 */
DEFINE_GETTER(IndexIVF, char, quantizer_trains_alone)

/// whether object owns the quantizer
DEFINE_GETTER(IndexIVF, int, own_fields)

using faiss::IndexIVF;

int faiss_IndexIVF_merge_from(
    FaissIndexIVF* index, FaissIndexIVF* other, idx_t add_id) {
    try {
        reinterpret_cast<IndexIVF*>(index)->merge_from(
            *reinterpret_cast<IndexIVF*>(other), add_id);
    } CATCH_AND_HANDLE
}

int faiss_IndexIVF_copy_subset_to(
    const FaissIndexIVF* index, FaissIndexIVF* other, int subset_type, long a1,
    long a2) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->copy_subset_to(
            *reinterpret_cast<IndexIVF*>(other), subset_type, a1, a2);
    } CATCH_AND_HANDLE
}

int faiss_IndexIVF_search_preassigned (const FaissIndexIVF* index,
    idx_t n, const float *x, idx_t k, const idx_t *assign,
    const float *centroid_dis, float *distances, idx_t *labels,
    int store_pairs) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->search_preassigned(
            n, x, k, assign, centroid_dis, distances, labels, store_pairs);
   } CATCH_AND_HANDLE
}

size_t faiss_IndexIVF_get_list_size(const FaissIndexIVF* index, size_t list_no) {
    return reinterpret_cast<const IndexIVF*>(index)->get_list_size(list_no);
}

int faiss_IndexIVF_make_direct_map(FaissIndexIVF* index,
    int new_maintain_direct_map) {
    try {
        reinterpret_cast<IndexIVF*>(index)->make_direct_map(
            static_cast<bool>(new_maintain_direct_map));
    } CATCH_AND_HANDLE
}

double faiss_IndexIVF_imbalance_factor (const FaissIndexIVF* index) {
    return reinterpret_cast<const IndexIVF*>(index)->imbalance_factor();
}

/// display some stats about the inverted lists
void faiss_IndexIVF_print_stats (const FaissIndexIVF* index) {
    reinterpret_cast<const IndexIVF*>(index)->print_stats();
}

void faiss_IndexIVFStats_reset(FaissIndexIVFStats* stats) {
    reinterpret_cast<IndexIVFStats*>(stats)->reset();    
}

DEFINE_DESTRUCTOR(IndexIVFFlat)

int faiss_IndexIVFFlat_new(FaissIndexIVFFlat** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexIVFFlat*>(new IndexIVFFlat());
    } CATCH_AND_HANDLE
}

int faiss_IndexIVFFlat_new_with(FaissIndexIVFFlat** p_index,
    FaissIndex* quantizer, size_t d, size_t nlist);

int faiss_IndexIVFFlat_new_with_metric(
    FaissIndexIVFFlat** p_index, FaissIndex* quantizer, size_t d, size_t nlist,
    FaissMetricType metric);

int faiss_IndexIVFFlat_add_core(FaissIndexIVFFlat* index, idx_t n, 
    const float * x, const long *xids, const long *precomputed_idx);

/** Update a subset of vectors.
 *
 * The index must have a direct_map
 *
 * @param nv     nb of vectors to update
 * @param idx    vector indices to update, size nv
 * @param v      vectors of new values, size nv*d
 */
int faiss_IndexIVFFlat_update_vectors(FaissIndexIVFFlat* index, int nv,
    idx_t *idx, const float *v);
