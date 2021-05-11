/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "IndexIVFFlat_c.h"
#include <faiss/IndexIVFFlat.h>
#include "Clustering_c.h"
#include "Index_c.h"
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexIVFFlat;
using faiss::MetricType;

DEFINE_DESTRUCTOR(IndexIVFFlat)
DEFINE_INDEX_DOWNCAST(IndexIVFFlat)

/// number of possible key values
DEFINE_GETTER(IndexIVFFlat, size_t, nlist)
/// number of probes at query time
DEFINE_GETTER(IndexIVFFlat, size_t, nprobe)
DEFINE_SETTER(IndexIVFFlat, size_t, nprobe)

/// quantizer that maps vectors to inverted lists
DEFINE_GETTER_PERMISSIVE(IndexIVFFlat, FaissIndex*, quantizer)

/**
 * = 0: use the quantizer as index in a kmeans training
 * = 1: just pass on the training set to the train() of the quantizer
 * = 2: kmeans training on a flat index + add the centroids to the quantizer
 */
DEFINE_GETTER(IndexIVFFlat, char, quantizer_trains_alone)

/// whether object owns the quantizer
DEFINE_GETTER(IndexIVFFlat, int, own_fields)
DEFINE_SETTER(IndexIVFFlat, int, own_fields)

int faiss_IndexIVFFlat_new(FaissIndexIVFFlat** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexIVFFlat*>(new IndexIVFFlat());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFFlat_new_with(
        FaissIndexIVFFlat** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist) {
    try {
        auto q = reinterpret_cast<Index*>(quantizer);
        *p_index = reinterpret_cast<FaissIndexIVFFlat*>(
                new IndexIVFFlat(q, d, nlist));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFFlat_new_with_metric(
        FaissIndexIVFFlat** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        FaissMetricType metric) {
    try {
        auto q = reinterpret_cast<Index*>(quantizer);
        auto m = static_cast<MetricType>(metric);
        *p_index = reinterpret_cast<FaissIndexIVFFlat*>(
                new IndexIVFFlat(q, d, nlist, m));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFFlat_add_core(
        FaissIndexIVFFlat* index,
        idx_t n,
        const float* x,
        const idx_t* xids,
        const int64_t* precomputed_idx) {
    try {
        reinterpret_cast<IndexIVFFlat*>(index)->add_core(
                n, x, xids, precomputed_idx);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFFlat_update_vectors(
        FaissIndexIVFFlat* index,
        int nv,
        idx_t* idx,
        const float* v) {
    try {
        reinterpret_cast<IndexIVFFlat*>(index)->update_vectors(nv, idx, v);
    }
    CATCH_AND_HANDLE
}
