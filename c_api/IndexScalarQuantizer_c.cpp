/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "IndexScalarQuantizer_c.h"
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexIVFScalarQuantizer;
using faiss::IndexScalarQuantizer;

DEFINE_DESTRUCTOR(IndexScalarQuantizer)
DEFINE_INDEX_DOWNCAST(IndexScalarQuantizer)

int faiss_IndexScalarQuantizer_new(FaissIndexScalarQuantizer** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexScalarQuantizer*>(
                new IndexScalarQuantizer());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexScalarQuantizer_new_with(
        FaissIndexScalarQuantizer** p_index,
        idx_t d,
        FaissQuantizerType qt,
        FaissMetricType metric) {
    try {
        IndexScalarQuantizer* index = new IndexScalarQuantizer(
                d,
                static_cast<faiss::ScalarQuantizer::QuantizerType>(qt),
                static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexScalarQuantizer*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(IndexIVFScalarQuantizer)
DEFINE_INDEX_DOWNCAST(IndexIVFScalarQuantizer)

/// quantizer that maps vectors to inverted lists
DEFINE_GETTER_PERMISSIVE(IndexIVFScalarQuantizer, FaissIndex*, quantizer)

/// number of possible key values
DEFINE_GETTER(IndexIVFScalarQuantizer, size_t, nlist)
/// number of probes at query time
DEFINE_GETTER(IndexIVFScalarQuantizer, size_t, nprobe)
DEFINE_SETTER(IndexIVFScalarQuantizer, size_t, nprobe)

/// whether object owns the quantizer
DEFINE_GETTER(IndexIVFScalarQuantizer, int, own_fields)
DEFINE_SETTER(IndexIVFScalarQuantizer, int, own_fields)

int faiss_IndexIVFScalarQuantizer_new_with(
        FaissIndexIVFScalarQuantizer** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        FaissQuantizerType qt) {
    try {
        auto q = reinterpret_cast<Index*>(quantizer);
        auto qt_ = static_cast<faiss::ScalarQuantizer::QuantizerType>(qt);
        IndexIVFScalarQuantizer* index =
                new IndexIVFScalarQuantizer(q, d, nlist, qt_);
        *p_index = reinterpret_cast<FaissIndexIVFScalarQuantizer*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFScalarQuantizer_new_with_metric(
        FaissIndexIVFScalarQuantizer** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        FaissQuantizerType qt,
        FaissMetricType metric,
        int encode_residual) {
    try {
        auto q = reinterpret_cast<Index*>(quantizer);
        auto mt = static_cast<faiss::MetricType>(metric);
        auto er = static_cast<bool>(encode_residual);
        auto qt_ = static_cast<faiss::ScalarQuantizer::QuantizerType>(qt);
        IndexIVFScalarQuantizer* index =
                new IndexIVFScalarQuantizer(q, d, nlist, qt_, mt, er);
        *p_index = reinterpret_cast<FaissIndexIVFScalarQuantizer*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFScalarQuantizer_add_core(
        FaissIndexIVFScalarQuantizer* index,
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx) {
    try {
        reinterpret_cast<IndexIVFScalarQuantizer*>(index)->add_core(
                n, x, xids, precomputed_idx);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFScalarQuantizer_train_residual(
        FaissIndexIVFScalarQuantizer* index,
        idx_t n,
        const float* x) {
    try {
        reinterpret_cast<IndexIVFScalarQuantizer*>(index)->train_residual(n, x);
    }
    CATCH_AND_HANDLE
}
