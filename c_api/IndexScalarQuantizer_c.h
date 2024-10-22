/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_SCALAR_QUANTIZER_C_H
#define FAISS_INDEX_SCALAR_QUANTIZER_C_H

#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FaissQuantizerType {
    QT_8bit,         ///< 8 bits per component
    QT_4bit,         ///< 4 bits per component
    QT_8bit_uniform, ///< same, shared range for all dimensions
    QT_4bit_uniform,
    QT_fp16,
    QT_8bit_direct, ///< fast indexing of uint8s
    QT_6bit,        ///< 6 bits per component
    QT_bf16,
    QT_8bit_direct_signed, ///< fast indexing of signed int8s ranging from [-128
                           ///< to 127]
} FaissQuantizerType;

// forward declaration
typedef enum FaissMetricType FaissMetricType;

/** Opaque type for IndexScalarQuantizer */
FAISS_DECLARE_CLASS_INHERITED(IndexScalarQuantizer, Index)

int faiss_IndexScalarQuantizer_new(FaissIndexScalarQuantizer** p_index);

int faiss_IndexScalarQuantizer_new_with(
        FaissIndexScalarQuantizer** p_index,
        idx_t d,
        FaissQuantizerType qt,
        FaissMetricType metric);

FAISS_DECLARE_INDEX_DOWNCAST(IndexScalarQuantizer)

FAISS_DECLARE_DESTRUCTOR(IndexScalarQuantizer)

/** Opaque type for IndexIVFScalarQuantizer */
FAISS_DECLARE_CLASS_INHERITED(IndexIVFScalarQuantizer, Index)

FAISS_DECLARE_INDEX_DOWNCAST(IndexIVFScalarQuantizer)

FAISS_DECLARE_DESTRUCTOR(IndexIVFScalarQuantizer)

int faiss_IndexIVFScalarQuantizer_new(FaissIndexIVFScalarQuantizer** p_index);

int faiss_IndexIVFScalarQuantizer_new_with(
        FaissIndexIVFScalarQuantizer** p_index,
        FaissIndex* quantizer,
        idx_t d,
        size_t nlist,
        FaissQuantizerType qt);

int faiss_IndexIVFScalarQuantizer_new_with_metric(
        FaissIndexIVFScalarQuantizer** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        FaissQuantizerType qt,
        FaissMetricType metric,
        int encode_residual);

/// number of possible key values
FAISS_DECLARE_GETTER(IndexIVFScalarQuantizer, size_t, nlist)
/// number of probes at query time
FAISS_DECLARE_GETTER_SETTER(IndexIVFScalarQuantizer, size_t, nprobe)
/// quantizer that maps vectors to inverted lists
FAISS_DECLARE_GETTER(IndexIVFScalarQuantizer, FaissIndex*, quantizer)

/// whether object owns the quantizer
FAISS_DECLARE_GETTER_SETTER(IndexIVFScalarQuantizer, int, own_fields)

int faiss_IndexIVFScalarQuantizer_add_core(
        FaissIndexIVFScalarQuantizer* index,
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx);

#ifdef __cplusplus
}
#endif

#endif
