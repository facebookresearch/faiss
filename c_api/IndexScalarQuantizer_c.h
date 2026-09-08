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
    QT_0bit, ///< 0 bits per component, centroid-only distance (IVF only; a
             ///< flat index with QT_0bit stores nothing)
    QT_1bit_tqmse, ///< TurboQuant MSE-optimized, 1 bit per component
    QT_2bit_tqmse, ///< TurboQuant MSE-optimized, 2 bits per component
    QT_3bit_tqmse, ///< TurboQuant MSE-optimized, 3 bits per component
    QT_4bit_tqmse, ///< TurboQuant MSE-optimized, 4 bits per component
    QT_8bit_tqmse, ///< TurboQuant MSE-optimized, 8 bits per component
    QT_2bit_tq,    ///< Full TurboQuant (1-bit MSE + 1-bit QJL + factors)
    QT_3bit_tq,    ///< Full TurboQuant (2-bit MSE + 1-bit QJL + factors)
    QT_4bit_tq,    ///< Full TurboQuant (3-bit MSE + 1-bit QJL + factors)
    QT_5bit_tq,    ///< Full TurboQuant (4-bit MSE + 1-bit QJL + factors)
    QT_1bit_eden,  ///< EDEN Lloyd-Max scalar code, 1 bit per component
    QT_2bit_eden,  ///< EDEN Lloyd-Max scalar code, 2 bits per component
    QT_3bit_eden,  ///< EDEN Lloyd-Max scalar code, 3 bits per component
    QT_4bit_eden,  ///< EDEN Lloyd-Max scalar code, 4 bits per component
    QT_5bit_eden,  ///< EDEN Lloyd-Max scalar code, 5 bits per component
    QT_6bit_eden,  ///< EDEN Lloyd-Max scalar code, 6 bits per component
    QT_7bit_eden,  ///< EDEN Lloyd-Max scalar code, 7 bits per component
    QT_8bit_eden,  ///< EDEN Lloyd-Max scalar code, 8 bits per component
    QT_count
} FaissQuantizerType;

typedef enum FaissRangeStat {
    RS_minmax,    ///< [min - rs*(max-min), max + rs*(max-min)]
    RS_meanstd,   ///< [mean - std * rs, mean + std * rs]
    RS_quantiles, ///< [Q(rs), Q(1-rs)]
    RS_optim,     ///< alternate optimization of reconstruction error
} FaissRangeStat;

// forward declaration
typedef enum FaissMetricType FaissMetricType;

/** Opaque type for the ScalarQuantizer codec owned by an index.
 * Valid only while the owning index is alive and must not be freed.
 */
FAISS_DECLARE_CLASS(ScalarQuantizer)

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

/** Access the scalar quantizer codec of an IndexScalarQuantizer.
 * Borrowed reference, owned by the index; do not free.
 */
FaissScalarQuantizer* faiss_IndexScalarQuantizer_sq(
        FaissIndexScalarQuantizer* index);

/** Access the scalar quantizer codec of an IndexIVFScalarQuantizer.
 * Borrowed reference, owned by the index; do not free.
 */
FaissScalarQuantizer* faiss_IndexIVFScalarQuantizer_sq(
        FaissIndexIVFScalarQuantizer* index);

/// quantizer type of this codec
FAISS_DECLARE_GETTER(ScalarQuantizer, FaissQuantizerType, qtype)
/// bits per scalar code
FAISS_DECLARE_GETTER(ScalarQuantizer, size_t, bits)
/// size of the input vectors
FAISS_DECLARE_GETTER(ScalarQuantizer, size_t, d)
/// bytes per encoded vector
FAISS_DECLARE_GETTER(ScalarQuantizer, size_t, code_size)
/// range estimation strategy (uniform encoder)
FAISS_DECLARE_GETTER_SETTER(ScalarQuantizer, FaissRangeStat, rangestat)
/// argument to the range estimation strategy (rs)
FAISS_DECLARE_GETTER_SETTER(ScalarQuantizer, float, rangestat_arg)

/// Number of trained values
size_t faiss_ScalarQuantizer_trained_size(const FaissScalarQuantizer* sq);

/** Copy the trained values into out, which must hold at least
 * faiss_ScalarQuantizer_trained_size() floats.
 */
void faiss_ScalarQuantizer_trained(const FaissScalarQuantizer* sq, float* out);

#ifdef __cplusplus
}
#endif

#endif
