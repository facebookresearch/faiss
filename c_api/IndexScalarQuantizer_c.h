/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
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

#ifdef __cplusplus
}
#endif

#endif
