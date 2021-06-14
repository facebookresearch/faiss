/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_INDEX_PRETRANSFORM_C_H
#define FAISS_INDEX_PRETRANSFORM_C_H

#include "Index_c.h"
#include "VectorTransform_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Index that applies a LinearTransform transform on vectors before
 *  handing them over to a sub-index */
FAISS_DECLARE_CLASS_INHERITED(IndexPreTransform, Index)
FAISS_DECLARE_DESTRUCTOR(IndexPreTransform)
FAISS_DECLARE_INDEX_DOWNCAST(IndexPreTransform)

FAISS_DECLARE_GETTER(IndexPreTransform, FaissIndex*, index)
FAISS_DECLARE_GETTER_SETTER(IndexPreTransform, int, own_fields)

int faiss_IndexPreTransform_new(FaissIndexPreTransform** p_index);

int faiss_IndexPreTransform_new_with(
        FaissIndexPreTransform** p_index,
        FaissIndex* index);

int faiss_IndexPreTransform_new_with_transform(
        FaissIndexPreTransform** p_index,
        FaissVectorTransform* ltrans,
        FaissIndex* index);

int faiss_IndexPreTransform_prepend_transform(
        FaissIndexPreTransform* index,
        FaissVectorTransform* ltrans);

#ifdef __cplusplus
}
#endif

#endif
