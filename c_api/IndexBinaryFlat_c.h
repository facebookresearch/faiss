/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_BINARY_FLAT_C_H
#define FAISS_INDEX_BINARY_FLAT_C_H

#include "IndexBinary_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Index based on a flat file
 *
 */
FAISS_DECLARE_CLASS_INHERITED(IndexBinaryFlat, IndexBinary)
FAISS_DECLARE_DESTRUCTOR(IndexBinaryFlat)
FAISS_DECLARE_INDEX_BINARY_DOWNCAST(IndexBinaryFlat)

#ifdef __cplusplus
}
#endif

#endif
