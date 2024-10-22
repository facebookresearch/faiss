/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_FACTORY_C_H
#define FAISS_INDEX_FACTORY_C_H

#include "IndexBinary_c.h"
#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Build an index with the sequence of processing steps described in
 *  the string.
 */
int faiss_index_factory(
        FaissIndex** p_index,
        int d,
        const char* description,
        FaissMetricType metric);

/** Build a binary index with the sequence of processing steps described in
 *  the string.
 */
int faiss_index_binary_factory(
        FaissIndexBinary** p_index,
        int d,
        const char* description);

#ifdef __cplusplus
}
#endif

#endif
