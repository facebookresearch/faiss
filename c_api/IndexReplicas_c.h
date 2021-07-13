/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef INDEXREPLICAS_C_H
#define INDEXREPLICAS_C_H

#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Index that concatenates the results from several sub-indexes
 */
FAISS_DECLARE_CLASS_INHERITED(IndexReplicas, Index)
FAISS_DECLARE_DESTRUCTOR(IndexReplicas)

FAISS_DECLARE_GETTER_SETTER(IndexReplicas, int, own_fields)

int faiss_IndexReplicas_new(FaissIndexReplicas** p_index, idx_t d);

int faiss_IndexReplicas_new_with_options(
        FaissIndexReplicas** p_index,
        idx_t d,
        int threaded);

int faiss_IndexReplicas_add_replica(
        FaissIndexReplicas* index,
        FaissIndex* replica);

int faiss_IndexReplicas_remove_replica(
        FaissIndexReplicas* index,
        FaissIndex* replica);

FaissIndex* faiss_IndexReplicas_at(FaissIndexReplicas* index, int i);

#ifdef __cplusplus
}
#endif
#endif
