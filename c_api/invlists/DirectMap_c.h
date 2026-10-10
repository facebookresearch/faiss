/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_DIRECT_MAP_C_H
#define FAISS_DIRECT_MAP_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FaissDirectMapType {
    DIRECT_MAP_NO_MAP = 0,    // default
    DIRECT_MAP_ARRAY = 1,     // sequential ids (only for add, no add_with_ids)
    DIRECT_MAP_HASHTABLE = 2, // arbitrary ids
} FaissDirectMapType;

#ifdef __cplusplus
}
#endif

#endif
