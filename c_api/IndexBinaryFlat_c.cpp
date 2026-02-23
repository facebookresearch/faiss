/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include "IndexBinaryFlat_c.h"
#include <faiss/IndexBinaryFlat.h>
#include "macros_impl.h"

extern "C" {

using faiss::IndexBinaryFlat;

DEFINE_DESTRUCTOR(IndexBinaryFlat)
DEFINE_INDEX_BINARY_DOWNCAST(IndexBinaryFlat)

/** Select between using a heap or counting to select the k smallest values
 * when scanning inverted lists.
 */
DEFINE_GETTER(IndexBinaryFlat, int, use_heap)
DEFINE_SETTER(IndexBinaryFlat, int, use_heap)

/// configure batch size for queries
DEFINE_GETTER(IndexBinaryFlat, int, query_batch_size)
DEFINE_SETTER(IndexBinaryFlat, int, query_batch_size)
}
