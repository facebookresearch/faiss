/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexLSH_c.h"
#include <faiss/IndexLSH.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexLSH;

DEFINE_DESTRUCTOR(IndexLSH)
DEFINE_INDEX_DOWNCAST(IndexLSH)

DEFINE_GETTER(IndexLSH, int, nbits)
DEFINE_GETTER(IndexLSH, int, code_size)
DEFINE_GETTER_PERMISSIVE(IndexLSH, int, rotate_data)
DEFINE_GETTER_PERMISSIVE(IndexLSH, int, train_thresholds)

int faiss_IndexLSH_new(FaissIndexLSH** p_index, idx_t d, int nbits) {
    try {
        *p_index = reinterpret_cast<FaissIndexLSH*>(new IndexLSH(d, nbits));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexLSH_new_with_options(
        FaissIndexLSH** p_index,
        idx_t d,
        int nbits,
        int rotate_data,
        int train_thresholds) {
    try {
        *p_index = reinterpret_cast<FaissIndexLSH*>(new IndexLSH(
                d,
                nbits,
                static_cast<bool>(rotate_data),
                static_cast<bool>(train_thresholds)));
    }
    CATCH_AND_HANDLE
}
