/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "IndexScalarQuantizer_c.h"
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexScalarQuantizer;

DEFINE_DESTRUCTOR(IndexScalarQuantizer)
DEFINE_INDEX_DOWNCAST(IndexScalarQuantizer)

int faiss_IndexScalarQuantizer_new(FaissIndexScalarQuantizer** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexScalarQuantizer*>(
                new IndexScalarQuantizer());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexScalarQuantizer_new_with(
        FaissIndexScalarQuantizer** p_index,
        idx_t d,
        FaissQuantizerType qt,
        FaissMetricType metric) {
    try {
        IndexScalarQuantizer* index = new IndexScalarQuantizer(
                d,
                static_cast<faiss::ScalarQuantizer::QuantizerType>(qt),
                static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexScalarQuantizer*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}
