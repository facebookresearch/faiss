/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "index_factory_c.h"
#include <faiss/index_factory.h>
#include <cstring>
#include "macros_impl.h"

using faiss::Index;

/** Build and index with the sequence of processing steps described in
 *  the string.
 */
int faiss_index_factory(
        FaissIndex** p_index,
        int d,
        const char* description,
        FaissMetricType metric) {
    try {
        *p_index = reinterpret_cast<FaissIndex*>(faiss::index_factory(
                d, description, static_cast<faiss::MetricType>(metric)));
    }
    CATCH_AND_HANDLE
}
