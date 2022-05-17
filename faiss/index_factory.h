/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>

namespace faiss {

/** Build and index with the sequence of processing steps described in
 *  the string. */
Index* index_factory(
        int d,
        const char* description,
        MetricType metric = METRIC_L2);

/// set to > 0 to get more logs from index_factory
FAISS_API extern int index_factory_verbose;

IndexBinary* index_binary_factory(int d, const char* description);

} // namespace faiss
