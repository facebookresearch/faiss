
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSLVQ4x8.h>

namespace faiss {

IndexSVSLVQ4x8::IndexSVSLVQ4x8(idx_t d, MetricType metric)
        : IndexSVSLVQ4x4(d, metric) {}

} // namespace faiss
