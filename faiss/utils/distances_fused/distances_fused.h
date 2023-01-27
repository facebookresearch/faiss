/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file contains a fused kernel that combines distance computation
// and the search for the CLOSEST point. Currently, this is done for small
// dimensionality vectors when it is beneficial to avoid storing temporary
// dot product information in RAM. This is particularly effective
// when training PQx10 or PQx12 with the default parameters.
//
// InterruptCallback::check() is not used, because it is assumed that the
// kernel takes a little time because of a tiny dimensionality.
//
// Later on, similar optimization can be implemented for large size vectors,
// but a different kernel is needed.
//

#pragma once

#include <faiss/impl/ResultHandler.h>

#include <faiss/utils/Heap.h>

namespace faiss {

// Returns true if the fused kernel is available and the data was processed.
// Returns false if the fused kernel is not available.
bool exhaustive_L2sqr_fused_cmax(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        SingleBestResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms);

} // namespace faiss
