/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/platform_macros.h>

#include <faiss/utils/Heap.h>

#if defined(__AVX2__) || defined(__aarch64__)

namespace faiss {

// Returns true if the fused kernel is available and the data was processed.
// Returns false if the fused kernel is not available.
bool exhaustive_L2sqr_fused_cmax_simdlib(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms);

} // namespace faiss

#endif
