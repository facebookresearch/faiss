/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

/// BLAS-accelerated exhaustive L2 search for the k=1 (top-1) case.
/// Specializations live in the per-SIMD translation units under simd_impl/.
template <SIMDLevel>
void exhaustive_L2sqr_blas_cmax(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms);

} // namespace faiss
