/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>

#include <faiss/utils/distances.h>
#include <faiss/utils/exhaustive_search_ops.h>

namespace faiss {

template <>
void exhaustive_L2sqr_blas_simd<SIMDLevel::NONE>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    exhaustive_L2sqr_blas_default_impl(x, y, d, nx, ny, res);
}

template <>
void exhaustive_L2sqr_blas<Top1BlockResultHandler<CMax<float, int64_t>>>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    DISPATCH_SIMDLevel(
            exhaustive_L2sqr_blas_simd, x, y, d, nx, ny, res, y_norms);
}

} // namespace faiss
