/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances_fused/exhaustive_l2sqr_fused_cmax_256bit.h>
#include <faiss/utils/exhaustive_search_ops.h>

namespace faiss {

template <>
bool exhaustive_L2sqr_fused_cmax_simdlib<SIMDLevel::ARM_NEON>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // Process only cases with certain dimensionalities.
    // An acceptable dimensionality value is limited by the number of
    // available registers.

    // Please feel free to alter 2nd and 3rd parameters if you have access
    // to ARM-based machine so that you are able to benchmark this code.
    // Or to enable other dimensions.
    switch (d) {
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 1, 4, 2, SIMDLevel::ARM_NEON)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 2, 2, 2, SIMDLevel::ARM_NEON)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 3, 2, 2, SIMDLevel::ARM_NEON)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 4, 2, 1, SIMDLevel::ARM_NEON)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 5, 1, 1, SIMDLevel::ARM_NEON)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 6, 1, 1, SIMDLevel::ARM_NEON)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 7, 1, 1, SIMDLevel::ARM_NEON)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 8, 1, 1, SIMDLevel::ARM_NEON)
    }

    return false;
}

template <>
void exhaustive_L2sqr_blas_simd<SIMDLevel::ARM_NEON>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    if (nx == 0 || ny == 0) {
        return;
    }

    if (exhaustive_L2sqr_fused_cmax_simdlib<SIMDLevel::ARM_NEON>(
                x, y, d, nx, ny, res, y_norms)) {
        return;
    }

    exhaustive_L2sqr_blas_simd<SIMDLevel::NONE>(x, y, d, nx, ny, res, y_norms);
}

} // namespace faiss
