/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances_fused/distances_fused.h>

#include <faiss/impl/simd_dispatch.h>

namespace faiss {

// Scalar fallback: no fused kernel available.
template <>
bool exhaustive_L2sqr_fused_cmax<SIMDLevel::NONE>(
        const float*,
        const float*,
        size_t,
        size_t,
        size_t,
        Top1BlockResultHandler<CMax<float, int64_t>>&,
        const float*) {
    return false;
}

#ifdef COMPILE_SIMD_RISCV_RVV
template <>
bool exhaustive_L2sqr_fused_cmax<SIMDLevel::RISCV_RVV>(
        const float*,
        const float*,
        size_t,
        size_t,
        size_t,
        Top1BlockResultHandler<CMax<float, int64_t>>&,
        const float*) {
    return false;
}
#endif // COMPILE_SIMD_RISCV_RVV

bool exhaustive_L2sqr_fused_cmax(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    if (nx == 0 || ny == 0) {
        // nothing to do
        return true;
    }

    return with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_A0>(
            [&]<SIMDLevel SL>() {
                return exhaustive_L2sqr_fused_cmax<SL>(
                        x, y, d, nx, ny, res, y_norms);
            });
}

} // namespace faiss
