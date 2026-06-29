/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ARM NEON compilation unit for the simdlib-based fused distance kernel.

#ifdef COMPILE_SIMD_ARM_NEON

#include <faiss/impl/simdlib/simdlib_neon.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/distances_fused/simdlib_kernel-inl.h>

namespace faiss {

template <>
bool exhaustive_L2sqr_fused_cmax<SIMDLevel::ARM_NEON>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
#define DISPATCH(DIM, NX_POINTS_PER_LOOP, NY_POINTS_PER_LOOP)     \
    case DIM: {                                                   \
        exhaustive_L2sqr_fused_cmax<                              \
                DIM,                                              \
                NX_POINTS_PER_LOOP,                               \
                NY_POINTS_PER_LOOP,                               \
                SIMDLevel::ARM_NEON>(x, y, nx, ny, res, y_norms); \
        return true;                                              \
    }

    // Please feel free to alter 2nd and 3rd parameters if you have access
    // to ARM-based machine so that you are able to benchmark this code.
    // Or to enable other dimensions.
    switch (d) {
        DISPATCH(1, 4, 2)
        DISPATCH(2, 2, 2)
        DISPATCH(3, 2, 2)
        DISPATCH(4, 2, 1)
        DISPATCH(5, 1, 1)
        DISPATCH(6, 1, 1)
        DISPATCH(7, 1, 1)
        DISPATCH(8, 1, 1)
    }

    return false;
#undef DISPATCH
}

} // namespace faiss

#endif // COMPILE_SIMD_ARM_NEON
