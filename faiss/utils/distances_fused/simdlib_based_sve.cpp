/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ARM SVE compilation unit for the simdlib-based fused distance kernel.

#ifdef COMPILE_SIMD_ARM_SVE

#include <faiss/impl/simdlib/simdlib_neon.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/distances_fused/simdlib_kernel-inl.h>

namespace faiss {

template <>
bool exhaustive_L2sqr_fused_cmax<SIMDLevel::ARM_SVE>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
#define DISPATCH(DIM, NX_POINTS_PER_LOOP, NY_POINTS_PER_LOOP)    \
    case DIM: {                                                  \
        exhaustive_L2sqr_fused_cmax<                             \
                DIM,                                             \
                NX_POINTS_PER_LOOP,                              \
                NY_POINTS_PER_LOOP,                              \
                SIMDLevel::ARM_SVE>(x, y, nx, ny, res, y_norms); \
        return true;                                             \
    }

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

#endif // COMPILE_SIMD_ARM_SVE
