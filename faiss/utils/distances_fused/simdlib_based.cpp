/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AVX2 compilation unit for the simdlib-based fused distance kernel.

#ifdef COMPILE_SIMD_AVX2

#include <faiss/impl/simdlib/simdlib_avx2.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/distances_fused/simdlib_kernel-inl.h>

namespace faiss {

template <>
bool exhaustive_L2sqr_fused_cmax<SIMDLevel::AVX2>(
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

#define DISPATCH(DIM, NX_POINTS_PER_LOOP, NY_POINTS_PER_LOOP) \
    case DIM: {                                               \
        exhaustive_L2sqr_fused_cmax<                          \
                DIM,                                          \
                NX_POINTS_PER_LOOP,                           \
                NY_POINTS_PER_LOOP,                           \
                SIMDLevel::AVX2>(x, y, nx, ny, res, y_norms); \
        return true;                                          \
    }

    // faiss/benchs/bench_quantizer.py was used for benchmarking
    // and tuning 2nd and 3rd parameters values.
    // Basically, the larger the values for 2nd and 3rd parameters are,
    // the faster the execution is, but the more SIMD registers are needed.
    // This can be compensated with L1 cache, this is why this
    // code might operate with more registers than available
    // because of concurrent ports operations for ALU and LOAD/STORE.

    // It was possible to tweak these parameters on x64 machine.
    switch (d) {
        DISPATCH(1, 6, 1)
        DISPATCH(2, 6, 1)
        DISPATCH(3, 6, 1)
        DISPATCH(4, 8, 1)
        DISPATCH(5, 8, 1)
        DISPATCH(6, 8, 1)
        DISPATCH(7, 8, 1)
        DISPATCH(8, 8, 1)
        DISPATCH(9, 8, 1)
        DISPATCH(10, 8, 1)
        DISPATCH(11, 8, 1)
        DISPATCH(12, 8, 1)
        DISPATCH(13, 6, 1)
        DISPATCH(14, 6, 1)
        DISPATCH(15, 6, 1)
        DISPATCH(16, 6, 1)
    }

    return false;
#undef DISPATCH
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX2
