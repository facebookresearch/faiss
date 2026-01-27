/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances_fused/distances_fused.h>

#include <faiss/impl/platform_macros.h> // NOLINT
#include <faiss/utils/simd_levels.h> // NOLINT(facebook-unused-include-check) used in #ifdef __AVX512F__

#include <faiss/utils/distances_fused/avx512.h> // NOLINT
#include <faiss/utils/distances_fused/simdlib_based.h>

namespace faiss {

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

#ifdef __AVX512F__
    // Runtime check: only use AVX512 if the CPU supports it
    if (SIMDConfig::level >= SIMDLevel::AVX512) {
        return exhaustive_L2sqr_fused_cmax_AVX512(
                x, y, d, nx, ny, res, y_norms);
    }
#endif
#if defined(__AVX2__)
    // avx2 kernel
    return exhaustive_L2sqr_fused_cmax_simdlib<SIMDLevel::AVX2>(
            x, y, d, nx, ny, res, y_norms);
#elif defined(__aarch64__) && defined(COMPILE_SIMD_ARM_NEON)
    // arm neon kernel
    return exhaustive_L2sqr_fused_cmax_simdlib<SIMDLevel::ARM_NEON>(
            x, y, d, nx, ny, res, y_norms);
#else
    // not supported, please use a general-purpose kernel
    return false;
#endif
}

} // namespace faiss
