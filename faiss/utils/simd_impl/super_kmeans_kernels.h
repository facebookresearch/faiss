/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <faiss/utils/simd_levels.h>

namespace faiss {
namespace detail {

// Squared L2 over `n` dimensions; n in [1, pdx_block_size].
// Primary template is the scalar fallback; SIMDLevels without a dedicated
// specialization (ARM_NEON, NONE, ...) use it directly.
template <SIMDLevel Level>
inline float block_l2(const float* x, const float* y, int n) {
    float s = 0.0f;
    for (int m = 0; m < n; ++m) {
        const float d = x[m] - y[m];
        s += d * d;
    }
    return s;
}

template <SIMDLevel Level>
inline void block_l2_4j(
        const float* x,
        const float* y,
        int ystride,
        int n,
        int nj,
        float* out) {
    for (int t = 0; t < nj; ++t) {
        out[t] = block_l2<Level>(x, y + static_cast<long>(t) * ystride, n);
    }
}

// COMPILE_SIMD_* is a build-system define (link-time promise that the
// specialization will be available). Mirrors the impl-file guards.
#ifdef COMPILE_SIMD_AVX2
template <>
float block_l2<SIMDLevel::AVX2>(const float* x, const float* y, int n);
#endif

#ifdef COMPILE_SIMD_AVX512
template <>
float block_l2<SIMDLevel::AVX512>(const float* x, const float* y, int n);
#endif

#ifdef COMPILE_SIMD_ARM_SVE
template <>
float block_l2<SIMDLevel::ARM_SVE>(const float* x, const float* y, int n);
#endif

#ifdef COMPILE_SIMD_RISCV_RVV
template <>
float block_l2<SIMDLevel::RISCV_RVV>(const float* x, const float* y, int n);

template <>
void block_l2_4j<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        int ystride,
        int n,
        int nj,
        float* out);
#endif

} // namespace detail
} // namespace faiss
