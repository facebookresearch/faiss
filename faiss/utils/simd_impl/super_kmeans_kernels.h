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
// specialization (ARM_NEON, ARM_SVE, NONE, ...) use it directly.
template <SIMDLevel Level>
inline float block_l2(const float* x, const float* y, int n) {
    float s = 0.0f;
    for (int m = 0; m < n; ++m) {
        const float d = x[m] - y[m];
        s += d * d;
    }
    return s;
}

// Mirrors the impl-file guards; turns off-arch direct calls into a
// compile-time error instead of a linker error.
#if defined(__x86_64__)
template <>
float block_l2<SIMDLevel::AVX2>(const float* x, const float* y, int n);

template <>
float block_l2<SIMDLevel::AVX512>(const float* x, const float* y, int n);
#endif // __x86_64__

} // namespace detail
} // namespace faiss
