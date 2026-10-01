/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/utils/fp16.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {
namespace detail {

// c[j] += fp32(x[j]) * w for j < d
template <SIMDLevel Level>
inline void fp16_madd(size_t d, const uint16_t* x, float w, float* c) {
    for (size_t j = 0; j < d; ++j) {
        c[j] += decode_fp16(x[j]) * w;
    }
}

template <SIMDLevel Level>
inline void fp16_to_fp32_kernel(size_t n, const uint16_t* x, float* out) {
    for (size_t j = 0; j < n; ++j) {
        out[j] = decode_fp16(x[j]);
    }
}

#ifdef COMPILE_SIMD_AVX2
template <>
void fp16_madd<SIMDLevel::AVX2>(size_t d, const uint16_t* x, float w, float* c);

template <>
void fp16_to_fp32_kernel<SIMDLevel::AVX2>(
        size_t n,
        const uint16_t* x,
        float* out);
#endif

#ifdef COMPILE_SIMD_ARM_NEON
template <>
void fp16_madd<SIMDLevel::ARM_NEON>(
        size_t d,
        const uint16_t* x,
        float w,
        float* c);

template <>
void fp16_to_fp32_kernel<SIMDLevel::ARM_NEON>(
        size_t n,
        const uint16_t* x,
        float* out);
#endif

} // namespace detail
} // namespace faiss
