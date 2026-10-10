/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <faiss/utils/simd_impl/fp16_kernels.h>

#include <immintrin.h>

namespace faiss {
namespace detail {

template <>
void fp16_madd<SIMDLevel::AVX2>(
        size_t d,
        const uint16_t* x,
        float w,
        float* c) {
    const __m256 wv = _mm256_set1_ps(w);
    size_t j = 0;
    for (; j + 8 <= d; j += 8) {
        const __m256 xv = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + j)));
        _mm256_storeu_ps(
                c + j,
                _mm256_add_ps(_mm256_loadu_ps(c + j), _mm256_mul_ps(xv, wv)));
    }
    for (; j < d; ++j) {
        c[j] += decode_fp16(x[j]) * w;
    }
}

template <>
void fp16_to_fp32_kernel<SIMDLevel::AVX2>(
        size_t n,
        const uint16_t* x,
        float* out) {
    size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const __m128i h =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + j));
        _mm256_storeu_ps(out + j, _mm256_cvtph_ps(h));
    }
    for (; j < n; ++j) {
        out[j] = decode_fp16(x[j]);
    }
}

} // namespace detail
} // namespace faiss

#endif // COMPILE_SIMD_AVX2
