/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_impl/fp16_kernels.h>

#include <immintrin.h>

namespace faiss {
namespace detail {

template <>
void fp16_madd<SIMDLevel::AVX512>(
        size_t d,
        const uint16_t* x,
        float w,
        float* c) {
    const __m512 wv = _mm512_set1_ps(w);
    size_t j = 0;
    for (; j + 16 <= d; j += 16) {
        const __m512 xv = _mm512_cvtph_ps(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j)));
        _mm512_storeu_ps(
                c + j,
                _mm512_add_ps(_mm512_loadu_ps(c + j), _mm512_mul_ps(xv, wv)));
    }
    for (; j < d; ++j) {
        c[j] += decode_fp16(x[j]) * w;
    }
}

template <>
void fp16_to_fp32_kernel<SIMDLevel::AVX512>(
        size_t n,
        const uint16_t* x,
        float* out) {
    size_t j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m256i h =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j));
        _mm512_storeu_ps(out + j, _mm512_cvtph_ps(h));
    }
    for (; j < n; ++j) {
        out[j] = decode_fp16(x[j]);
    }
}

} // namespace detail
} // namespace faiss
