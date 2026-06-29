/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__) || defined(__AVX512BF16__)
#include <immintrin.h>
#endif

namespace faiss {

namespace {

union fp32_bits {
    uint32_t as_u32;
    float as_f32;
};

} // namespace

inline uint16_t encode_bf16(const float f) {
    // Round off
    fp32_bits fp;
    fp.as_f32 = f;
    return static_cast<uint16_t>((fp.as_u32 + 0x8000) >> 16);
}

inline float decode_bf16(const uint16_t v) {
    fp32_bits fp;
    fp.as_u32 = (uint32_t(v) << 16);
    return fp.as_f32;
}

inline void encode_bf16_simd(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
#ifdef __AVX512BF16__
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m256bh encoded = _mm512_cvtneps_pbh(v);
        _mm256_storeu_epi16(dst + i, (__m256i)encoded);
    }
#endif
    for (; i < n; i++) {
        dst[i] = encode_bf16(src[i]);
    }
}

inline void decode_bf16_simd(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));
        __m512i w = _mm512_cvtepu16_epi32(v);
        w = _mm512_slli_epi32(w, 16);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(w));
    }
#endif
    for (; i < n; i++) {
        dst[i] = decode_bf16(src[i]);
    }
}

} // namespace faiss
