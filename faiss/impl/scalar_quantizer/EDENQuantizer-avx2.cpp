/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <faiss/impl/EDENQuantizer.h>
#include <faiss/impl/simdlib/simdlib_avx2.h>

#include <immintrin.h>

namespace faiss {

namespace eden_distance {

namespace {

constexpr int kCodeDotLUTKindHalfByte = 2;

inline __m256i load_code_bytes_8(const uint8_t* const code[8], size_t i) {
    return _mm256_setr_epi32(
            code[0][i],
            code[1][i],
            code[2][i],
            code[3][i],
            code[4][i],
            code[5][i],
            code[6][i],
            code[7][i]);
}

} // namespace

void compute_code_dot_lut_batch_8_avx2(
        const uint8_t* const code[8],
        const float* lut,
        int lut_kind,
        size_t packed_size,
        float dots[8]) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i low_mask = _mm256_set1_epi32(0x0f);
    const __m256i high_offset = _mm256_set1_epi32(16);

    if (lut_kind == kCodeDotLUTKindHalfByte) {
        for (size_t i = 0; i < packed_size; i++) {
            const __m256i bytes = load_code_bytes_8(code, i);
            const __m256i low = _mm256_and_si256(bytes, low_mask);
            const __m256i high =
                    _mm256_add_epi32(_mm256_srli_epi32(bytes, 4), high_offset);
            acc = _mm256_add_ps(acc, _mm256_i32gather_ps(lut, low, 4));
            acc = _mm256_add_ps(acc, _mm256_i32gather_ps(lut, high, 4));
            lut += 32;
        }
    } else {
        for (size_t i = 0; i < packed_size; i++) {
            const __m256i bytes = load_code_bytes_8(code, i);
            acc = _mm256_add_ps(acc, _mm256_i32gather_ps(lut, bytes, 4));
            lut += 256;
        }
    }

    _mm256_storeu_ps(dots, acc);
}

} // namespace eden_distance

} // namespace faiss

#endif
