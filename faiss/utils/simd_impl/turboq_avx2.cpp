/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <faiss/utils/turboq_simd.h>
#include <immintrin.h>

namespace faiss::turboq {

template <>
float masked_sum<SIMDLevel::AVX2>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    const __m256i bit_masks = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    __m256 acc = _mm256_setzero_ps();

    size_t full_bytes = d / 8;
    for (size_t byte_idx = 0; byte_idx < full_bytes; byte_idx++) {
        __m256i byte_broadcast =
                _mm256_set1_epi32(static_cast<int>(bits[byte_idx]));
        __m256i masked = _mm256_and_si256(byte_broadcast, bit_masks);
        __m256i cmp = _mm256_cmpeq_epi32(masked, bit_masks);
        __m256 mask = _mm256_castsi256_ps(cmp);
        __m256 vals = _mm256_loadu_ps(arr + byte_idx * 8);
        acc = _mm256_add_ps(acc, _mm256_and_ps(mask, vals));
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float result = _mm_cvtss_f32(sums);

    // Tail
    size_t tail_start = full_bytes * 8;
    if (tail_start < d) {
        uint8_t last_byte = bits[full_bytes];
        for (size_t j = tail_start; j < d; j++) {
            if (last_byte & (1 << (j - tail_start))) {
                result += arr[j];
            }
        }
    }

    return result;
}

} // namespace faiss::turboq

#endif
