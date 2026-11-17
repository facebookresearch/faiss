/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <immintrin.h>

#include <faiss/utils/simd_impl/masked_sum.h>

namespace faiss {
namespace {

template <>
inline float masked_sum<SIMDLevel::AVX2>(
        const float* x,
        const uint8_t* bits,
        size_t d) {
    const __m256i bit_masks = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    __m256 acc = _mm256_setzero_ps();
    const size_t full_bytes = d / 8;
    for (size_t byte_idx = 0; byte_idx < full_bytes; byte_idx++) {
        const __m256i byte_broadcast = _mm256_set1_epi32(bits[byte_idx]);
        const __m256i masked = _mm256_and_si256(byte_broadcast, bit_masks);
        const __m256 mask =
                _mm256_castsi256_ps(_mm256_cmpeq_epi32(masked, bit_masks));
        const __m256 vals = _mm256_loadu_ps(x + byte_idx * 8);
        acc = _mm256_add_ps(acc, _mm256_and_ps(mask, vals));
    }
    __m128 sum128 = _mm_add_ps(
            _mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums) +
            masked_sum_scalar(x + full_bytes * 8, bits + full_bytes, d % 8);
}

} // namespace
} // namespace faiss
