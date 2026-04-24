/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/utils/turboq_simd.h>
#include <immintrin.h>

#include <cstring>

namespace faiss::turboq {

template <>
float masked_sum<SIMDLevel::AVX512>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    __m512 acc = _mm512_setzero_ps();

    size_t i = 0;
    size_t full_16 = (d / 16) * 16;
    for (; i < full_16; i += 16) {
        uint16_t mask16;
        memcpy(&mask16, bits + i / 8, sizeof(mask16));
        __mmask16 k = _cvtu32_mask16(mask16);
        __m512 vals = _mm512_loadu_ps(arr + i);
        acc = _mm512_mask_add_ps(acc, k, acc, vals);
    }

    float result = _mm512_reduce_add_ps(acc);

    // Tail: remaining elements
    if (i < d) {
        size_t remaining = d - i;
        __mmask16 tail_mask = _cvtu32_mask16((1u << remaining) - 1);
        __m512 tail_vals = _mm512_maskz_loadu_ps(tail_mask, arr + i);

        // Load remaining bits
        uint16_t bits_tail = 0;
        size_t bytes_remaining = (remaining + 7) / 8;
        memcpy(&bits_tail, bits + i / 8, bytes_remaining);
        __mmask16 bits_k = _cvtu32_mask16(bits_tail);
        __mmask16 combined = _kand_mask16(tail_mask, bits_k);

        __m512 masked_tail = _mm512_maskz_mov_ps(combined, tail_vals);
        result += _mm512_reduce_add_ps(masked_tail);
    }

    return result;
}

} // namespace faiss::turboq

#endif
