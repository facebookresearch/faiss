/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

#include <immintrin.h>

#include <faiss/utils/simd_impl/masked_sum.h>

namespace faiss {
namespace {

template <>
inline float masked_sum<SIMDLevel::AVX512>(
        const float* x,
        const uint8_t* bits,
        size_t d) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        uint16_t mask16;
        memcpy(&mask16, bits + i / 8, sizeof(mask16));
        acc = _mm512_mask_add_ps(acc, mask16, acc, _mm512_loadu_ps(x + i));
    }
    if (i < d) {
        uint32_t mask = bits[i / 8];
        if (d - i > 8) {
            mask |= uint32_t(bits[i / 8 + 1]) << 8;
        }
        const __mmask16 k = mask & ((1u << (d - i)) - 1);
        acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(k, x + i));
    }
    return _mm512_reduce_add_ps(acc);
}

} // namespace
} // namespace faiss
