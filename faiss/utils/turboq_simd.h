/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <faiss/utils/simd_levels.h>

namespace faiss::turboq {

/// Compute sum of arr[j] where bit j of the bitmask is set.
/// Used for SIMD masked accumulation of QJL and 1-bit MSE dot products.
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
float masked_sum(const float* arr, const uint8_t* bits, size_t d);

template <>
inline float masked_sum<SIMDLevel::NONE>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    float result = 0;
    for (size_t byte_idx = 0; byte_idx < (d + 7) / 8; byte_idx++) {
        uint8_t b = bits[byte_idx];
        size_t base = byte_idx * 8;
        size_t end = std::min(base + 8, d);
        for (size_t j = base; j < end; j++) {
            if (b & (1 << (j - base))) {
                result += arr[j];
            }
        }
    }
    return result;
}

} // namespace faiss::turboq
