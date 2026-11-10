/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/utils/simd_levels.h>

namespace faiss {

// Internal linkage: each file compiles these with its own flags.
// masked_sum-avx2.h and masked_sum-avx512.h specialize masked_sum.
namespace {

inline float masked_sum_scalar(const float* x, const uint8_t* bits, size_t d) {
    float sum = 0;
    for (size_t j = 0; j < d; j++) {
        if ((bits[j / 8] >> (j % 8)) & 1) {
            sum += x[j];
        }
    }
    return sum;
}

template <SIMDLevel SL>
inline float masked_sum(const float* x, const uint8_t* bits, size_t d) {
    return masked_sum_scalar(x, bits, d);
}

} // namespace

} // namespace faiss
