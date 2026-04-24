/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/turboq_simd.h>

#ifdef COMPILE_SIMD_ARM_NEON

namespace faiss::turboq {

template <>
float masked_sum<SIMDLevel::ARM_NEON>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    return masked_sum<SIMDLevel::NONE>(arr, bits, d);
}

} // namespace faiss::turboq

#endif
