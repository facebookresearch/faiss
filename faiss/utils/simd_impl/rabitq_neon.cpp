/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_simd.h>

#ifdef COMPILE_SIMD_ARM_NEON

namespace faiss::rabitq {

template <>
uint64_t bitwise_and_dot_product<SIMDLevel::ARM_NEON>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return bitwise_and_dot_product<SIMDLevel::NONE>(query, data, size, qb);
}

template <>
uint64_t bitwise_xor_dot_product<SIMDLevel::ARM_NEON>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return bitwise_xor_dot_product<SIMDLevel::NONE>(query, data, size, qb);
}

template <>
uint64_t popcount<SIMDLevel::ARM_NEON>(const uint8_t* data, size_t size) {
    return popcount<SIMDLevel::NONE>(data, size);
}

} // namespace faiss::rabitq

namespace faiss::rabitq::multibit {

template <>
float compute_inner_product<SIMDLevel::ARM_NEON>(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    return compute_inner_product<SIMDLevel::NONE>(
            sign_bits, ex_code, rotated_q, d, ex_bits, cb);
}

} // namespace faiss::rabitq::multibit

#endif // COMPILE_SIMD_ARM_NEON
