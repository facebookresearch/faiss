/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <faiss/utils/popcount.h>
#include <faiss/utils/simd_levels.h>

namespace faiss::rabitq {

/**
 * Compute dot product between query and binary data using popcount on AND.
 *
 * @param query   Pointer to rearranged rotated query data
 * @param data    Pointer to binary data
 * @param size    Size in bytes
 * @param qb      Number of quantization bits
 * @return        Unsigned integer dot product
 */
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
uint64_t bitwise_and_dot_product(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb);

/**
 * Compute dot product between query and binary data using popcount on XOR.
 *
 * @param query   Pointer to rearranged rotated query data
 * @param data    Pointer to binary data
 * @param size    Size in bytes
 * @param qb      Number of quantization bits
 * @return        Unsigned integer dot product
 */
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
uint64_t bitwise_xor_dot_product(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb);

/**
 * Count total set bits in data.
 *
 * @param data    Pointer to binary data
 * @param size    Size in bytes
 * @return        Total popcount
 */
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
uint64_t popcount(const uint8_t* data, size_t size);

// NONE specializations — scalar fallbacks

template <>
inline uint64_t bitwise_and_dot_product<SIMDLevel::NONE>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;
    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *(const uint64_t*)(data + offset);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(const uint64_t*)(query + j * size + offset);
            sum += popcount64(qv & yv) << j;
        }
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(query + j * size + offset);
            sum += popcount32(qv & yv) << j;
        }
    }
    return sum;
}

template <>
inline uint64_t bitwise_xor_dot_product<SIMDLevel::NONE>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;
    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *(const uint64_t*)(data + offset);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(const uint64_t*)(query + j * size + offset);
            sum += popcount64(qv ^ yv) << j;
        }
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(query + j * size + offset);
            sum += popcount32(qv ^ yv) << j;
        }
    }
    return sum;
}

template <>
inline uint64_t popcount<SIMDLevel::NONE>(const uint8_t* data, size_t size) {
    uint64_t sum = 0;
    size_t offset = 0;
    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *(const uint64_t*)(data + offset);
        sum += popcount64(yv);
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        sum += popcount32(yv);
    }
    return sum;
}

} // namespace faiss::rabitq

/*********************************************************
 * Multi-bit RaBitQ inner product kernels.
 *
 * Compute: sum_i rotated_q[i] * ((sign_bit_i << ex_bits) + ex_code_val_i + cb)
 *
 * Strategy:
 *   ex_bits == 1: Specialized kernel — both sign_bits and ex_code are
 *                 1-bit-per-dim packed, enabling direct bit→mask→float
 *                 conversion with zero per-element extraction.
 *   ex_bits >= 2: Bit-plane decomposition (BMI2 required) — PEXT extracts
 *                 each bit plane in one instruction, then the same
 *                 bit→mask→float kernel computes each plane's dot product.
 *   Fallback:     Scalar extraction via 64-bit window read + shift + mask.
 *********************************************************/
namespace faiss::rabitq::multibit {

/// Scalar inner product for multi-bit RaBitQ.
/// Extracts each code value in O(1) via 64-bit window read + shift + mask.
/// Also serves as the tail handler for SIMD kernels via the @p start parameter.
inline float ip_scalar(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t start,
        size_t d,
        size_t ex_bits,
        float cb) {
    float result = 0.0f;
    const int sign_shift = static_cast<int>(ex_bits);
    const uint64_t code_mask = (1ULL << ex_bits) - 1;
    for (size_t i = start; i < d; i++) {
        int sb = (sign_bits[i / 8] >> (i % 8)) & 1;
        size_t bit_pos = i * ex_bits;
        size_t byte_idx = bit_pos / 8;
        size_t bit_offset = bit_pos % 8;
        uint64_t raw = 0;
        memcpy(&raw, ex_code + byte_idx, sizeof(uint64_t));
        int ex_val = static_cast<int>((raw >> bit_offset) & code_mask);
        result += rotated_q[i] *
                (static_cast<float>((sb << sign_shift) + ex_val) + cb);
    }
    return result;
}

/**
 * Dispatch to the best available kernel for the given ex_bits.
 *
 * @param sign_bits  packed sign bits (1 bit/dim, standard byte packing)
 * @param ex_code    packed extra-bit codes (ex_bits bits/dim)
 * @param rotated_q  rotated query vector (float[d])
 * @param d          dimensionality
 * @param ex_bits    number of extra bits per dimension (nb_bits - 1)
 * @param cb         constant bias: -(2^ex_bits - 0.5)
 * @return           inner product value
 */
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
float compute_inner_product(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb);

// NONE specialization — pure scalar
template <>
inline float compute_inner_product<SIMDLevel::NONE>(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    return ip_scalar(sign_bits, ex_code, rotated_q, 0, d, ex_bits, cb);
}

} // namespace faiss::rabitq::multibit
