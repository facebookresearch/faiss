/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_simd.h>

#ifdef COMPILE_SIMD_ARM_NEON

#include <arm_neon.h>

#include <array>

namespace faiss::rabitq {

namespace {

inline uint32x4_t accumulate_popcount(uint32x4_t acc, uint8x16_t value) {
    const uint16x8_t pair_counts = vpaddlq_u8(vcntq_u8(value));
    return vpadalq_u16(acc, pair_counts);
}

inline uint64_t reduce_popcount(uint32x4_t acc) {
    return vaddlvq_u32(acc);
}

} // namespace

template <>
uint64_t bitwise_and_dot_product<SIMDLevel::ARM_NEON>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    std::array<uint32x4_t, 8> accumulators;
    for (size_t j = 0; j < qb; j++) {
        accumulators[j] = vdupq_n_u32(0);
    }

    size_t offset = 0;
    for (; offset + 16 <= size; offset += 16) {
        const uint8x16_t value = vld1q_u8(data + offset);
        for (size_t j = 0; j < qb; j++) {
            const uint8x16_t query_value = vld1q_u8(query + j * size + offset);
            accumulators[j] = accumulate_popcount(
                    accumulators[j], vandq_u8(query_value, value));
        }
    }

    uint64_t result = 0;
    for (size_t j = 0; j < qb; j++) {
        result += reduce_popcount(accumulators[j]) << j;
    }
    for (; offset + 8 <= size; offset += 8) {
        uint64_t value;
        memcpy(&value, data + offset, sizeof(value));
        for (size_t j = 0; j < qb; j++) {
            uint64_t query_value;
            memcpy(&query_value,
                   query + j * size + offset,
                   sizeof(query_value));
            result += static_cast<uint64_t>(popcount64(query_value & value))
                    << j;
        }
    }
    for (; offset < size; offset++) {
        const uint8_t value = data[offset];
        for (size_t j = 0; j < qb; j++) {
            result += static_cast<uint64_t>(
                              popcount32(query[j * size + offset] & value))
                    << j;
        }
    }
    return result;
}

template <>
BitwiseAndDotProductResult bitwise_and_dot_product_with_popcount<
        SIMDLevel::ARM_NEON>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    std::array<uint32x4_t, 8> dot_accumulators;
    for (size_t j = 0; j < qb; j++) {
        dot_accumulators[j] = vdupq_n_u32(0);
    }
    uint32x4_t popcount_accumulator = vdupq_n_u32(0);

    size_t offset = 0;
    for (; offset + 16 <= size; offset += 16) {
        const uint8x16_t value = vld1q_u8(data + offset);
        popcount_accumulator = accumulate_popcount(popcount_accumulator, value);
        for (size_t j = 0; j < qb; j++) {
            const uint8x16_t query_value = vld1q_u8(query + j * size + offset);
            dot_accumulators[j] = accumulate_popcount(
                    dot_accumulators[j], vandq_u8(query_value, value));
        }
    }

    uint64_t dot_product = 0;
    for (size_t j = 0; j < qb; j++) {
        dot_product += reduce_popcount(dot_accumulators[j]) << j;
    }
    uint64_t popcount_sum = reduce_popcount(popcount_accumulator);
    for (; offset + 8 <= size; offset += 8) {
        uint64_t value;
        memcpy(&value, data + offset, sizeof(value));
        popcount_sum += popcount64(value);
        for (size_t j = 0; j < qb; j++) {
            uint64_t query_value;
            memcpy(&query_value,
                   query + j * size + offset,
                   sizeof(query_value));
            dot_product +=
                    static_cast<uint64_t>(popcount64(query_value & value)) << j;
        }
    }
    for (; offset < size; offset++) {
        const uint8_t value = data[offset];
        popcount_sum += popcount32(value);
        for (size_t j = 0; j < qb; j++) {
            dot_product += static_cast<uint64_t>(
                                   popcount32(query[j * size + offset] & value))
                    << j;
        }
    }
    return {dot_product, popcount_sum};
}

template <>
uint64_t bitwise_xor_dot_product<SIMDLevel::ARM_NEON>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    std::array<uint32x4_t, 8> accumulators;
    for (size_t j = 0; j < qb; j++) {
        accumulators[j] = vdupq_n_u32(0);
    }

    size_t offset = 0;
    for (; offset + 16 <= size; offset += 16) {
        const uint8x16_t value = vld1q_u8(data + offset);
        for (size_t j = 0; j < qb; j++) {
            const uint8x16_t query_value = vld1q_u8(query + j * size + offset);
            accumulators[j] = accumulate_popcount(
                    accumulators[j], veorq_u8(query_value, value));
        }
    }

    uint64_t result = 0;
    for (size_t j = 0; j < qb; j++) {
        result += reduce_popcount(accumulators[j]) << j;
    }
    for (; offset + 8 <= size; offset += 8) {
        uint64_t value;
        memcpy(&value, data + offset, sizeof(value));
        for (size_t j = 0; j < qb; j++) {
            uint64_t query_value;
            memcpy(&query_value,
                   query + j * size + offset,
                   sizeof(query_value));
            result += static_cast<uint64_t>(popcount64(query_value ^ value))
                    << j;
        }
    }
    for (; offset < size; offset++) {
        const uint8_t value = data[offset];
        for (size_t j = 0; j < qb; j++) {
            result += static_cast<uint64_t>(
                              popcount32(query[j * size + offset] ^ value))
                    << j;
        }
    }
    return result;
}

template <>
uint64_t popcount<SIMDLevel::ARM_NEON>(const uint8_t* data, size_t size) {
    uint32x4_t accumulator = vdupq_n_u32(0);
    size_t offset = 0;
    for (; offset + 16 <= size; offset += 16) {
        accumulator = accumulate_popcount(accumulator, vld1q_u8(data + offset));
    }

    uint64_t result = reduce_popcount(accumulator);
    for (; offset + 8 <= size; offset += 8) {
        uint64_t value;
        memcpy(&value, data + offset, sizeof(value));
        result += popcount64(value);
    }
    for (; offset < size; offset++) {
        result += popcount32(data[offset]);
    }
    return result;
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
