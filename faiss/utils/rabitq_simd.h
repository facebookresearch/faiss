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
#include <cstring>

#include <faiss/utils/popcount.h>
#include <faiss/utils/simd_levels.h>

namespace faiss::rabitq {

/// SIMD levels with RaBitQ query/LUT quantization implementations.
constexpr int RABITQ_QUANTIZATION_SIMD_LEVELS = (1 << int(SIMDLevel::NONE)) |
        (1 << int(SIMDLevel::AVX2)) | (1 << int(SIMDLevel::AVX512));

struct BitwiseAndDotProductResult {
    uint64_t dot_product;
    uint64_t popcount;
};

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
 * Compute bitwise_and_dot_product(query, data, size, qb) and popcount(data,
 * size) in one pass over data.
 */
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
BitwiseAndDotProductResult bitwise_and_dot_product_with_popcount(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb);

/// Four independent AND-dot-product/popcount pairs for a four-plane query.
/// Query contains 4 * size bytes; each data pointer contains `size` bytes.
/// Byte-aligned buffers and arbitrary byte counts are supported.
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
inline void bitwise_q4_batch_4(
        const uint8_t* query,
        const uint8_t* const* data,
        size_t size,
        BitwiseAndDotProductResult* results) {
    for (size_t i = 0; i < 4; ++i) {
        results[i] = bitwise_and_dot_product_with_popcount<SL>(
                query, data[i], size, 4);
    }
}

template <>
void bitwise_q4_batch_4<SIMDLevel::AVX512_VPOPCNT>(
        const uint8_t* query,
        const uint8_t* const* data,
        size_t size,
        BitwiseAndDotProductResult* results);

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

/**
 * Rearrange per-dimension quantized query codes into bit-plane layout.
 *
 * @p rotated_qq holds one qb-bit code per dimension (one byte each, value in
 * [0, 2^qb)). @p out receives qb bit-planes of ((d + 7) / 8) bytes each:
 * bit-plane j packs bit j of code i into bit (i % 8) of byte (i / 8). @p out
 * must have room for qb * ((d + 7) / 8) bytes and is fully overwritten.
 *
 * @param rotated_qq  per-dimension codes (d bytes)
 * @param d           dimensionality
 * @param qb          quantization bits per dimension (1..8)
 * @param out         bit-plane output buffer (qb * ((d + 7) / 8) bytes)
 */
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
void rearrange_bit_planes(
        const uint8_t* rotated_qq,
        size_t d,
        size_t qb,
        uint8_t* out);

/// Find min/max of one 16-entry FastScan LUT row.
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
void lut_minmax_16(const float* tab, float& mn, float& mx);

/// Find min/max of an arbitrary-length float vector. For n == 0, mn and mx
/// are left unchanged.
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
void minmax_values(const float* values, size_t n, float& mn, float& mx);

/// Quantize one 16-entry FastScan LUT row with non-negative half-up rounding.
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
void lut_quantize_16_to_uint8(
        const float* tab,
        float mn,
        float a,
        uint8_t* out);

/// Quantize rotated query values and accumulate query correction terms.
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
void quantize_query_values(
        const float* rq,
        size_t d,
        float v_min,
        float inv_delta,
        uint8_t max_code,
        bool centered,
        uint8_t* rqq,
        size_t& sum_qq,
        int64_t& sum2_signed_odd_int);

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
        uint64_t yv;
        std::memcpy(&yv, data + offset, sizeof(yv));
        for (int j = 0; j < qb; j++) {
            uint64_t qv;
            std::memcpy(&qv, query + j * size + offset, sizeof(qv));
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
inline BitwiseAndDotProductResult bitwise_and_dot_product_with_popcount<
        SIMDLevel::NONE>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t dot_product = 0;
    uint64_t popcount_sum = 0;
    size_t offset = 0;
    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        uint64_t yv;
        std::memcpy(&yv, data + offset, sizeof(yv));
        popcount_sum += popcount64(yv);
        for (int j = 0; j < qb; j++) {
            uint64_t qv;
            std::memcpy(&qv, query + j * size + offset, sizeof(qv));
            dot_product += popcount64(qv & yv) << j;
        }
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        popcount_sum += popcount32(yv);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(query + j * size + offset);
            dot_product += popcount32(qv & yv) << j;
        }
    }
    return {dot_product, popcount_sum};
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
        uint64_t yv;
        std::memcpy(&yv, data + offset, sizeof(yv));
        for (int j = 0; j < qb; j++) {
            uint64_t qv;
            std::memcpy(&qv, query + j * size + offset, sizeof(qv));
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
        uint64_t yv;
        std::memcpy(&yv, data + offset, sizeof(yv));
        sum += popcount64(yv);
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        sum += popcount32(yv);
    }
    return sum;
}

template <>
inline void rearrange_bit_planes<SIMDLevel::NONE>(
        const uint8_t* rotated_qq,
        size_t d,
        size_t qb,
        uint8_t* out) {
    const size_t offset = (d + 7) / 8;
    memset(out, 0, offset * qb);
    for (size_t idim = 0; idim < d; idim++) {
        for (size_t iv = 0; iv < qb; iv++) {
            const bool bit = ((rotated_qq[idim] & (1 << iv)) != 0);
            out[iv * offset + idim / 8] |= bit ? (1 << (idim % 8)) : 0;
        }
    }
}

inline uint8_t round_clamped_byte_scalar(float x, uint8_t max_code) {
    if (x <= 0.0f) {
        return 0;
    }
    if (x >= max_code) {
        return max_code;
    }
    return static_cast<uint8_t>(static_cast<int>(x + 0.5f));
}

inline uint8_t round_nonnegative_byte_scalar(float x) {
    return round_clamped_byte_scalar(x, 255);
}

template <>
inline void lut_minmax_16<SIMDLevel::NONE>(
        const float* tab,
        float& mn,
        float& mx) {
    mn = tab[0];
    mx = tab[0];
    for (size_t s = 1; s < 16; s++) {
        mn = std::min(mn, tab[s]);
        mx = std::max(mx, tab[s]);
    }
}

template <>
inline void minmax_values<SIMDLevel::NONE>(
        const float* values,
        size_t n,
        float& mn,
        float& mx) {
    if (n == 0) {
        return;
    }
    mn = values[0];
    mx = values[0];
    for (size_t i = 1; i < n; i++) {
        mn = std::min(mn, values[i]);
        mx = std::max(mx, values[i]);
    }
}

template <>
inline void lut_quantize_16_to_uint8<SIMDLevel::NONE>(
        const float* tab,
        float mn,
        float a,
        uint8_t* out) {
    for (size_t s = 0; s < 16; s++) {
        out[s] = round_nonnegative_byte_scalar(a * (tab[s] - mn));
    }
}

template <>
inline void quantize_query_values<SIMDLevel::NONE>(
        const float* rq,
        size_t d,
        float v_min,
        float inv_delta,
        uint8_t max_code,
        bool centered,
        uint8_t* rqq,
        size_t& sum_qq,
        int64_t& sum2_signed_odd_int) {
    if (centered) {
        for (size_t i = 0; i < d; i++) {
            const uint8_t v_qq = round_clamped_byte_scalar(
                    (rq[i] - v_min) * inv_delta, max_code);
            rqq[i] = v_qq;
            sum_qq += v_qq;

            const int64_t signed_odd_int = int64_t(v_qq) * 2 - max_code;
            sum2_signed_odd_int += signed_odd_int * signed_odd_int;
        }
    } else {
        for (size_t i = 0; i < d; i++) {
            const uint8_t v_qq = round_clamped_byte_scalar(
                    (rq[i] - v_min) * inv_delta, max_code);
            rqq[i] = v_qq;
            sum_qq += v_qq;
        }
    }
}

} // namespace faiss::rabitq

/*********************************************************
 * Multi-bit RaBitQ inner product kernels.
 *
 * Score = sum_i rotated_q[i] * ((sign_i << ex_bits) + ex_i + cb)
 *
 * ex_bits == 1: bits map straight to floats. No extraction.
 * ex_bits 2..7: BMI2 PEXT extracts each bit plane.
 * ex_bits == 8: whole bytes. No extraction and no BMI2.
 * Fallback:     scalar math reads one code at a time.
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
    if (ex_bits == 8) {
        // RBQ9 is byte-aligned, including the last dimension. Do not require
        // trailing factor bytes for the scalar reference or SIMD tail.
        for (size_t i = start; i < d; ++i) {
            const int sb = (sign_bits[i / 8] >> (i % 8)) & 1;
            result += rotated_q[i] *
                    (static_cast<float>((sb << 8) + ex_code[i]) + cb);
        }
        return result;
    }
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

/**
 * compute_inner_product with the sign term supplied by the caller:
 *
 *   (1 << ex_bits) * sign_ip + sum_i rotated_q[i] * ex_i + cb * q_sum
 *
 * With sign_ip = fvec_masked_sum(rotated_q, sign_bits, d) and
 * q_sum = sum_i rotated_q[i], this equals
 * compute_inner_product(sign_bits, ex_code, rotated_q, d, ex_bits, cb) up to
 * float rounding. It serves codes whose sign bits are not stored next to
 * ex_code, e.g. split between memory and disk: the caller accumulates the
 * sign term piece by piece with fvec_masked_sum. q_sum depends on the
 * query only, so it can be computed once per query.
 *
 * Unlike compute_inner_product, this reads only the (d * ex_bits + 7) / 8
 * bytes of ex_code and never the bytes after them.
 *
 * @param ex_code    packed extra-bit codes (ex_bits bits/dim); may be null
 *                   when ex_bits == 0
 * @param rotated_q  rotated query vector (float[d])
 * @param d          dimensionality
 * @param ex_bits    number of extra bits per dimension, in [0, 8]
 * @param cb         constant bias: -(2^ex_bits - 0.5)
 * @param sign_ip    sum_i rotated_q[i] * sign_i
 * @param q_sum      sum_i rotated_q[i]
 * @return           inner product value
 */
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
float compute_inner_product_with_sign_ip(
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb,
        float sign_ip,
        float q_sum);

/// SIMD levels with a split inner-product kernel.
constexpr int RABITQ_SPLIT_IP_SIMD_LEVELS = (1 << int(SIMDLevel::NONE)) |
        (1 << int(SIMDLevel::AVX2)) | (1 << int(SIMDLevel::AVX512));

/// Load the code bytes at ex_code + offset into the low bytes of a uint64,
/// reading 8 bytes when they all lie within the first total_bytes bytes of
/// ex_code and only the remaining ones otherwise.
static inline uint64_t load_ex_code_window(
        const uint8_t* ex_code,
        size_t offset,
        size_t total_bytes) {
    uint64_t window = 0;
    if (offset + sizeof(window) <= total_bytes) {
        memcpy(&window, ex_code + offset, sizeof(window));
    } else {
        memcpy(&window, ex_code + offset, total_bytes - offset);
    }
    return window;
}

/// Scalar sum over [start, d) of rotated_q[i] * ex_i, for ex_bits in [1, 8].
/// Reads only the (d * ex_bits + 7) / 8 bytes of ex_code. Also serves as the
/// tail handler of the SIMD kernels.
static inline float ex_ip_scalar(
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t start,
        size_t d,
        size_t ex_bits) {
    float result = 0.0f;
    if (ex_bits == 8) {
        for (size_t i = start; i < d; ++i) {
            result += rotated_q[i] * static_cast<float>(ex_code[i]);
        }
        return result;
    }
    const unsigned code_mask = (1u << ex_bits) - 1;
    for (size_t i = start; i < d; i++) {
        const size_t bit_pos = i * ex_bits;
        const size_t byte_idx = bit_pos / 8;
        const size_t bit_offset = bit_pos % 8;
        // With ex_bits <= 8 a code spans at most two bytes.
        unsigned raw = ex_code[byte_idx];
        if (bit_offset + ex_bits > 8) {
            raw |= static_cast<unsigned>(ex_code[byte_idx + 1]) << 8;
        }
        const unsigned ex_val = (raw >> bit_offset) & code_mask;
        result += rotated_q[i] * static_cast<float>(ex_val);
    }
    return result;
}

/// The terms compute_inner_product_with_sign_ip adds to the extra-bit inner
/// product ex_ip.
static inline float add_sign_and_bias_terms(
        float ex_ip,
        size_t ex_bits,
        float cb,
        float sign_ip,
        float q_sum) {
    return static_cast<float>(1u << ex_bits) * sign_ip + ex_ip + cb * q_sum;
}

template <>
inline float compute_inner_product_with_sign_ip<SIMDLevel::NONE>(
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb,
        float sign_ip,
        float q_sum) {
    const float ex_ip = ex_bits == 0
            ? 0.0f
            : ex_ip_scalar(ex_code, rotated_q, 0, d, ex_bits);
    return add_sign_and_bias_terms(ex_ip, ex_bits, cb, sign_ip, q_sum);
}

} // namespace faiss::rabitq::multibit
