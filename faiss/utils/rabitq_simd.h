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

// Only include x86 SIMD intrinsics on x86/x86_64 architectures
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
        defined(_M_IX86)
#include <immintrin.h>
#endif // defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||

namespace faiss::rabitq {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
        defined(_M_IX86)
/**
 * Returns the lookup table for AVX512 popcount operations.
 * This table is used for lookup-based popcount implementation.
 *
 * Source: https://github.com/WojciechMula/sse-popcount.
 *
 * @return Lookup table as __m512i register
 */
#if defined(__AVX512F__)
inline __m512i get_lookup_512() {
    return _mm512_set_epi8(
            /* f */ 4,
            /* e */ 3,
            /* d */ 3,
            /* c */ 2,
            /* b */ 3,
            /* a */ 2,
            /* 9 */ 2,
            /* 8 */ 1,
            /* 7 */ 3,
            /* 6 */ 2,
            /* 5 */ 2,
            /* 4 */ 1,
            /* 3 */ 2,
            /* 2 */ 1,
            /* 1 */ 1,
            /* 0 */ 0,
            /* f */ 4,
            /* e */ 3,
            /* d */ 3,
            /* c */ 2,
            /* b */ 3,
            /* a */ 2,
            /* 9 */ 2,
            /* 8 */ 1,
            /* 7 */ 3,
            /* 6 */ 2,
            /* 5 */ 2,
            /* 4 */ 1,
            /* 3 */ 2,
            /* 2 */ 1,
            /* 1 */ 1,
            /* 0 */ 0,
            /* f */ 4,
            /* e */ 3,
            /* d */ 3,
            /* c */ 2,
            /* b */ 3,
            /* a */ 2,
            /* 9 */ 2,
            /* 8 */ 1,
            /* 7 */ 3,
            /* 6 */ 2,
            /* 5 */ 2,
            /* 4 */ 1,
            /* 3 */ 2,
            /* 2 */ 1,
            /* 1 */ 1,
            /* 0 */ 0,
            /* f */ 4,
            /* e */ 3,
            /* d */ 3,
            /* c */ 2,
            /* b */ 3,
            /* a */ 2,
            /* 9 */ 2,
            /* 8 */ 1,
            /* 7 */ 3,
            /* 6 */ 2,
            /* 5 */ 2,
            /* 4 */ 1,
            /* 3 */ 2,
            /* 2 */ 1,
            /* 1 */ 1,
            /* 0 */ 0);
}
#endif // defined(__AVX512F__)
#if defined(__AVX2__)
/**
 * Returns the lookup table for AVX2 popcount operations.
 * This table is used for lookup-based popcount implementation.
 *
 * @return Lookup table as __m256i register
 */
inline __m256i get_lookup_256() {
    return _mm256_setr_epi8(
            /* 0 */ 0,
            /* 1 */ 1,
            /* 2 */ 1,
            /* 3 */ 2,
            /* 4 */ 1,
            /* 5 */ 2,
            /* 6 */ 2,
            /* 7 */ 3,
            /* 8 */ 1,
            /* 9 */ 2,
            /* a */ 2,
            /* b */ 3,
            /* c */ 2,
            /* d */ 3,
            /* e */ 3,
            /* f */ 4,
            /* 0 */ 0,
            /* 1 */ 1,
            /* 2 */ 1,
            /* 3 */ 2,
            /* 4 */ 1,
            /* 5 */ 2,
            /* 6 */ 2,
            /* 7 */ 3,
            /* 8 */ 1,
            /* 9 */ 2,
            /* a */ 2,
            /* b */ 3,
            /* c */ 2,
            /* d */ 3,
            /* e */ 3,
            /* f */ 4);
}
#endif // defined(__AVX2__)

#if defined(__AVX512F__)
/**
 * Popcount for a 512-bit register, using lookup tables if necessary.
 *
 * @param v Input vector to count bits in
 * @return Vector int32_t[16] with popcount results.
 */
inline __m512i popcount_512(__m512i v) {
#if defined(__AVX512VPOPCNTDQ__)
    return _mm512_popcnt_epi64(v);
#else
    const __m512i lookup = get_lookup_512();
    const __m512i low_mask = _mm512_set1_epi8(0x0f);

    const __m512i lo = _mm512_and_si512(v, low_mask);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), low_mask);
    const __m512i popcnt_lo = _mm512_shuffle_epi8(lookup, lo);
    const __m512i popcnt_hi = _mm512_shuffle_epi8(lookup, hi);
    const __m512i popcnt = _mm512_add_epi8(popcnt_lo, popcnt_hi);
    return _mm512_sad_epu8(_mm512_setzero_si512(), popcnt);
#endif // defined(__AVX512VPOPCNTDQ__)
}
#endif // defined(__AVX512F__)

#if defined(__AVX2__)
/**
 * Popcount for a 256-bit register, using lookup tables if necessary.
 *
 * @param v Input vector to count bits in
 * @return uint64_t[4] of popcounts for each portion of the input vector.
 */
inline __m256i popcount_256(__m256i v) {
    const __m256i lookup = get_lookup_256();
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    const __m256i lo = _mm256_and_si256(v, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    const __m256i popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
    const __m256i popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);
    // Reduce uint8_t[32] into uint64_t[4] by addition.
    return _mm256_sad_epu8(_mm256_setzero_si256(), popcnt);
}

inline uint64_t reduce_add_256(__m256i v) {
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256((__m256i*)lanes, v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
#endif // defined(__AVX2__)

#if defined(__SSE4_1__)
inline __m128i popcount_128(__m128i v) {
    // Scalar popcount for each 64-bit lane
    uint64_t lane0 = _mm_extract_epi64(v, 0);
    uint64_t lane1 = _mm_extract_epi64(v, 1);
    uint64_t pop0 = __builtin_popcountll(lane0);
    uint64_t pop1 = __builtin_popcountll(lane1);
    return _mm_set_epi64x(pop1, pop0);
}

inline uint64_t reduce_add_128(__m128i v) {
    alignas(16) uint64_t lanes[2];
    _mm_store_si128((__m128i*)lanes, v);
    return lanes[0] + lanes[1];
}
#endif // defined(__SSE4_1__)
#endif // defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||

/**
 * Compute dot product between query and binary data using popcount operations.
 *
 * @param query          Pointer to rearranged rotated query data
 * @param data    Pointer to binary data
 * @param d              Dimension
 * @param qb             Number of quantization bits
 * @return               Unsigned integer dot product
 */
inline uint64_t bitwise_and_dot_product(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;
#if defined(__AVX512F__)
    // Handle 512-bit chunks.
    if (size_t step = 512 / 8; offset + step <= size) {
        __m512i sum_512 = _mm512_setzero_si512();
        for (; offset + step <= size; offset += step) {
            __m512i v_x = _mm512_loadu_si512((const __m512i*)(data + offset));
            for (int j = 0; j < qb; j++) {
                __m512i v_q = _mm512_loadu_si512(
                        (const __m512i*)(query + j * size + offset));
                __m512i v_and = _mm512_and_si512(v_q, v_x);
                __m512i v_popcnt = popcount_512(v_and);
                __m512i v_shifted = _mm512_slli_epi64(v_popcnt, j);
                sum_512 = _mm512_add_epi64(sum_512, v_shifted);
            }
        }
        sum += _mm512_reduce_add_epi64(sum_512);
    }
#endif // defined(__AVX512F__)
#if defined(__AVX2__)
    if (size_t step = 256 / 8; offset + step <= size) {
        __m256i sum_256 = _mm256_setzero_si256();
        for (; offset + step <= size; offset += step) {
            __m256i v_x = _mm256_loadu_si256((const __m256i*)(data + offset));
            for (int j = 0; j < qb; j++) {
                __m256i v_q = _mm256_loadu_si256(
                        (const __m256i*)(query + j * size + offset));
                __m256i v_and = _mm256_and_si256(v_q, v_x);
                __m256i v_popcnt = popcount_256(v_and);
                __m256i v_shifted = _mm256_slli_epi64(v_popcnt, j);
                sum_256 = _mm256_add_epi64(sum_256, v_shifted);
            }
        }
        sum += reduce_add_256(sum_256);
    }
#endif // defined(__AVX2__)
#if defined(__SSE4_1__)
    __m128i sum_128 = _mm_setzero_si128();
    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128((const __m128i*)(data + offset));
        for (int j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    (const __m128i*)(query + j * size + offset));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            __m128i v_popcnt = popcount_128(v_and);
            __m128i v_shifted = _mm_slli_epi64(v_popcnt, j);
            sum_128 = _mm_add_epi64(sum_128, v_shifted);
        }
    }
    sum += reduce_add_128(sum_128);
#endif // defined(__SSE4_1__)
    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *(const uint64_t*)(data + offset);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(const uint64_t*)(query + j * size + offset);
            sum += __builtin_popcountll(qv & yv) << j;
        }
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(query + j * size + offset);
            sum += __builtin_popcount(qv & yv) << j;
        }
    }
    return sum;
}

/**
 * Compute dot product between query and binary data using popcount operations.
 *
 * @param query          Pointer to rearranged rotated query data
 * @param data    Pointer to binary data
 * @param d              Dimension
 * @param qb             Number of quantization bits
 * @return               Unsigned integer dot product
 */
inline uint64_t bitwise_xor_dot_product(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;
#if defined(__AVX512F__)
    // Handle 512-bit chunks.
    if (size_t step = 512 / 8; offset + step <= size) {
        __m512i sum_512 = _mm512_setzero_si512();
        for (; offset + step <= size; offset += step) {
            __m512i v_x = _mm512_loadu_si512((const __m512i*)(data + offset));
            for (int j = 0; j < qb; j++) {
                __m512i v_q = _mm512_loadu_si512(
                        (const __m512i*)(query + j * size + offset));
                __m512i v_xor = _mm512_xor_si512(v_q, v_x);
                __m512i v_popcnt = popcount_512(v_xor);
                __m512i v_shifted = _mm512_slli_epi64(v_popcnt, j);
                sum_512 = _mm512_add_epi64(sum_512, v_shifted);
            }
        }
        sum += _mm512_reduce_add_epi64(sum_512);
    }
#endif
#if defined(__AVX2__)
    if (size_t step = 256 / 8; offset + step <= size) {
        __m256i sum_256 = _mm256_setzero_si256();
        for (; offset + step <= size; offset += step) {
            __m256i v_x = _mm256_loadu_si256((const __m256i*)(data + offset));
            for (int j = 0; j < qb; j++) {
                __m256i v_q = _mm256_loadu_si256(
                        (const __m256i*)(query + j * size + offset));
                __m256i v_xor = _mm256_xor_si256(v_q, v_x);
                __m256i v_popcnt = popcount_256(v_xor);
                __m256i v_shifted = _mm256_slli_epi64(v_popcnt, j);
                sum_256 = _mm256_add_epi64(sum_256, v_shifted);
            }
        }
        sum += reduce_add_256(sum_256);
    }
#endif
#if defined(__SSE4_1__)
    __m128i sum_128 = _mm_setzero_si128();
    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128((const __m128i*)(data + offset));
        for (int j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    (const __m128i*)(query + j * size + offset));
            __m128i v_xor = _mm_xor_si128(v_q, v_x);
            __m128i v_popcnt = popcount_128(v_xor);
            __m128i v_shifted = _mm_slli_epi64(v_popcnt, j);
            sum_128 = _mm_add_epi64(sum_128, v_shifted);
        }
    }
    sum += reduce_add_128(sum_128);
#endif
    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *(const uint64_t*)(data + offset);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(const uint64_t*)(query + j * size + offset);
            sum += __builtin_popcountll(qv ^ yv) << j;
        }
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(query + j * size + offset);
            sum += __builtin_popcount(qv ^ yv) << j;
        }
    }
    return sum;
}

inline uint64_t popcount(const uint8_t* data, size_t size) {
    uint64_t sum = 0;
    size_t offset = 0;
#if defined(__AVX512F__)
    // Handle 512-bit chunks.
    if (offset + 512 / 8 <= size) {
        __m512i sum_512 = _mm512_setzero_si512();
        for (size_t end; (end = offset + 512 / 8) <= size; offset = end) {
            __m512i v_x = _mm512_loadu_si512((const __m512i*)(data + offset));
            __m512i v_popcnt = popcount_512(v_x);
            sum_512 = _mm512_add_epi64(sum_512, v_popcnt);
        }
        sum += _mm512_reduce_add_epi64(sum_512);
    }
#endif // defined(__AVX512F__)
#if defined(__AVX2__)
    if (offset + 256 / 8 <= size) {
        __m256i sum_256 = _mm256_setzero_si256();
        for (size_t end; (end = offset + 256 / 8) <= size; offset = end) {
            __m256i v_x = _mm256_loadu_si256((const __m256i*)(data + offset));
            __m256i v_popcnt = popcount_256(v_x);
            sum_256 = _mm256_add_epi64(sum_256, v_popcnt);
        }
        sum += reduce_add_256(sum_256);
    }
#endif // defined(__AVX2__)
#if defined(__SSE4_1__)
    __m128i sum_128 = _mm_setzero_si128();
    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128((const __m128i*)(data + offset));
        sum_128 = _mm_add_epi64(sum_128, popcount_128(v_x));
    }
    sum += reduce_add_128(sum_128);
#endif // defined(__SSE4_1__)

    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *(const uint64_t*)(data + offset);
        sum += __builtin_popcountll(yv);
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        sum += __builtin_popcount(yv);
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

#if defined(__x86_64__) || defined(_M_X64)

#if defined(__AVX2__)
/// Horizontal sum of 8 floats in a __m256 register.
inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, shuf));
}
#endif // __AVX2__

/*********************************************************
 * Specialized 1-bit kernels (ex_bits == 1).
 *
 * For 1 extra bit, both sign_bits and ex_code are 1-bit-per-dim packed,
 * so we convert bits to floats directly — no extraction loops needed.
 *********************************************************/

#if defined(__AVX512F__)
/// AVX-512: 16 dims/iter, ex_bits == 1.
inline float ip_1exbit_avx512(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        float cb) {
    __m512 acc = _mm512_setzero_ps();
    const __m512 v_cb = _mm512_set1_ps(cb);
    const __m512 v_two = _mm512_set1_ps(2.0f);
    const __m512 v_one = _mm512_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        uint16_t sb16;
        memcpy(&sb16, sign_bits + i / 8, sizeof(uint16_t));
        uint16_t eb16;
        memcpy(&eb16, ex_code + i / 8, sizeof(uint16_t));

        __m512 sb_f = _mm512_maskz_mov_ps(_cvtu32_mask16(sb16), v_one);
        __m512 eb_f = _mm512_maskz_mov_ps(_cvtu32_mask16(eb16), v_one);

        __m512 recon = _mm512_add_ps(_mm512_fmadd_ps(sb_f, v_two, eb_f), v_cb);
        __m512 rq = _mm512_loadu_ps(rotated_q + i);
        acc = _mm512_fmadd_ps(rq, recon, acc);
    }

    float result = _mm512_reduce_add_ps(acc);
    result += ip_scalar(sign_bits, ex_code, rotated_q, i, d, 1, cb);
    return result;
}
#endif // __AVX512F__

#if defined(__AVX2__)
/// AVX2: 8 dims/iter, ex_bits == 1.
inline float ip_1exbit_avx2(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        float cb) {
    __m256 acc = _mm256_setzero_ps();
    const __m256 v_cb = _mm256_set1_ps(cb);
    const __m256 v_two = _mm256_set1_ps(2.0f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256i bit_pos = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i zero = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        uint8_t sb = sign_bits[i / 8];
        uint8_t eb = ex_code[i / 8];

        __m256i sb_cmp = _mm256_cmpgt_epi32(
                _mm256_and_si256(_mm256_set1_epi32(sb), bit_pos), zero);
        __m256 sb_f = _mm256_and_ps(_mm256_castsi256_ps(sb_cmp), v_one);

        __m256i eb_cmp = _mm256_cmpgt_epi32(
                _mm256_and_si256(_mm256_set1_epi32(eb), bit_pos), zero);
        __m256 eb_f = _mm256_and_ps(_mm256_castsi256_ps(eb_cmp), v_one);

        __m256 recon = _mm256_add_ps(_mm256_fmadd_ps(sb_f, v_two, eb_f), v_cb);
        __m256 rq = _mm256_loadu_ps(rotated_q + i);
        acc = _mm256_fmadd_ps(rq, recon, acc);
    }

    float result = hsum_avx2(acc);
    result += ip_scalar(sign_bits, ex_code, rotated_q, i, d, 1, cb);
    return result;
}
#endif // __AVX2__

/*********************************************************
 * Bit-plane decomposition kernels (ex_bits >= 2, BMI2 required).
 *
 * Decomposes the inner product as:
 *   ex_ip = (1 << ex_bits) * sign_dot
 *         + Σ_{b=0}^{ex_bits-1} (1 << b) * plane_dot_b
 *         + cb * total_q
 *
 * Each plane_dot_b is a float × bit-vector dot product, computed using
 * the same bit→mask→float conversion as the 1-bit kernel. PEXT
 * extracts each bit plane from the packed ex_code in one instruction
 * per 8 dimensions.
 *********************************************************/

#if defined(__AVX2__) && defined(__BMI2__)
/// AVX2 + BMI2 bit-plane decomposition: 8 dims/iter, ex_bits in [2, 7].
/// Caller must ensure ex_bits <= 7 (pext_masks[7] / v_weights[8]).
inline float ip_bitplane_avx2(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    __m256 acc = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256i bit_pos = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i zero = _mm256_setzero_si256();
    const __m256 v_cb = _mm256_set1_ps(cb);

    // Precompute PEXT masks and plane weights
    uint64_t pext_masks[7];
    __m256 v_weights[8];
    for (size_t b = 0; b < ex_bits; b++) {
        uint64_t m = 0;
        for (int j = 0; j < 8; j++) {
            m |= (1ULL << (b + j * ex_bits));
        }
        pext_masks[b] = m;
        v_weights[b] = _mm256_set1_ps(static_cast<float>(1u << b));
    }
    v_weights[ex_bits] = _mm256_set1_ps(static_cast<float>(1u << ex_bits));

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        // Sign bit → float via bit mask comparison
        __m256i sb_cmp = _mm256_cmpgt_epi32(
                _mm256_and_si256(_mm256_set1_epi32(sign_bits[i / 8]), bit_pos),
                zero);
        __m256 recon = _mm256_mul_ps(
                _mm256_and_ps(_mm256_castsi256_ps(sb_cmp), v_one),
                v_weights[ex_bits]);

        // Load packed ex_code for 8 dims (8 × ex_bits bits = ex_bits bytes)
        uint64_t ex64 = 0;
        memcpy(&ex64, ex_code + (i / 8) * ex_bits, sizeof(uint64_t));

        // Extract each bit plane via PEXT → bit mask → float
        for (size_t b = 0; b < ex_bits; b++) {
            auto plane = static_cast<uint8_t>(_pext_u64(ex64, pext_masks[b]));
            __m256i p_cmp = _mm256_cmpgt_epi32(
                    _mm256_and_si256(_mm256_set1_epi32(plane), bit_pos), zero);
            __m256 p_f = _mm256_and_ps(_mm256_castsi256_ps(p_cmp), v_one);
            recon = _mm256_fmadd_ps(p_f, v_weights[b], recon);
        }

        __m256 rq = _mm256_loadu_ps(rotated_q + i);
        acc = _mm256_fmadd_ps(rq, _mm256_add_ps(recon, v_cb), acc);
    }

    float result = hsum_avx2(acc);
    result += ip_scalar(sign_bits, ex_code, rotated_q, i, d, ex_bits, cb);
    return result;
}
#endif // __AVX2__ && __BMI2__

#endif // x86_64

/**
 * Dispatch to the best available kernel for the given ex_bits.
 *
 * Routing (compile-time):
 *   ex_bits == 1:  specialized 1-bit kernel (AVX-512 > AVX2 > scalar)
 *   ex_bits >= 2:  bit-plane decomposition (AVX2+BMI2 > scalar)
 *
 * @param sign_bits  packed sign bits (1 bit/dim, standard byte packing)
 * @param ex_code    packed extra-bit codes (ex_bits bits/dim)
 * @param rotated_q  rotated query vector (float[d])
 * @param d          dimensionality
 * @param ex_bits    number of extra bits per dimension (nb_bits - 1)
 * @param cb         constant bias: -(2^ex_bits - 0.5)
 * @return           inner product value
 */
inline float compute_inner_product(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    if (ex_bits == 1) {
#if defined(__AVX512F__)
        return ip_1exbit_avx512(sign_bits, ex_code, rotated_q, d, cb);
#elif defined(__AVX2__)
        return ip_1exbit_avx2(sign_bits, ex_code, rotated_q, d, cb);
#else
        return ip_scalar(sign_bits, ex_code, rotated_q, 0, d, 1, cb);
#endif
    }

#if defined(__AVX2__) && defined(__BMI2__)
    if (ex_bits <= 7) {
        return ip_bitplane_avx2(sign_bits, ex_code, rotated_q, d, ex_bits, cb);
    }
#endif
    return ip_scalar(sign_bits, ex_code, rotated_q, 0, d, ex_bits, cb);
}

} // namespace faiss::rabitq::multibit
