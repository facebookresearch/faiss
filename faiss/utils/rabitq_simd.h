/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

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
