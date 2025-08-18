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
#endif

namespace faiss {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
        defined(_M_IX86)
/**
 * Returns the lookup table for AVX512 popcount operations.
 * This table is used for lookup-based popcount implementation.
 *
 * @return Lookup table as __m512i register
 */
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

/**
 * Performs lookup-based popcount on AVX512 registers.
 *
 * @param v_and Input vector to count bits in
 * @return Vector with popcount results
 */
inline __m512i popcount_lookup_avx512(__m512i v_and) {
    const __m512i lookup = get_lookup_512();
    const __m512i low_mask = _mm512_set1_epi8(0x0f);

    const __m512i lo = _mm512_and_si512(v_and, low_mask);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v_and, 4), low_mask);
    const __m512i popcnt1 = _mm512_shuffle_epi8(lookup, lo);
    const __m512i popcnt2 = _mm512_shuffle_epi8(lookup, hi);
    return _mm512_add_epi8(popcnt1, popcnt2);
}

/**
 * Performs lookup-based popcount on AVX2 registers.
 *
 * @param v_and Input vector to count bits in
 * @return Vector with popcount results
 */
inline __m256i popcount_lookup_avx2(__m256i v_and) {
    const __m256i lookup = get_lookup_256();
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    const __m256i lo = _mm256_and_si256(v_and, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v_and, 4), low_mask);
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    return _mm256_add_epi8(popcnt1, popcnt2);
}
#endif

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)

/**
 * AVX512-optimized version of dot product computation between query and binary
 * data. Requires AVX512F and AVX512VPOPCNTDQ instruction sets.
 *
 * @param query          Pointer to rearranged rotated query data
 * @param binary_data    Pointer to binary data
 * @param d              Dimension
 * @param qb             Number of quantization bits
 * @return               Dot product result as float
 */
inline float rabitq_dp_popcnt_avx512(
        const uint8_t* query,
        const uint8_t* binary_data,
        size_t d,
        size_t qb) {
    __m512i sum_512 = _mm512_setzero_si512();

    const size_t di_8b = (d + 7) / 8;

    const size_t d_512 = (d / 512) * 512;
    const size_t d_256 = (d / 256) * 256;
    const size_t d_128 = (d / 128) * 128;

    for (size_t i = 0; i < d_512; i += 512) {
        __m512i v_x = _mm512_loadu_si512((const __m512i*)(binary_data + i / 8));
        for (size_t j = 0; j < qb; j++) {
            __m512i v_q = _mm512_loadu_si512(
                    (const __m512i*)(query + j * di_8b + i / 8));
            __m512i v_and = _mm512_and_si512(v_q, v_x);
            __m512i v_popcnt = _mm512_popcnt_epi32(v_and);
            sum_512 = _mm512_add_epi32(sum_512, _mm512_slli_epi32(v_popcnt, j));
        }
    }

    __m256i sum_256 = _mm256_add_epi32(
            _mm512_extracti32x8_epi32(sum_512, 0),
            _mm512_extracti32x8_epi32(sum_512, 1));

    if (d_256 != d_512) {
        __m256i v_x =
                _mm256_loadu_si256((const __m256i*)(binary_data + d_512 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m256i v_q = _mm256_loadu_si256(
                    (const __m256i*)(query + j * di_8b + d_512 / 8));
            __m256i v_and = _mm256_and_si256(v_q, v_x);
            __m256i v_popcnt = _mm256_popcnt_epi32(v_and);
            sum_256 = _mm256_add_epi32(sum_256, _mm256_slli_epi32(v_popcnt, j));
        }
    }

    __m128i sum_128 = _mm_add_epi32(
            _mm256_extracti32x4_epi32(sum_256, 0),
            _mm256_extracti32x4_epi32(sum_256, 1));

    if (d_128 != d_256) {
        __m128i v_x =
                _mm_loadu_si128((const __m128i*)(binary_data + d_256 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    (const __m128i*)(query + j * di_8b + d_256 / 8));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            __m128i v_popcnt = _mm_popcnt_epi32(v_and);
            sum_128 = _mm_add_epi32(sum_128, _mm_slli_epi32(v_popcnt, j));
        }
    }

    if (d != d_128) {
        const size_t leftovers = d - d_128;
        const __mmask16 mask = (1 << ((leftovers + 7) / 8)) - 1;

        __m128i v_x = _mm_maskz_loadu_epi8(
                mask, (const __m128i*)(binary_data + d_128 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m128i v_q = _mm_maskz_loadu_epi8(
                    mask, (const __m128i*)(query + j * di_8b + d_128 / 8));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            __m128i v_popcnt = _mm_popcnt_epi32(v_and);
            sum_128 = _mm_add_epi32(sum_128, _mm_slli_epi32(v_popcnt, j));
        }
    }

    int sum_64le = 0;
    sum_64le += _mm_extract_epi32(sum_128, 0);
    sum_64le += _mm_extract_epi32(sum_128, 1);
    sum_64le += _mm_extract_epi32(sum_128, 2);
    sum_64le += _mm_extract_epi32(sum_128, 3);

    return static_cast<float>(sum_64le);
}
#endif

#if defined(__AVX512F__) && !defined(__AVX512VPOPCNTDQ__)
/**
 * AVX512-optimized version of dot product computation between query and binary
 * data. Uses AVX512F instructions but does not require AVX512VPOPCNTDQ.
 *
 * @param query          Pointer to rearranged rotated query data
 * @param binary_data    Pointer to binary data
 * @param d              Dimension
 * @param qb             Number of quantization bits
 * @return               Dot product result as float
 */
inline float rabitq_dp_popcnt_avx512_fallback(
        const uint8_t* query,
        const uint8_t* binary_data,
        size_t d,
        size_t qb) {
    const size_t di_8b = (d + 7) / 8;
    const size_t d_512 = (d / 512) * 512;
    const size_t d_256 = (d / 256) * 256;
    const size_t d_128 = (d / 128) * 128;

    // Use the lookup-based popcount helper function

    __m512i sum_512 = _mm512_setzero_si512();

    // Process 512 bits (64 bytes) at a time using lookup-based popcount
    for (size_t i = 0; i < d_512; i += 512) {
        __m512i v_x = _mm512_loadu_si512((const __m512i*)(binary_data + i / 8));
        for (size_t j = 0; j < qb; j++) {
            __m512i v_q = _mm512_loadu_si512(
                    (const __m512i*)(query + j * di_8b + i / 8));
            __m512i v_and = _mm512_and_si512(v_q, v_x);

            // Use the popcount_lookup_avx512 helper function
            __m512i v_popcnt = popcount_lookup_avx512(v_and);

            // Sum bytes to 32-bit integers
            __m512i v_sad = _mm512_sad_epu8(v_popcnt, _mm512_setzero_si512());

            // Shift by j and add to sum
            __m512i v_shifted = _mm512_slli_epi64(v_sad, j);
            sum_512 = _mm512_add_epi64(sum_512, v_shifted);
        }
    }

    // Handle 256-bit section if needed
    __m256i sum_256 = _mm256_setzero_si256();
    if (d_256 != d_512) {
        __m256i v_x =
                _mm256_loadu_si256((const __m256i*)(binary_data + d_512 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m256i v_q = _mm256_loadu_si256(
                    (const __m256i*)(query + j * di_8b + d_512 / 8));
            __m256i v_and = _mm256_and_si256(v_q, v_x);

            // Use the popcount_lookup_avx2 helper function
            __m256i v_popcnt = popcount_lookup_avx2(v_and);

            // Sum bytes to 64-bit integers
            __m256i v_sad = _mm256_sad_epu8(v_popcnt, _mm256_setzero_si256());

            // Shift by j and add to sum
            __m256i v_shifted = _mm256_slli_epi64(v_sad, j);
            sum_256 = _mm256_add_epi64(sum_256, v_shifted);
        }
    }

    // Handle 128-bit section and leftovers
    __m128i sum_128 = _mm_setzero_si128();
    if (d_128 != d_256) {
        __m128i v_x =
                _mm_loadu_si128((const __m128i*)(binary_data + d_256 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    (const __m128i*)(query + j * di_8b + d_256 / 8));
            __m128i v_and = _mm_and_si128(v_q, v_x);

            // Scalar popcount for each 64-bit lane
            uint64_t lane0 = _mm_extract_epi64(v_and, 0);
            uint64_t lane1 = _mm_extract_epi64(v_and, 1);
            uint64_t pop0 = __builtin_popcountll(lane0) << j;
            uint64_t pop1 = __builtin_popcountll(lane1) << j;
            sum_128 = _mm_add_epi64(sum_128, _mm_set_epi64x(pop1, pop0));
        }
    }

    // Handle remaining bytes (less than 16)
    uint64_t sum_leftover = 0;
    size_t d_leftover = d - d_128;
    if (d_leftover > 0) {
        for (size_t j = 0; j < qb; j++) {
            for (size_t k = 0; k < (d_leftover + 7) / 8; ++k) {
                uint8_t qv = query[j * di_8b + d_128 / 8 + k];
                uint8_t yv = binary_data[d_128 / 8 + k];
                sum_leftover += (__builtin_popcount(qv & yv) << j);
            }
        }
    }

    // Horizontal sum of all lanes
    uint64_t sum = 0;

    // Sum from 512-bit registers
    alignas(64) uint64_t lanes512[8];
    _mm512_store_si512((__m512i*)lanes512, sum_512);
    for (int i = 0; i < 8; ++i) {
        sum += lanes512[i];
    }

    // Sum from 256-bit registers
    alignas(32) uint64_t lanes256[4];
    _mm256_store_si256((__m256i*)lanes256, sum_256);
    for (int i = 0; i < 4; ++i) {
        sum += lanes256[i];
    }

    // Sum from 128-bit registers
    alignas(16) uint64_t lanes128[2];
    _mm_store_si128((__m128i*)lanes128, sum_128);
    sum += lanes128[0] + lanes128[1];

    // Add leftovers
    sum += sum_leftover;

    return static_cast<float>(sum);
}
#endif

#ifdef __AVX2__

/**
 * AVX2-optimized version of dot product computation between query and binary
 * data.
 *
 * @param query          Pointer to rearranged rotated query data
 * @param binary_data    Pointer to binary data
 * @param d              Dimension
 * @param qb             Number of quantization bits
 * @return               Dot product result as float
 */

inline float rabitq_dp_popcnt_avx2(
        const uint8_t* query,
        const uint8_t* binary_data,
        size_t d,
        size_t qb) {
    const size_t di_8b = (d + 7) / 8;
    const size_t d_256 = (d / 256) * 256;
    const size_t d_128 = (d / 128) * 128;

    // Use the lookup-based popcount helper function

    __m256i sum_256 = _mm256_setzero_si256();

    // Process 256 bits (32 bytes) at a time using lookup-based popcount
    for (size_t i = 0; i < d_256; i += 256) {
        __m256i v_x = _mm256_loadu_si256((const __m256i*)(binary_data + i / 8));
        for (size_t j = 0; j < qb; j++) {
            __m256i v_q = _mm256_loadu_si256(
                    (const __m256i*)(query + j * di_8b + i / 8));
            __m256i v_and = _mm256_and_si256(v_q, v_x);

            // Use the popcount_lookup_avx2 helper function
            __m256i v_popcnt = popcount_lookup_avx2(v_and);

            // Convert byte counts to 64-bit lanes and shift by j
            __m256i v_sad = _mm256_sad_epu8(v_popcnt, _mm256_setzero_si256());
            __m256i v_shifted = _mm256_slli_epi64(v_sad, static_cast<int>(j));
            sum_256 = _mm256_add_epi64(sum_256, v_shifted);
        }
    }

    // Handle leftovers with 128-bit SIMD
    __m128i sum_128 = _mm_setzero_si128();
    if (d_128 != d_256) {
        __m128i v_x =
                _mm_loadu_si128((const __m128i*)(binary_data + d_256 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    (const __m128i*)(query + j * di_8b + d_256 / 8));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            // Scalar popcount for each 64-bit lane
            uint64_t lane0 = _mm_extract_epi64(v_and, 0);
            uint64_t lane1 = _mm_extract_epi64(v_and, 1);
            uint64_t pop0 = __builtin_popcountll(lane0) << j;
            uint64_t pop1 = __builtin_popcountll(lane1) << j;
            sum_128 = _mm_add_epi64(sum_128, _mm_set_epi64x(pop1, pop0));
        }
    }

    // Handle remaining bytes (less than 16)
    uint64_t sum_leftover = 0;
    size_t d_leftover = d - d_128;
    if (d_leftover > 0) {
        for (size_t j = 0; j < qb; j++) {
            for (size_t k = 0; k < (d_leftover + 7) / 8; ++k) {
                uint8_t qv = query[j * di_8b + d_128 / 8 + k];
                uint8_t yv = binary_data[d_128 / 8 + k];
                sum_leftover += (__builtin_popcount(qv & yv) << j);
            }
        }
    }

    // Horizontal sum of all lanes
    uint64_t sum = 0;
    // sum_256: 4 lanes of 64 bits
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256((__m256i*)lanes, sum_256);
    for (int i = 0; i < 4; ++i) {
        sum += lanes[i];
    }
    // sum_128: 2 lanes of 64 bits
    alignas(16) uint64_t lanes128[2];
    _mm_store_si128((__m128i*)lanes128, sum_128);
    sum += lanes128[0] + lanes128[1];
    // leftovers
    sum += sum_leftover;

    return static_cast<float>(sum);
}
#endif

/**
 * Compute dot product between query and binary data using popcount operations.
 *
 * @param query          Pointer to rearranged rotated query data
 * @param binary_data    Pointer to binary data
 * @param d              Dimension
 * @param qb             Number of quantization bits
 * @return               Dot product result as float
 */
inline float rabitq_dp_popcnt(
        const uint8_t* query,
        const uint8_t* binary_data,
        size_t d,
        size_t qb) {
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    return rabitq_dp_popcnt_avx512(query, binary_data, d, qb);
#elif defined(__AVX512F__)
    return rabitq_dp_popcnt_avx512_fallback(query, binary_data, d, qb);
#elif defined(__AVX2__)
    return rabitq_dp_popcnt_avx2(query, binary_data, d, qb);
#else
    const size_t di_8b = (d + 7) / 8;
    const size_t di_64b = (di_8b / 8) * 8;

    uint64_t dot_qo = 0;
    for (size_t j = 0; j < qb; j++) {
        const uint8_t* query_j = query + j * di_8b;

        // process 64-bit popcounts
        uint64_t count_dot = 0;
        for (size_t i = 0; i < di_64b; i += 8) {
            const auto qv = *(const uint64_t*)(query_j + i);
            const auto yv = *(const uint64_t*)(binary_data + i);
            count_dot += __builtin_popcountll(qv & yv);
        }

        // process leftovers
        for (size_t i = di_64b; i < di_8b; i++) {
            const auto qv = *(query_j + i);
            const auto yv = *(binary_data + i);
            count_dot += __builtin_popcount(qv & yv);
        }

        dot_qo += (count_dot << j);
    }

    return static_cast<float>(dot_qo);
#endif
}

} // namespace faiss
