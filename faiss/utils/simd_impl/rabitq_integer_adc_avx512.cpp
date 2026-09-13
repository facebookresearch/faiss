/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_integer_adc.h>

#include <immintrin.h>

#include <algorithm>

namespace faiss::rabitq_integer_adc {

namespace {

// Keep each int32 lane far from overflow before reducing into int64. A lane
// receives one sum of four byte products per 64 input dimensions.
constexpr size_t kDotChunk = 4096;

inline int64_t reduce(__m512i accumulator) {
    return _mm512_reduce_add_epi32(accumulator);
}

inline int64_t biased_tail_product(int8_t query, int8_t level) {
    // VPDPBUSD multiplies unsigned bytes by signed bytes. Flipping the sign
    // bit maps a signed level l to the unsigned value l + 128. The caller's
    // query_correction subtracts 128 * sum(query) once from the full dot.
    const uint8_t biased_level = uint8_t(level) ^ uint8_t(0x80);
    return int64_t(query) * int64_t(biased_level);
}

} // namespace

int64_t dot_product_avx512_vnni(
        const int8_t* query,
        const int8_t* levels,
        size_t d,
        int64_t query_correction) {
    const __m512i sign_bit = _mm512_set1_epi32(0x80808080);
    int64_t result = query_correction;
    size_t j = 0;
    while (j + 64 <= d) {
        const size_t end = std::min(d - (d - j) % 64, j + kDotChunk);
        __m512i accumulator = _mm512_setzero_si512();
        for (; j + 64 <= end; j += 64) {
            const __m512i q = _mm512_loadu_si512(query + j);
            const __m512i biased_levels =
                    _mm512_xor_si512(_mm512_loadu_si512(levels + j), sign_bit);
            accumulator = _mm512_dpbusd_epi32(accumulator, biased_levels, q);
        }
        result += reduce(accumulator);
    }
    for (; j < d; j++) {
        result += biased_tail_product(query[j], levels[j]);
    }
    return result;
}

void dot_product_batch_4_avx512_vnni(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t query_correction,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3) {
    const __m512i sign_bit = _mm512_set1_epi32(0x80808080);
    dot0 = dot1 = dot2 = dot3 = query_correction;
    size_t j = 0;
    while (j + 64 <= d) {
        const size_t end = std::min(d - (d - j) % 64, j + kDotChunk);
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        for (; j + 64 <= end; j += 64) {
            const __m512i q = _mm512_loadu_si512(query + j);
            const __m512i level0 =
                    _mm512_xor_si512(_mm512_loadu_si512(levels0 + j), sign_bit);
            const __m512i level1 =
                    _mm512_xor_si512(_mm512_loadu_si512(levels1 + j), sign_bit);
            const __m512i level2 =
                    _mm512_xor_si512(_mm512_loadu_si512(levels2 + j), sign_bit);
            const __m512i level3 =
                    _mm512_xor_si512(_mm512_loadu_si512(levels3 + j), sign_bit);
            acc0 = _mm512_dpbusd_epi32(acc0, level0, q);
            acc1 = _mm512_dpbusd_epi32(acc1, level1, q);
            acc2 = _mm512_dpbusd_epi32(acc2, level2, q);
            acc3 = _mm512_dpbusd_epi32(acc3, level3, q);
        }
        dot0 += reduce(acc0);
        dot1 += reduce(acc1);
        dot2 += reduce(acc2);
        dot3 += reduce(acc3);
    }
    for (; j < d; j++) {
        dot0 += biased_tail_product(query[j], levels0[j]);
        dot1 += biased_tail_product(query[j], levels1[j]);
        dot2 += biased_tail_product(query[j], levels2[j]);
        dot3 += biased_tail_product(query[j], levels3[j]);
    }
}

void dot_product_batch_8_avx512_vnni(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t query_correction,
        int64_t dots[8]) {
    const __m512i sign_bit = _mm512_set1_epi32(0x80808080);
    for (size_t k = 0; k < 8; ++k) {
        dots[k] = query_correction;
    }
    size_t j = 0;
    while (j + 64 <= d) {
        const size_t end = std::min(d - (d - j) % 64, j + kDotChunk);
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        __m512i acc4 = _mm512_setzero_si512();
        __m512i acc5 = _mm512_setzero_si512();
        __m512i acc6 = _mm512_setzero_si512();
        __m512i acc7 = _mm512_setzero_si512();
        for (; j + 64 <= end; j += 64) {
            const __m512i q = _mm512_loadu_si512(query + j);
            acc0 = _mm512_dpbusd_epi32(
                    acc0,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[0] + j), sign_bit),
                    q);
            acc1 = _mm512_dpbusd_epi32(
                    acc1,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[1] + j), sign_bit),
                    q);
            acc2 = _mm512_dpbusd_epi32(
                    acc2,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[2] + j), sign_bit),
                    q);
            acc3 = _mm512_dpbusd_epi32(
                    acc3,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[3] + j), sign_bit),
                    q);
            acc4 = _mm512_dpbusd_epi32(
                    acc4,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[4] + j), sign_bit),
                    q);
            acc5 = _mm512_dpbusd_epi32(
                    acc5,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[5] + j), sign_bit),
                    q);
            acc6 = _mm512_dpbusd_epi32(
                    acc6,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[6] + j), sign_bit),
                    q);
            acc7 = _mm512_dpbusd_epi32(
                    acc7,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[7] + j), sign_bit),
                    q);
        }
        dots[0] += reduce(acc0);
        dots[1] += reduce(acc1);
        dots[2] += reduce(acc2);
        dots[3] += reduce(acc3);
        dots[4] += reduce(acc4);
        dots[5] += reduce(acc5);
        dots[6] += reduce(acc6);
        dots[7] += reduce(acc7);
    }
    for (; j < d; ++j) {
        for (size_t k = 0; k < 8; ++k) {
            dots[k] += biased_tail_product(query[j], levels[k][j]);
        }
    }
}

} // namespace faiss::rabitq_integer_adc
