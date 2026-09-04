/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/utils/rabitq_simd.h>
#include <immintrin.h>
#include <limits>

namespace faiss::rabitq {

namespace {

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
#endif
}

// AVX2 helpers needed for AVX512 fallback paths (compute_inner_product)
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

inline __m256i popcount_256(__m256i v) {
    const __m256i lookup = get_lookup_256();
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    const __m256i lo = _mm256_and_si256(v, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    const __m256i popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
    const __m256i popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);
    return _mm256_sad_epu8(_mm256_setzero_si256(), popcnt);
}

inline uint64_t reduce_add_256(__m256i v) {
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256((__m256i*)lanes, v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

inline __m128i popcount_128(__m128i v) {
    uint64_t lane0 = _mm_extract_epi64(v, 0);
    uint64_t lane1 = _mm_extract_epi64(v, 1);
    uint64_t pop0 = popcount64(lane0);
    uint64_t pop1 = popcount64(lane1);
    return _mm_set_epi64x(pop1, pop0);
}

inline uint64_t reduce_add_128(__m128i v) {
    alignas(16) uint64_t lanes[2];
    _mm_store_si128((__m128i*)lanes, v);
    return lanes[0] + lanes[1];
}

inline __m512i round_nonnegative_ps_to_i32(__m512 x) {
    return _mm512_cvttps_epi32(_mm512_add_ps(x, _mm512_set1_ps(0.5f)));
}

inline void store_i32_as_u8_16(__m512i values, uint8_t* out) {
    const __m128i packed = _mm512_cvtusepi32_epi8(values);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), packed);
}

} // namespace

template <>
void lut_minmax_16<SIMDLevel::AVX512>(const float* tab, float& mn, float& mx) {
    const __m512 v = _mm512_loadu_ps(tab);
    mn = _mm512_reduce_min_ps(v);
    mx = _mm512_reduce_max_ps(v);
}

template <>
void minmax_values<SIMDLevel::AVX512>(
        const float* values,
        size_t n,
        float& mn,
        float& mx) {
    if (n == 0) {
        return;
    }

    size_t i = 0;
    __m512 mn_vec = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 mx_vec = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_loadu_ps(values + i);
        mn_vec = _mm512_min_ps(mn_vec, v);
        mx_vec = _mm512_max_ps(mx_vec, v);
    }

    if (i < n) {
        const __mmask16 mask =
                static_cast<__mmask16>((uint32_t(1) << (n - i)) - 1);
        const __m512 v = _mm512_maskz_loadu_ps(mask, values + i);
        mn_vec = _mm512_mask_min_ps(mn_vec, mask, mn_vec, v);
        mx_vec = _mm512_mask_max_ps(mx_vec, mask, mx_vec, v);
    }

    mn = _mm512_reduce_min_ps(mn_vec);
    mx = _mm512_reduce_max_ps(mx_vec);
}

template <>
void lut_quantize_16_to_uint8<SIMDLevel::AVX512>(
        const float* tab,
        float mn,
        float a,
        uint8_t* out) {
    const __m512 values = _mm512_loadu_ps(tab);
    const __m512 a_vec = _mm512_set1_ps(a);
    const __m512 scaled =
            _mm512_fmsub_ps(values, a_vec, _mm512_set1_ps(mn * a));
    __m512i rounded = round_nonnegative_ps_to_i32(scaled);
    rounded = _mm512_max_epi32(rounded, _mm512_setzero_si512());
    store_i32_as_u8_16(rounded, out);
}

template <>
void quantize_query_values<SIMDLevel::AVX512>(
        const float* rq,
        size_t d,
        float v_min,
        float inv_delta,
        uint8_t max_code,
        bool centered,
        uint8_t* rqq,
        size_t& sum_qq,
        int64_t& sum2_signed_odd_int) {
    const __m512 inv_delta_vec = _mm512_set1_ps(inv_delta);
    const __m512 v_min_times_inv_delta_vec = _mm512_set1_ps(v_min * inv_delta);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 max_code_ps = _mm512_set1_ps(static_cast<float>(max_code));
    const __m512i max_code_i32 = _mm512_set1_epi32(max_code);
    const __m512i two = _mm512_set1_epi32(2);

    size_t i = 0;
    if (centered) {
        __m512i sum_acc_lo = _mm512_setzero_si512();
        __m512i sum_acc_hi = _mm512_setzero_si512();
        __m512i sq_acc_lo = _mm512_setzero_si512();
        __m512i sq_acc_hi = _mm512_setzero_si512();
        for (; i + 16 <= d; i += 16) {
            const __m512 values = _mm512_loadu_ps(rq + i);
            __m512 scaled = _mm512_fmsub_ps(
                    values, inv_delta_vec, v_min_times_inv_delta_vec);
            scaled = _mm512_min_ps(_mm512_max_ps(scaled, zero), max_code_ps);
            __m512i rounded = round_nonnegative_ps_to_i32(scaled);

            sum_acc_lo = _mm512_add_epi64(
                    sum_acc_lo,
                    _mm512_cvtepi32_epi64(_mm512_castsi512_si256(rounded)));
            sum_acc_hi = _mm512_add_epi64(
                    sum_acc_hi,
                    _mm512_cvtepi32_epi64(
                            _mm512_extracti64x4_epi64(rounded, 1)));
            const __m512i signed_odd = _mm512_sub_epi32(
                    _mm512_mullo_epi32(rounded, two), max_code_i32);
            const __m512i signed_odd_sqr =
                    _mm512_mullo_epi32(signed_odd, signed_odd);
            sq_acc_lo = _mm512_add_epi64(
                    sq_acc_lo,
                    _mm512_cvtepi32_epi64(
                            _mm512_castsi512_si256(signed_odd_sqr)));
            sq_acc_hi = _mm512_add_epi64(
                    sq_acc_hi,
                    _mm512_cvtepi32_epi64(
                            _mm512_extracti64x4_epi64(signed_odd_sqr, 1)));
            store_i32_as_u8_16(rounded, rqq + i);
        }
        sum_qq += static_cast<uint64_t>(
                _mm512_reduce_add_epi64(sum_acc_lo) +
                _mm512_reduce_add_epi64(sum_acc_hi));
        sum2_signed_odd_int += _mm512_reduce_add_epi64(sq_acc_lo);
        sum2_signed_odd_int += _mm512_reduce_add_epi64(sq_acc_hi);
    } else {
        __m512i sum_acc_lo = _mm512_setzero_si512();
        __m512i sum_acc_hi = _mm512_setzero_si512();
        for (; i + 16 <= d; i += 16) {
            const __m512 values = _mm512_loadu_ps(rq + i);
            __m512 scaled = _mm512_fmsub_ps(
                    values, inv_delta_vec, v_min_times_inv_delta_vec);
            scaled = _mm512_min_ps(_mm512_max_ps(scaled, zero), max_code_ps);
            __m512i rounded = round_nonnegative_ps_to_i32(scaled);

            sum_acc_lo = _mm512_add_epi64(
                    sum_acc_lo,
                    _mm512_cvtepi32_epi64(_mm512_castsi512_si256(rounded)));
            sum_acc_hi = _mm512_add_epi64(
                    sum_acc_hi,
                    _mm512_cvtepi32_epi64(
                            _mm512_extracti64x4_epi64(rounded, 1)));
            store_i32_as_u8_16(rounded, rqq + i);
        }
        sum_qq += static_cast<uint64_t>(
                _mm512_reduce_add_epi64(sum_acc_lo) +
                _mm512_reduce_add_epi64(sum_acc_hi));
    }

    for (; i < d; i++) {
        const uint8_t v_qq = round_clamped_byte_scalar(
                (rq[i] - v_min) * inv_delta, max_code);
        rqq[i] = v_qq;
        sum_qq += v_qq;

        if (centered) {
            const int64_t signed_odd_int = int64_t(v_qq) * 2 - max_code;
            sum2_signed_odd_int += signed_odd_int * signed_odd_int;
        }
    }
}

template <>
uint64_t bitwise_and_dot_product<SIMDLevel::AVX512>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;
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
BitwiseAndDotProductResult bitwise_and_dot_product_with_popcount<
        SIMDLevel::AVX512>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t dot_product = 0;
    uint64_t popcount_sum = 0;
    size_t offset = 0;
    if (size_t step = 512 / 8; offset + step <= size) {
        __m512i dot_512 = _mm512_setzero_si512();
        __m512i pop_512 = _mm512_setzero_si512();
        for (; offset + step <= size; offset += step) {
            __m512i v_x = _mm512_loadu_si512((const __m512i*)(data + offset));
            pop_512 = _mm512_add_epi64(pop_512, popcount_512(v_x));
            for (int j = 0; j < qb; j++) {
                __m512i v_q = _mm512_loadu_si512(
                        (const __m512i*)(query + j * size + offset));
                __m512i v_and = _mm512_and_si512(v_q, v_x);
                __m512i v_popcnt = popcount_512(v_and);
                __m512i v_shifted = _mm512_slli_epi64(v_popcnt, j);
                dot_512 = _mm512_add_epi64(dot_512, v_shifted);
            }
        }
        dot_product += _mm512_reduce_add_epi64(dot_512);
        popcount_sum += _mm512_reduce_add_epi64(pop_512);
    }
    if (size_t step = 256 / 8; offset + step <= size) {
        __m256i dot_256 = _mm256_setzero_si256();
        __m256i pop_256 = _mm256_setzero_si256();
        for (; offset + step <= size; offset += step) {
            __m256i v_x = _mm256_loadu_si256((const __m256i*)(data + offset));
            pop_256 = _mm256_add_epi64(pop_256, popcount_256(v_x));
            for (int j = 0; j < qb; j++) {
                __m256i v_q = _mm256_loadu_si256(
                        (const __m256i*)(query + j * size + offset));
                __m256i v_and = _mm256_and_si256(v_q, v_x);
                __m256i v_popcnt = popcount_256(v_and);
                __m256i v_shifted = _mm256_slli_epi64(v_popcnt, j);
                dot_256 = _mm256_add_epi64(dot_256, v_shifted);
            }
        }
        dot_product += reduce_add_256(dot_256);
        popcount_sum += reduce_add_256(pop_256);
    }
    __m128i dot_128 = _mm_setzero_si128();
    __m128i pop_128 = _mm_setzero_si128();
    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128((const __m128i*)(data + offset));
        pop_128 = _mm_add_epi64(pop_128, popcount_128(v_x));
        for (int j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    (const __m128i*)(query + j * size + offset));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            __m128i v_popcnt = popcount_128(v_and);
            __m128i v_shifted = _mm_slli_epi64(v_popcnt, j);
            dot_128 = _mm_add_epi64(dot_128, v_shifted);
        }
    }
    dot_product += reduce_add_128(dot_128);
    popcount_sum += reduce_add_128(pop_128);
    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *(const uint64_t*)(data + offset);
        popcount_sum += popcount64(yv);
        for (int j = 0; j < qb; j++) {
            const auto qv = *(const uint64_t*)(query + j * size + offset);
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
uint64_t bitwise_xor_dot_product<SIMDLevel::AVX512>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;
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
uint64_t popcount<SIMDLevel::AVX512>(const uint8_t* data, size_t size) {
    uint64_t sum = 0;
    size_t offset = 0;
    if (offset + 512 / 8 <= size) {
        __m512i sum_512 = _mm512_setzero_si512();
        for (size_t end; (end = offset + 512 / 8) <= size; offset = end) {
            __m512i v_x = _mm512_loadu_si512((const __m512i*)(data + offset));
            __m512i v_popcnt = popcount_512(v_x);
            sum_512 = _mm512_add_epi64(sum_512, v_popcnt);
        }
        sum += _mm512_reduce_add_epi64(sum_512);
    }
    if (offset + 256 / 8 <= size) {
        __m256i sum_256 = _mm256_setzero_si256();
        for (size_t end; (end = offset + 256 / 8) <= size; offset = end) {
            __m256i v_x = _mm256_loadu_si256((const __m256i*)(data + offset));
            __m256i v_popcnt = popcount_256(v_x);
            sum_256 = _mm256_add_epi64(sum_256, v_popcnt);
        }
        sum += reduce_add_256(sum_256);
    }
    __m128i sum_128 = _mm_setzero_si128();
    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128((const __m128i*)(data + offset));
        sum_128 = _mm_add_epi64(sum_128, popcount_128(v_x));
    }
    sum += reduce_add_128(sum_128);
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

template <>
float selected_float_sum<SIMDLevel::AVX512>(
        const uint8_t* sign_bits,
        const float* values,
        size_t d) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        uint16_t packed = 0;
        memcpy(&packed, sign_bits + i / 8, sizeof(packed));
        sum = _mm512_add_ps(
                sum,
                _mm512_maskz_loadu_ps(
                        static_cast<__mmask16>(packed), values + i));
    }
    float result = _mm512_reduce_add_ps(sum);
    result += selected_float_sum<SIMDLevel::NONE>(
            sign_bits + i / 8, values + i, d - i);
    return result;
}

template <>
void rearrange_bit_planes<SIMDLevel::AVX512>(
        const uint8_t* rotated_qq,
        size_t d,
        size_t qb,
        uint8_t* out) {
    const size_t offset = (d + 7) / 8;
    memset(out, 0, offset * qb);
    size_t idim = 0;
    for (; idim + 64 <= d; idim += 64) {
        __m512i vals = _mm512_loadu_si512((const __m512i*)(rotated_qq + idim));
        for (size_t iv = 0; iv < qb; iv++) {
            __m512i mask = _mm512_set1_epi8(static_cast<char>(1 << iv));
            __mmask64 bits = _mm512_test_epi8_mask(vals, mask);
            memcpy(&out[iv * offset + idim / 8], &bits, 8);
        }
    }
    for (; idim + 32 <= d; idim += 32) {
        __m256i vals = _mm256_loadu_si256((const __m256i*)(rotated_qq + idim));
        for (size_t iv = 0; iv < qb; iv++) {
            __m256i mask = _mm256_set1_epi8(static_cast<char>(1 << iv));
            __m256i bits =
                    _mm256_cmpeq_epi8(_mm256_and_si256(vals, mask), mask);
            uint32_t packed = static_cast<uint32_t>(_mm256_movemask_epi8(bits));
            memcpy(&out[iv * offset + idim / 8], &packed, 4);
        }
    }
    for (; idim < d; idim++) {
        for (size_t iv = 0; iv < qb; iv++) {
            const bool bit = ((rotated_qq[idim] & (1 << iv)) != 0);
            out[iv * offset + idim / 8] |= bit ? (1 << (idim % 8)) : 0;
        }
    }
}

} // namespace faiss::rabitq

namespace faiss::rabitq::multibit {

namespace {

template <size_t NBITS>
inline __m256i dense_decode_8_i32_avx512(const uint8_t* code) {
    static_assert(NBITS >= 2 && NBITS <= 8);
    uint64_t packed = 0;
    memcpy(&packed, code, NBITS);
    constexpr uint64_t mask = (uint64_t{1} << NBITS) - 1;
    return _mm256_setr_epi32(
            (packed >> (0 * NBITS)) & mask,
            (packed >> (1 * NBITS)) & mask,
            (packed >> (2 * NBITS)) & mask,
            (packed >> (3 * NBITS)) & mask,
            (packed >> (4 * NBITS)) & mask,
            (packed >> (5 * NBITS)) & mask,
            (packed >> (6 * NBITS)) & mask,
            (packed >> (7 * NBITS)) & mask);
}

template <size_t NBITS>
inline __m512 dense_decode_16_avx512(const uint8_t* code) {
    const __m256i lo = dense_decode_8_i32_avx512<NBITS>(code);
    const __m256i hi = dense_decode_8_i32_avx512<NBITS>(code + NBITS);
    const __m512i values =
            _mm512_inserti32x8(_mm512_castsi256_si512(lo), hi, 1);
    return _mm512_cvtepi32_ps(values);
}

inline __m512i dense_decode_3_64_u8_avx512(const uint8_t* code) {
    const __m256i shuf_0 = _mm256_setr_epi8(
            0,
            -1,
            0,
            1,
            1,
            -1,
            2,
            -1,
            3,
            -1,
            3,
            4,
            4,
            -1,
            5,
            -1,
            6,
            -1,
            6,
            7,
            7,
            -1,
            8,
            -1,
            9,
            -1,
            9,
            10,
            10,
            -1,
            11,
            -1);
    const __m256i shuf_1 = _mm256_setr_epi8(
            0,
            -1,
            1,
            -1,
            1,
            2,
            2,
            -1,
            3,
            -1,
            4,
            -1,
            4,
            5,
            5,
            -1,
            6,
            -1,
            7,
            -1,
            7,
            8,
            8,
            -1,
            9,
            -1,
            10,
            -1,
            10,
            11,
            11,
            -1);
    const __m256i shuf_2 = _mm256_setr_epi8(
            12,
            -1,
            12,
            13,
            13,
            -1,
            14,
            -1,
            15,
            -1,
            15,
            0,
            0,
            -1,
            1,
            -1,
            2,
            -1,
            2,
            3,
            3,
            -1,
            4,
            -1,
            5,
            -1,
            5,
            6,
            6,
            -1,
            7,
            -1);
    const __m256i shuf_3 = _mm256_setr_epi8(
            12,
            -1,
            13,
            -1,
            13,
            14,
            14,
            -1,
            15,
            -1,
            0,
            -1,
            0,
            1,
            1,
            -1,
            2,
            -1,
            3,
            -1,
            3,
            4,
            4,
            -1,
            5,
            -1,
            6,
            -1,
            6,
            7,
            7,
            -1);
    const __m512i shuf_02 =
            _mm512_inserti32x8(_mm512_castsi256_si512(shuf_0), shuf_2, 1);
    const __m512i shuf_13 =
            _mm512_inserti32x8(_mm512_castsi256_si512(shuf_1), shuf_3, 1);
    const __m256i shifts_left =
            _mm256_setr_epi16(5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3);
    const __m256i shifts_right =
            _mm256_setr_epi16(0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2);
    const __m512i v_shl = _mm512_inserti32x8(
            _mm512_castsi256_si512(shifts_left), shifts_left, 1);
    const __m512i v_shr = _mm512_inserti32x8(
            _mm512_castsi256_si512(shifts_right), shifts_right, 1);

    const __m128i raw_0 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(code));
    const __m128i raw_2_low =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + 16));
    const __m128i raw_2 = _mm_blend_epi16(raw_0, raw_2_low, 0x0f);
    const __m256i raw_01 =
            _mm256_inserti32x4(_mm256_castsi128_si256(raw_0), raw_0, 1);
    const __m256i raw_23 =
            _mm256_inserti32x4(_mm256_castsi128_si256(raw_2), raw_2, 1);
    const __m512i raw =
            _mm512_inserti32x8(_mm512_castsi256_si512(raw_01), raw_23, 1);
    const __m512i right =
            _mm512_srlv_epi16(_mm512_shuffle_epi8(raw, shuf_02), v_shr);
    const __m512i left =
            _mm512_sllv_epi16(_mm512_shuffle_epi8(raw, shuf_13), v_shl);
    return _mm512_and_si512(
            _mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaaULL, right, left),
            _mm512_set1_epi8(7));
}

inline __m512i dense_decode_4_64_u8_avx512(const uint8_t* code) {
    const __m256i packed =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code));
    const __m512i widened = _mm512_cvtepu8_epi16(packed);
    return _mm512_and_si512(
            _mm512_or_si512(
                    widened,
                    _mm512_slli_epi16(_mm512_srli_epi16(widened, 4), 8)),
            _mm512_set1_epi16(0x0f0f));
}

inline __m512i dense_decode_5_64_u8_avx512(const uint8_t* code) {
    const __m512i low = _mm512_zextsi256_si512(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code)));
    const __m128i high =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + 32));
    const __m512i raw = _mm512_inserti32x4(low, high, 2);
    const __m512i spread = _mm512_permutexvar_epi64(
            _mm512_setr_epi64(0, 1, 1, 2, 2, 3, 3, 4), raw);
    const __m512i shuf_a = _mm512_setr_epi64(
            0x04030302ff01ff00ULL,
            0x09080807ff06ff05ULL,
            0x06050504ff03ff02ULL,
            0x0b0a0a09ff08ff07ULL,
            0x08070706ff05ff04ULL,
            0x0d0c0c0bff0aff09ULL,
            0x0a090908ff07ff06ULL,
            0x0f0e0e0dff0cff0bULL);
    const __m512i shuf_b = _mm512_setr_epi64(
            0xff04ff0302010100ULL,
            0xff09ff0807060605ULL,
            0xff06ff0504030302ULL,
            0xff0bff0a09080807ULL,
            0xff08ff0706050504ULL,
            0xff0dff0c0b0a0a09ULL,
            0xff0aff0908070706ULL,
            0xff0fff0e0d0c0c0bULL);
    const __m512i right = _mm512_srlv_epi16(
            _mm512_shuffle_epi8(spread, shuf_a),
            _mm512_set1_epi64(0x0006000400020000ULL));
    const __m512i left = _mm512_sllv_epi16(
            _mm512_shuffle_epi8(spread, shuf_b),
            _mm512_set1_epi64(0x0005000700010003ULL));
    return _mm512_and_si512(
            _mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaaULL, right, left),
            _mm512_set1_epi8(0x1f));
}

inline __m512i dense_decode_6_64_u8_avx512(const uint8_t* code) {
    const __m512i packed =
            _mm512_maskz_loadu_epi8((uint64_t{1} << 48) - 1, code);
    const __m512i expanded = _mm512_permutexvar_epi32(
            _mm512_setr_epi32(
                    0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1),
            packed);
    const __m512i shuf_0 = _mm512_broadcast_i32x4(
            _mm_setr_epi8(0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11));
    const __m512i shuf_1 = _mm512_broadcast_i32x4(_mm_setr_epi8(
            0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1));
    const __m512i left = _mm512_sllv_epi16(
            _mm512_shuffle_epi8(expanded, shuf_1),
            _mm512_set1_epi32(0x00060002));
    const __m512i right = _mm512_srlv_epi16(
            _mm512_shuffle_epi8(expanded, shuf_0),
            _mm512_set1_epi32(0x00040000));
    return _mm512_and_si512(
            _mm512_mask_blend_epi8(0x5555555555555555ULL, left, right),
            _mm512_set1_epi8(0x3f));
}

template <size_t NBITS>
inline __m512i dense_decode_64_u8_avx512(const uint8_t* code) {
    if constexpr (NBITS == 3) {
        return dense_decode_3_64_u8_avx512(code);
    } else if constexpr (NBITS == 4) {
        return dense_decode_4_64_u8_avx512(code);
    } else if constexpr (NBITS == 5) {
        return dense_decode_5_64_u8_avx512(code);
    } else if constexpr (NBITS == 6) {
        return dense_decode_6_64_u8_avx512(code);
    } else {
        static_assert(NBITS == 8);
        return _mm512_loadu_si512(code);
    }
}

inline void dense_fma_64_avx512(
        __m512& acc,
        __m512i decoded,
        const float* query,
        __m512 bias) {
#define FAISS_RABITQ_DENSE_FMA_QUARTER(Q, BYTES)                               \
    do {                                                                       \
        const __m512 values = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(BYTES)); \
        acc = _mm512_fmadd_ps(                                                 \
                _mm512_loadu_ps(query + 16 * (Q)),                             \
                _mm512_add_ps(values, bias),                                   \
                acc);                                                          \
    } while (false)
    FAISS_RABITQ_DENSE_FMA_QUARTER(0, _mm512_castsi512_si128(decoded));
    FAISS_RABITQ_DENSE_FMA_QUARTER(1, _mm512_extracti32x4_epi32(decoded, 1));
    FAISS_RABITQ_DENSE_FMA_QUARTER(2, _mm512_extracti32x4_epi32(decoded, 2));
    FAISS_RABITQ_DENSE_FMA_QUARTER(3, _mm512_extracti32x4_epi32(decoded, 3));
#undef FAISS_RABITQ_DENSE_FMA_QUARTER
}

template <size_t NBITS>
float ip_dense_avx512(
        const uint8_t* __restrict code,
        const float* __restrict query,
        size_t d,
        float cb) {
    __m512 acc = _mm512_setzero_ps();
    const __m512 bias = _mm512_set1_ps(cb);
    size_t i = 0;
    if constexpr ((NBITS >= 3 && NBITS <= 6) || NBITS == 8) {
        for (; i + 64 <= d; i += 64) {
            dense_fma_64_avx512(
                    acc,
                    dense_decode_64_u8_avx512<NBITS>(code + (i * NBITS) / 8),
                    query + i,
                    bias);
        }
    }
    for (; i + 16 <= d; i += 16) {
        const __m512 values =
                dense_decode_16_avx512<NBITS>(code + (i * NBITS) / 8);
        acc = _mm512_fmadd_ps(
                _mm512_loadu_ps(query + i), _mm512_add_ps(values, bias), acc);
    }
    return _mm512_reduce_add_ps(acc) +
            compute_inner_product_dense<SIMDLevel::NONE>(
                    code + (i * NBITS) / 8, query + i, d - i, NBITS, cb);
}

template <size_t NBITS>
void ip_dense_batch_4_avx512(
        const uint8_t* const codes[4],
        const float* __restrict query,
        size_t d,
        float cb,
        float out[4]) {
    __m512 acc[4] = {
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps()};
    const __m512 bias = _mm512_set1_ps(cb);
    size_t i = 0;
    if constexpr ((NBITS >= 3 && NBITS <= 6) || NBITS == 8) {
        for (; i + 64 <= d; i += 64) {
            for (size_t j = 0; j < 4; j++) {
                dense_fma_64_avx512(
                        acc[j],
                        dense_decode_64_u8_avx512<NBITS>(
                                codes[j] + (i * NBITS) / 8),
                        query + i,
                        bias);
            }
        }
    }
    for (; i + 16 <= d; i += 16) {
        const __m512 q = _mm512_loadu_ps(query + i);
        for (size_t j = 0; j < 4; j++) {
            const __m512 values =
                    dense_decode_16_avx512<NBITS>(codes[j] + (i * NBITS) / 8);
            acc[j] = _mm512_fmadd_ps(q, _mm512_add_ps(values, bias), acc[j]);
        }
    }
    for (size_t j = 0; j < 4; j++) {
        out[j] = _mm512_reduce_add_ps(acc[j]) +
                compute_inner_product_dense<SIMDLevel::NONE>(
                         codes[j] + (i * NBITS) / 8,
                         query + i,
                         d - i,
                         NBITS,
                         cb);
    }
}

} // namespace

template <>
float compute_inner_product_dense<SIMDLevel::AVX512>(
        const uint8_t* __restrict code,
        const float* __restrict query,
        size_t d,
        size_t nbits,
        float cb) {
#define FAISS_RABITQ_DENSE_CASE(N) \
    case N:                        \
        return ip_dense_avx512<N>(code, query, d, cb)
    switch (nbits) {
        FAISS_RABITQ_DENSE_CASE(2);
        FAISS_RABITQ_DENSE_CASE(3);
        FAISS_RABITQ_DENSE_CASE(4);
        FAISS_RABITQ_DENSE_CASE(5);
        FAISS_RABITQ_DENSE_CASE(6);
        FAISS_RABITQ_DENSE_CASE(7);
        FAISS_RABITQ_DENSE_CASE(8);
        default:
            return compute_inner_product_dense<SIMDLevel::NONE>(
                    code, query, d, nbits, cb);
    }
#undef FAISS_RABITQ_DENSE_CASE
}

template <>
void compute_inner_product_dense_batch_4<SIMDLevel::AVX512>(
        const uint8_t* const codes[4],
        const float* query,
        size_t d,
        size_t nbits,
        float cb,
        float out[4]) {
#define FAISS_RABITQ_DENSE_CASE(N) \
    case N:                        \
        return ip_dense_batch_4_avx512<N>(codes, query, d, cb, out)
    switch (nbits) {
        FAISS_RABITQ_DENSE_CASE(2);
        FAISS_RABITQ_DENSE_CASE(3);
        FAISS_RABITQ_DENSE_CASE(4);
        FAISS_RABITQ_DENSE_CASE(5);
        FAISS_RABITQ_DENSE_CASE(6);
        FAISS_RABITQ_DENSE_CASE(7);
        FAISS_RABITQ_DENSE_CASE(8);
        default:
            for (size_t i = 0; i < 4; i++) {
                out[i] = compute_inner_product_dense<SIMDLevel::NONE>(
                        codes[i], query, d, nbits, cb);
            }
    }
#undef FAISS_RABITQ_DENSE_CASE
}

template <>
float compute_inner_product_byte<SIMDLevel::AVX512>(
        const uint8_t* __restrict code,
        const float* __restrict query,
        size_t d,
        float cb) {
    __m512 acc = _mm512_setzero_ps();
    const __m512 bias = _mm512_set1_ps(cb);
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m128i bytes =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + i));
        const __m512 values = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
        acc = _mm512_fmadd_ps(
                _mm512_loadu_ps(query + i), _mm512_add_ps(values, bias), acc);
    }
    return _mm512_reduce_add_ps(acc) + ip_byte_scalar(code, query, i, d, cb);
}

template <>
void compute_inner_product_byte_batch_4<SIMDLevel::AVX512>(
        const uint8_t* const codes[4],
        const float* __restrict query,
        size_t d,
        float cb,
        float out[4]) {
    __m512 acc[4] = {
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps()};
    const __m512 bias = _mm512_set1_ps(cb);
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m512 q = _mm512_loadu_ps(query + i);
        for (size_t j = 0; j < 4; j++) {
            const __m128i bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(codes[j] + i));
            const __m512 values =
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
            acc[j] = _mm512_fmadd_ps(q, _mm512_add_ps(values, bias), acc[j]);
        }
    }
    for (size_t j = 0; j < 4; j++) {
        out[j] = _mm512_reduce_add_ps(acc[j]) +
                ip_byte_scalar(codes[j], query, i, d, cb);
    }
}

namespace {

#if (defined(__GNUC__) || defined(__clang__)) && \
        (defined(__x86_64__) || defined(__i386__))
#define FAISS_RABITQ_HAS_BMI2_TARGET 1
#define FAISS_RABITQ_TARGET_BMI2 __attribute__((target("bmi2")))

inline bool cpu_supports_fast_bmi2() {
    static const bool supported = __builtin_cpu_supports("bmi2");
    return supported && SIMDConfig::bmi2_fast;
}
#else
#define FAISS_RABITQ_HAS_BMI2_TARGET 0
#define FAISS_RABITQ_TARGET_BMI2
#endif

inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, shuf));
}

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

// AVX2+BMI2 bitplane kernel used as fallback for ex_bits >= 2.
// AVX512 TU has AVX2 available. BMI2 guarded separately since
// VIA Eden X4 has AVX2 without BMI2.
#if FAISS_RABITQ_HAS_BMI2_TARGET
FAISS_RABITQ_TARGET_BMI2 inline float ip_bitplane_avx2(
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
        __m256i sb_cmp = _mm256_cmpgt_epi32(
                _mm256_and_si256(_mm256_set1_epi32(sign_bits[i / 8]), bit_pos),
                zero);
        __m256 recon = _mm256_mul_ps(
                _mm256_and_ps(_mm256_castsi256_ps(sb_cmp), v_one),
                v_weights[ex_bits]);

        uint64_t ex64 = 0;
        memcpy(&ex64, ex_code + (i / 8) * ex_bits, sizeof(uint64_t));

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

// The 16-lane AVX-512 bit extraction below needs more than one 64-bit PEXT
// window once ex_bits exceeds four. Keep a genuine four-code path for those
// wider codes by sharing each query load across four 8-lane reconstructions.
FAISS_RABITQ_TARGET_BMI2 inline void ip_bitplane_batch_4_avx2(
        const uint8_t* const sign_bits[4],
        const uint8_t* const ex_codes[4],
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb,
        float out[4]) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256i bit_pos = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i zero = _mm256_setzero_si256();
    const __m256 v_cb = _mm256_set1_ps(cb);

    uint64_t pext_masks[7];
    __m256 v_weights[8];
    for (size_t b = 0; b < ex_bits; b++) {
        uint64_t mask = 0;
        for (int j = 0; j < 8; j++) {
            mask |= (1ULL << (b + j * ex_bits));
        }
        pext_masks[b] = mask;
        v_weights[b] = _mm256_set1_ps(static_cast<float>(1u << b));
    }
    v_weights[ex_bits] = _mm256_set1_ps(static_cast<float>(1u << ex_bits));

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256 query = _mm256_loadu_ps(rotated_q + i);

        auto reconstruct = [&](size_t code_index) FAISS_RABITQ_TARGET_BMI2 {
            const __m256i sign_cmp = _mm256_cmpgt_epi32(
                    _mm256_and_si256(
                            _mm256_set1_epi32(sign_bits[code_index][i / 8]),
                            bit_pos),
                    zero);
            __m256 reconstruction = _mm256_mul_ps(
                    _mm256_and_ps(_mm256_castsi256_ps(sign_cmp), v_one),
                    v_weights[ex_bits]);

            uint64_t extra_bits = 0;
            memcpy(&extra_bits,
                   ex_codes[code_index] + (i / 8) * ex_bits,
                   sizeof(extra_bits));
            for (size_t b = 0; b < ex_bits; b++) {
                const auto plane = static_cast<uint8_t>(
                        _pext_u64(extra_bits, pext_masks[b]));
                const __m256i plane_cmp = _mm256_cmpgt_epi32(
                        _mm256_and_si256(_mm256_set1_epi32(plane), bit_pos),
                        zero);
                const __m256 plane_values =
                        _mm256_and_ps(_mm256_castsi256_ps(plane_cmp), v_one);
                reconstruction = _mm256_fmadd_ps(
                        plane_values, v_weights[b], reconstruction);
            }
            return _mm256_add_ps(reconstruction, v_cb);
        };

        acc0 = _mm256_fmadd_ps(query, reconstruct(0), acc0);
        acc1 = _mm256_fmadd_ps(query, reconstruct(1), acc1);
        acc2 = _mm256_fmadd_ps(query, reconstruct(2), acc2);
        acc3 = _mm256_fmadd_ps(query, reconstruct(3), acc3);
    }

    out[0] = hsum_avx2(acc0) +
            ip_scalar(sign_bits[0], ex_codes[0], rotated_q, i, d, ex_bits, cb);
    out[1] = hsum_avx2(acc1) +
            ip_scalar(sign_bits[1], ex_codes[1], rotated_q, i, d, ex_bits, cb);
    out[2] = hsum_avx2(acc2) +
            ip_scalar(sign_bits[2], ex_codes[2], rotated_q, i, d, ex_bits, cb);
    out[3] = hsum_avx2(acc3) +
            ip_scalar(sign_bits[3], ex_codes[3], rotated_q, i, d, ex_bits, cb);
}
#endif

// For five to seven extra bits, eight packed coordinates still fit in one
// uint64_t. Extract the per-coordinate integers directly with AVX-512
// variable shifts instead of rebuilding them one bitplane at a time.
template <size_t ExBits>
inline float ip_packed_8_avx512(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        float cb) {
    static_assert(ExBits >= 5 && ExBits <= 7);
    __m256 acc = _mm256_setzero_ps();
    const __m256i bit_pos = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i zero = _mm256_setzero_si256();
    const __m256 v_cb = _mm256_set1_ps(cb);
    const __m256 v_sign_weight =
            _mm256_set1_ps(static_cast<float>(1u << ExBits));
    const __m512i shifts = _mm512_setr_epi64(
            0,
            ExBits,
            2 * ExBits,
            3 * ExBits,
            4 * ExBits,
            5 * ExBits,
            6 * ExBits,
            7 * ExBits);
    const __m512i value_mask = _mm512_set1_epi64((1u << ExBits) - 1);

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        uint64_t packed = 0;
        memcpy(&packed, ex_code + (i / 8) * ExBits, ExBits);
        const __m512i values = _mm512_and_si512(
                _mm512_srlv_epi64(_mm512_set1_epi64(packed), shifts),
                value_mask);
        const __m256 extra_values =
                _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(values));

        const __m256i sign_cmp = _mm256_cmpgt_epi32(
                _mm256_and_si256(_mm256_set1_epi32(sign_bits[i / 8]), bit_pos),
                zero);
        const __m256 sign_values =
                _mm256_and_ps(_mm256_castsi256_ps(sign_cmp), v_sign_weight);
        const __m256 reconstruction =
                _mm256_add_ps(_mm256_add_ps(extra_values, sign_values), v_cb);
        acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(rotated_q + i), reconstruction, acc);
    }

    return hsum_avx2(acc) +
            ip_scalar(sign_bits, ex_code, rotated_q, i, d, ExBits, cb);
}

template <size_t ExBits>
inline void ip_packed_8_batch_4_avx512(
        const uint8_t* const sign_bits[4],
        const uint8_t* const ex_codes[4],
        const float* __restrict rotated_q,
        size_t d,
        float cb,
        float out[4]) {
    static_assert(ExBits >= 5 && ExBits <= 7);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const __m256i bit_pos = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i zero = _mm256_setzero_si256();
    const __m256 v_cb = _mm256_set1_ps(cb);
    const __m256 v_sign_weight =
            _mm256_set1_ps(static_cast<float>(1u << ExBits));
    const __m512i shifts = _mm512_setr_epi64(
            0,
            ExBits,
            2 * ExBits,
            3 * ExBits,
            4 * ExBits,
            5 * ExBits,
            6 * ExBits,
            7 * ExBits);
    const __m512i value_mask = _mm512_set1_epi64((1u << ExBits) - 1);

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256 query = _mm256_loadu_ps(rotated_q + i);

        auto reconstruct = [&](size_t code_index) {
            uint64_t packed = 0;
            memcpy(&packed, ex_codes[code_index] + (i / 8) * ExBits, ExBits);
            const __m512i values = _mm512_and_si512(
                    _mm512_srlv_epi64(_mm512_set1_epi64(packed), shifts),
                    value_mask);
            const __m256 extra_values =
                    _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(values));

            const __m256i sign_cmp = _mm256_cmpgt_epi32(
                    _mm256_and_si256(
                            _mm256_set1_epi32(sign_bits[code_index][i / 8]),
                            bit_pos),
                    zero);
            const __m256 sign_values =
                    _mm256_and_ps(_mm256_castsi256_ps(sign_cmp), v_sign_weight);
            return _mm256_add_ps(
                    _mm256_add_ps(extra_values, sign_values), v_cb);
        };

        acc0 = _mm256_fmadd_ps(query, reconstruct(0), acc0);
        acc1 = _mm256_fmadd_ps(query, reconstruct(1), acc1);
        acc2 = _mm256_fmadd_ps(query, reconstruct(2), acc2);
        acc3 = _mm256_fmadd_ps(query, reconstruct(3), acc3);
    }

    out[0] = hsum_avx2(acc0) +
            ip_scalar(sign_bits[0], ex_codes[0], rotated_q, i, d, ExBits, cb);
    out[1] = hsum_avx2(acc1) +
            ip_scalar(sign_bits[1], ex_codes[1], rotated_q, i, d, ExBits, cb);
    out[2] = hsum_avx2(acc2) +
            ip_scalar(sign_bits[2], ex_codes[2], rotated_q, i, d, ExBits, cb);
    out[3] = hsum_avx2(acc3) +
            ip_scalar(sign_bits[3], ex_codes[3], rotated_q, i, d, ExBits, cb);
}

template <size_t ExBits>
inline __m256i unpack_packed_8_avx512(
        const uint8_t* __restrict ex_code,
        const __m512i shifts,
        const __m512i value_mask) {
    uint64_t packed = 0;
    memcpy(&packed, ex_code, ExBits);
    const __m512i values = _mm512_and_si512(
            _mm512_srlv_epi64(_mm512_set1_epi64(packed), shifts), value_mask);
    return _mm512_cvtepi64_epi32(values);
}

template <size_t ExBits>
inline float ip_packed_16_avx512(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        float cb) {
    static_assert(ExBits >= 5 && ExBits <= 7);
    __m512 acc = _mm512_setzero_ps();
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_cb = _mm512_set1_ps(cb);
    const __m512 v_sign_weight =
            _mm512_set1_ps(static_cast<float>(1u << ExBits));
    const __m512i shifts = _mm512_setr_epi64(
            0,
            ExBits,
            2 * ExBits,
            3 * ExBits,
            4 * ExBits,
            5 * ExBits,
            6 * ExBits,
            7 * ExBits);
    const __m512i value_mask = _mm512_set1_epi64((1u << ExBits) - 1);

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const uint8_t* block = ex_code + (i / 8) * ExBits;
        const __m256i lo =
                unpack_packed_8_avx512<ExBits>(block, shifts, value_mask);
        const __m256i hi = unpack_packed_8_avx512<ExBits>(
                block + ExBits, shifts, value_mask);
        const __m512i values =
                _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __m512 extra_values = _mm512_cvtepi32_ps(values);

        uint16_t sign_plane = 0;
        memcpy(&sign_plane, sign_bits + i / 8, sizeof(sign_plane));
        const __m512 sign_values = _mm512_mul_ps(
                _mm512_maskz_mov_ps(_cvtu32_mask16(sign_plane), v_one),
                v_sign_weight);
        const __m512 reconstruction =
                _mm512_add_ps(_mm512_add_ps(extra_values, sign_values), v_cb);
        acc = _mm512_fmadd_ps(
                _mm512_loadu_ps(rotated_q + i), reconstruction, acc);
    }

    return _mm512_reduce_add_ps(acc) +
            ip_scalar(sign_bits, ex_code, rotated_q, i, d, ExBits, cb);
}

template <size_t ExBits>
inline void ip_packed_16_batch_4_avx512(
        const uint8_t* const sign_bits[4],
        const uint8_t* const ex_codes[4],
        const float* __restrict rotated_q,
        size_t d,
        float cb,
        float out[4]) {
    static_assert(ExBits >= 5 && ExBits <= 7);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_cb = _mm512_set1_ps(cb);
    const __m512 v_sign_weight =
            _mm512_set1_ps(static_cast<float>(1u << ExBits));
    const __m512i shifts = _mm512_setr_epi64(
            0,
            ExBits,
            2 * ExBits,
            3 * ExBits,
            4 * ExBits,
            5 * ExBits,
            6 * ExBits,
            7 * ExBits);
    const __m512i value_mask = _mm512_set1_epi64((1u << ExBits) - 1);

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m512 query = _mm512_loadu_ps(rotated_q + i);

        auto reconstruct = [&](size_t code_index) {
            const uint8_t* block = ex_codes[code_index] + (i / 8) * ExBits;
            const __m256i lo =
                    unpack_packed_8_avx512<ExBits>(block, shifts, value_mask);
            const __m256i hi = unpack_packed_8_avx512<ExBits>(
                    block + ExBits, shifts, value_mask);
            const __m512i values =
                    _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            const __m512 extra_values = _mm512_cvtepi32_ps(values);

            uint16_t sign_plane = 0;
            memcpy(&sign_plane,
                   sign_bits[code_index] + i / 8,
                   sizeof(sign_plane));
            const __m512 sign_values = _mm512_mul_ps(
                    _mm512_maskz_mov_ps(_cvtu32_mask16(sign_plane), v_one),
                    v_sign_weight);
            return _mm512_add_ps(
                    _mm512_add_ps(extra_values, sign_values), v_cb);
        };

        acc0 = _mm512_fmadd_ps(query, reconstruct(0), acc0);
        acc1 = _mm512_fmadd_ps(query, reconstruct(1), acc1);
        acc2 = _mm512_fmadd_ps(query, reconstruct(2), acc2);
        acc3 = _mm512_fmadd_ps(query, reconstruct(3), acc3);
    }

    out[0] = _mm512_reduce_add_ps(acc0) +
            ip_scalar(sign_bits[0], ex_codes[0], rotated_q, i, d, ExBits, cb);
    out[1] = _mm512_reduce_add_ps(acc1) +
            ip_scalar(sign_bits[1], ex_codes[1], rotated_q, i, d, ExBits, cb);
    out[2] = _mm512_reduce_add_ps(acc2) +
            ip_scalar(sign_bits[2], ex_codes[2], rotated_q, i, d, ExBits, cb);
    out[3] = _mm512_reduce_add_ps(acc3) +
            ip_scalar(sign_bits[3], ex_codes[3], rotated_q, i, d, ExBits, cb);
}

#if FAISS_RABITQ_HAS_BMI2_TARGET
FAISS_RABITQ_TARGET_BMI2 inline float ip_bitplane_avx512(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    __m512 acc = _mm512_setzero_ps();
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_cb = _mm512_set1_ps(cb);
    const __m512 v_sign_weight =
            _mm512_set1_ps(static_cast<float>(1u << ex_bits));

    uint64_t pext_masks[4];
    __m512 v_weights[4];
    for (size_t b = 0; b < ex_bits; b++) {
        uint64_t mask = 0;
        for (int j = 0; j < 16; j++) {
            mask |= (1ULL << (j * ex_bits + b));
        }
        pext_masks[b] = mask;
        v_weights[b] = _mm512_set1_ps(static_cast<float>(1u << b));
    }

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        uint16_t sign_plane = 0;
        memcpy(&sign_plane, sign_bits + i / 8, sizeof(sign_plane));
        __m512 reconstruction = _mm512_mul_ps(
                _mm512_maskz_mov_ps(_cvtu32_mask16(sign_plane), v_one),
                v_sign_weight);

        uint64_t extra_bits = 0;
        memcpy(&extra_bits, ex_code + (i / 8) * ex_bits, sizeof(extra_bits));
        for (size_t b = 0; b < ex_bits; b++) {
            const uint16_t plane =
                    static_cast<uint16_t>(_pext_u64(extra_bits, pext_masks[b]));
            const __m512 plane_values =
                    _mm512_maskz_mov_ps(_cvtu32_mask16(plane), v_one);
            reconstruction =
                    _mm512_fmadd_ps(plane_values, v_weights[b], reconstruction);
        }

        const __m512 query = _mm512_loadu_ps(rotated_q + i);
        acc = _mm512_fmadd_ps(query, _mm512_add_ps(reconstruction, v_cb), acc);
    }

    float result = _mm512_reduce_add_ps(acc);
    result += ip_scalar(sign_bits, ex_code, rotated_q, i, d, ex_bits, cb);
    return result;
}

FAISS_RABITQ_TARGET_BMI2 inline void ip_bitplane_batch_4_avx512(
        const uint8_t* const sign_bits[4],
        const uint8_t* const ex_codes[4],
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb,
        float out[4]) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_cb = _mm512_set1_ps(cb);
    const __m512 v_sign_weight =
            _mm512_set1_ps(static_cast<float>(1u << ex_bits));

    uint64_t pext_masks[4];
    __m512 v_weights[4];
    for (size_t b = 0; b < ex_bits; b++) {
        uint64_t mask = 0;
        for (int j = 0; j < 16; j++) {
            mask |= (1ULL << (j * ex_bits + b));
        }
        pext_masks[b] = mask;
        v_weights[b] = _mm512_set1_ps(static_cast<float>(1u << b));
    }

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m512 query = _mm512_loadu_ps(rotated_q + i);

        auto reconstruct = [&](size_t code_index) FAISS_RABITQ_TARGET_BMI2 {
            uint16_t sign_plane = 0;
            memcpy(&sign_plane,
                   sign_bits[code_index] + i / 8,
                   sizeof(sign_plane));
            __m512 reconstruction = _mm512_mul_ps(
                    _mm512_maskz_mov_ps(_cvtu32_mask16(sign_plane), v_one),
                    v_sign_weight);

            uint64_t extra_bits = 0;
            memcpy(&extra_bits,
                   ex_codes[code_index] + (i / 8) * ex_bits,
                   sizeof(extra_bits));
            for (size_t b = 0; b < ex_bits; b++) {
                const uint16_t plane = static_cast<uint16_t>(
                        _pext_u64(extra_bits, pext_masks[b]));
                const __m512 plane_values =
                        _mm512_maskz_mov_ps(_cvtu32_mask16(plane), v_one);
                reconstruction = _mm512_fmadd_ps(
                        plane_values, v_weights[b], reconstruction);
            }
            return _mm512_add_ps(reconstruction, v_cb);
        };

        acc0 = _mm512_fmadd_ps(query, reconstruct(0), acc0);
        acc1 = _mm512_fmadd_ps(query, reconstruct(1), acc1);
        acc2 = _mm512_fmadd_ps(query, reconstruct(2), acc2);
        acc3 = _mm512_fmadd_ps(query, reconstruct(3), acc3);
    }

    out[0] = _mm512_reduce_add_ps(acc0) +
            ip_scalar(sign_bits[0], ex_codes[0], rotated_q, i, d, ex_bits, cb);
    out[1] = _mm512_reduce_add_ps(acc1) +
            ip_scalar(sign_bits[1], ex_codes[1], rotated_q, i, d, ex_bits, cb);
    out[2] = _mm512_reduce_add_ps(acc2) +
            ip_scalar(sign_bits[2], ex_codes[2], rotated_q, i, d, ex_bits, cb);
    out[3] = _mm512_reduce_add_ps(acc3) +
            ip_scalar(sign_bits[3], ex_codes[3], rotated_q, i, d, ex_bits, cb);
}
#endif

} // namespace

template <>
float compute_inner_product<SIMDLevel::AVX512>(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    if (ex_bits == 1) {
        return ip_1exbit_avx512(sign_bits, ex_code, rotated_q, d, cb);
    }

    switch (ex_bits) {
        case 5:
            return ip_packed_16_avx512<5>(sign_bits, ex_code, rotated_q, d, cb);
        case 6:
            return ip_packed_16_avx512<6>(sign_bits, ex_code, rotated_q, d, cb);
        case 7:
            return ip_packed_16_avx512<7>(sign_bits, ex_code, rotated_q, d, cb);
        default:
            break;
    }

#if FAISS_RABITQ_HAS_BMI2_TARGET
    if (ex_bits <= 4 && cpu_supports_fast_bmi2()) {
        return ip_bitplane_avx512(
                sign_bits, ex_code, rotated_q, d, ex_bits, cb);
    }
    if (ex_bits <= 7 && cpu_supports_fast_bmi2()) {
        return ip_bitplane_avx2(sign_bits, ex_code, rotated_q, d, ex_bits, cb);
    }
#endif
    return ip_scalar(sign_bits, ex_code, rotated_q, 0, d, ex_bits, cb);
}

template <>
void compute_inner_product_batch_4<SIMDLevel::AVX512>(
        const uint8_t* const sign_bits[4],
        const uint8_t* const ex_codes[4],
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb,
        float out[4]) {
    switch (ex_bits) {
        case 5:
            ip_packed_16_batch_4_avx512<5>(
                    sign_bits, ex_codes, rotated_q, d, cb, out);
            return;
        case 6:
            ip_packed_16_batch_4_avx512<6>(
                    sign_bits, ex_codes, rotated_q, d, cb, out);
            return;
        case 7:
            ip_packed_16_batch_4_avx512<7>(
                    sign_bits, ex_codes, rotated_q, d, cb, out);
            return;
        default:
            break;
    }
#if FAISS_RABITQ_HAS_BMI2_TARGET
    if (ex_bits <= 4 && cpu_supports_fast_bmi2()) {
        ip_bitplane_batch_4_avx512(
                sign_bits, ex_codes, rotated_q, d, ex_bits, cb, out);
        return;
    }
    if (ex_bits <= 7 && cpu_supports_fast_bmi2()) {
        ip_bitplane_batch_4_avx2(
                sign_bits, ex_codes, rotated_q, d, ex_bits, cb, out);
        return;
    }
#endif
    for (size_t i = 0; i < 4; i++) {
        out[i] = compute_inner_product<SIMDLevel::AVX512>(
                sign_bits[i], ex_codes[i], rotated_q, d, ex_bits, cb);
    }
}

#undef FAISS_RABITQ_TARGET_BMI2
#undef FAISS_RABITQ_HAS_BMI2_TARGET

} // namespace faiss::rabitq::multibit

#endif // COMPILE_SIMD_AVX512
