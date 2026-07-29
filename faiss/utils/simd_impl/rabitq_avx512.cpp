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
#ifdef __BMI2__
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
#endif // __BMI2__

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

#ifdef __BMI2__
    if (ex_bits <= 7) {
        return ip_bitplane_avx2(sign_bits, ex_code, rotated_q, d, ex_bits, cb);
    }
#endif
    return ip_scalar(sign_bits, ex_code, rotated_q, 0, d, ex_bits, cb);
}

} // namespace faiss::rabitq::multibit

#endif // COMPILE_SIMD_AVX512
