/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <faiss/utils/rabitq_simd.h>
#include <immintrin.h>
#include <limits>

namespace faiss::rabitq {

namespace {

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

inline float reduce_min_256(__m256 v) {
    __m128 x =
            _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_min_ps(x, _mm_movehl_ps(x, x));
    x = _mm_min_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}

inline float reduce_max_256(__m256 v) {
    __m128 x =
            _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_max_ps(x, _mm_movehl_ps(x, x));
    x = _mm_max_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}

inline __m256i round_nonnegative_ps_to_i32(__m256 x) {
    return _mm256_cvttps_epi32(_mm256_add_ps(x, _mm256_set1_ps(0.5f)));
}

inline void store_i32_as_u8_8(__m256i values, uint8_t* out) {
    const __m128i packed16 = _mm_packus_epi32(
            _mm256_castsi256_si128(values),
            _mm256_extracti128_si256(values, 1));
    const __m128i packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
    _mm_storel_epi64(reinterpret_cast<__m128i*>(out), packed8);
}

inline void accumulate_i32_as_i64(
        __m256i values,
        __m256i& low_acc,
        __m256i& high_acc) {
    low_acc = _mm256_add_epi64(
            low_acc, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(values)));
    high_acc = _mm256_add_epi64(
            high_acc,
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(values, 1)));
}

} // namespace

template <>
void lut_minmax_16<SIMDLevel::AVX2>(const float* tab, float& mn, float& mx) {
    const __m256 lo = _mm256_loadu_ps(tab);
    const __m256 hi = _mm256_loadu_ps(tab + 8);
    const __m256 min_vec = _mm256_min_ps(lo, hi);
    const __m256 max_vec = _mm256_max_ps(lo, hi);
    mn = reduce_min_256(min_vec);
    mx = reduce_max_256(max_vec);
}

template <>
void minmax_values<SIMDLevel::AVX2>(
        const float* values,
        size_t n,
        float& mn,
        float& mx) {
    if (n == 0) {
        return;
    }

    size_t i = 0;
    __m256 min_vec = _mm256_set1_ps(std::numeric_limits<float>::max());
    __m256 max_vec = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    for (; i + 8 <= n; i += 8) {
        const __m256 values_vec = _mm256_loadu_ps(values + i);
        min_vec = _mm256_min_ps(min_vec, values_vec);
        max_vec = _mm256_max_ps(max_vec, values_vec);
    }

    mn = reduce_min_256(min_vec);
    mx = reduce_max_256(max_vec);
    for (; i < n; i++) {
        mn = std::min(mn, values[i]);
        mx = std::max(mx, values[i]);
    }
}

template <>
void lut_quantize_16_to_uint8<SIMDLevel::AVX2>(
        const float* tab,
        float mn,
        float a,
        uint8_t* out) {
    const __m256 a_vec = _mm256_set1_ps(a);
    const __m256 mn_times_a_vec = _mm256_set1_ps(mn * a);
    const __m256i zero = _mm256_setzero_si256();
    for (size_t i = 0; i < 16; i += 8) {
        const __m256 values = _mm256_loadu_ps(tab + i);
        const __m256 scaled = _mm256_fmsub_ps(values, a_vec, mn_times_a_vec);
        const __m256i rounded =
                _mm256_max_epi32(round_nonnegative_ps_to_i32(scaled), zero);
        store_i32_as_u8_8(rounded, out + i);
    }
}

template <>
void quantize_query_values<SIMDLevel::AVX2>(
        const float* rq,
        size_t d,
        float v_min,
        float inv_delta,
        uint8_t max_code,
        bool centered,
        uint8_t* rqq,
        size_t& sum_qq,
        int64_t& sum2_signed_odd_int) {
    const __m256 inv_delta_vec = _mm256_set1_ps(inv_delta);
    const __m256 v_min_times_inv_delta_vec = _mm256_set1_ps(v_min * inv_delta);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 max_code_ps = _mm256_set1_ps(max_code);
    const __m256i max_code_i32 = _mm256_set1_epi32(max_code);
    const __m256i two = _mm256_set1_epi32(2);
    __m256i sum_acc_lo = _mm256_setzero_si256();
    __m256i sum_acc_hi = _mm256_setzero_si256();
    __m256i sq_acc_lo = _mm256_setzero_si256();
    __m256i sq_acc_hi = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256 values = _mm256_loadu_ps(rq + i);
        __m256 scaled = _mm256_fmsub_ps(
                values, inv_delta_vec, v_min_times_inv_delta_vec);
        scaled = _mm256_min_ps(_mm256_max_ps(scaled, zero), max_code_ps);
        const __m256i rounded = round_nonnegative_ps_to_i32(scaled);
        accumulate_i32_as_i64(rounded, sum_acc_lo, sum_acc_hi);

        if (centered) {
            const __m256i signed_odd = _mm256_sub_epi32(
                    _mm256_mullo_epi32(rounded, two), max_code_i32);
            const __m256i signed_odd_sqr =
                    _mm256_mullo_epi32(signed_odd, signed_odd);
            accumulate_i32_as_i64(signed_odd_sqr, sq_acc_lo, sq_acc_hi);
        }
        store_i32_as_u8_8(rounded, rqq + i);
    }

    sum_qq += reduce_add_256(sum_acc_lo) + reduce_add_256(sum_acc_hi);
    if (centered) {
        sum2_signed_odd_int +=
                reduce_add_256(sq_acc_lo) + reduce_add_256(sq_acc_hi);
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
uint64_t bitwise_and_dot_product<SIMDLevel::AVX2>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;
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
        const uint64_t yv = *(const uint64_t*)(data + offset);
        for (int j = 0; j < qb; j++) {
            const uint64_t qv = *(const uint64_t*)(query + j * size + offset);
            sum += popcount64(qv & yv) << j;
        }
    }
    for (; offset < size; ++offset) {
        const uint8_t yv = *(data + offset);
        for (int j = 0; j < qb; j++) {
            const uint8_t qv = *(query + j * size + offset);
            sum += popcount32(qv & yv) << j;
        }
    }
    return sum;
}

template <>
BitwiseAndDotProductResult bitwise_and_dot_product_with_popcount<
        SIMDLevel::AVX2>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t dot_product = 0;
    uint64_t popcount_sum = 0;
    size_t offset = 0;
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
        const uint64_t yv = *(const uint64_t*)(data + offset);
        popcount_sum += popcount64(yv);
        for (int j = 0; j < qb; j++) {
            const uint64_t qv = *(const uint64_t*)(query + j * size + offset);
            dot_product += popcount64(qv & yv) << j;
        }
    }
    for (; offset < size; ++offset) {
        const uint8_t yv = *(data + offset);
        popcount_sum += popcount32(yv);
        for (int j = 0; j < qb; j++) {
            const uint8_t qv = *(query + j * size + offset);
            dot_product += popcount32(qv & yv) << j;
        }
    }
    return {dot_product, popcount_sum};
}

template <>
uint64_t bitwise_xor_dot_product<SIMDLevel::AVX2>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;
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
uint64_t popcount<SIMDLevel::AVX2>(const uint8_t* data, size_t size) {
    uint64_t sum = 0;
    size_t offset = 0;
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
float selected_float_sum<SIMDLevel::AVX2>(
        const uint8_t* sign_bits,
        const float* values,
        size_t d) {
    const __m256i bit_positions =
            _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256i packed = _mm256_set1_epi32(sign_bits[i / 8]);
        const __m256i selected = _mm256_cmpeq_epi32(
                _mm256_and_si256(packed, bit_positions), bit_positions);
        const __m256 values_i = _mm256_loadu_ps(values + i);
        sum = _mm256_add_ps(
                sum, _mm256_and_ps(values_i, _mm256_castsi256_ps(selected)));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
            lanes[5] + lanes[6] + lanes[7];
    result += selected_float_sum<SIMDLevel::NONE>(
            sign_bits + i / 8, values + i, d - i);
    return result;
}

template <>
void rearrange_bit_planes<SIMDLevel::AVX2>(
        const uint8_t* rotated_qq,
        size_t d,
        size_t qb,
        uint8_t* out) {
    const size_t offset = (d + 7) / 8;
    memset(out, 0, offset * qb);
    const size_t nchunks = d / 32;
    for (size_t chunk = 0; chunk < nchunks; chunk++) {
        __m256i vals =
                _mm256_loadu_si256((const __m256i*)(rotated_qq + chunk * 32));
        for (size_t iv = 0; iv < qb; iv++) {
            __m256i mask = _mm256_set1_epi8(static_cast<char>(1 << iv));
            __m256i bits =
                    _mm256_cmpeq_epi8(_mm256_and_si256(vals, mask), mask);
            uint32_t packed = static_cast<uint32_t>(_mm256_movemask_epi8(bits));
            memcpy(&out[iv * offset + chunk * 4], &packed, 4);
        }
    }
    for (size_t idim = nchunks * 32; idim < d; idim++) {
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
inline __m256 dense_decode_8_avx2(const uint8_t* code) {
    static_assert(NBITS >= 2 && NBITS <= 8);
    uint64_t packed = 0;
    memcpy(&packed, code, NBITS);
    constexpr uint64_t mask = (uint64_t{1} << NBITS) - 1;
    const __m256i values = _mm256_setr_epi32(
            (packed >> (0 * NBITS)) & mask,
            (packed >> (1 * NBITS)) & mask,
            (packed >> (2 * NBITS)) & mask,
            (packed >> (3 * NBITS)) & mask,
            (packed >> (4 * NBITS)) & mask,
            (packed >> (5 * NBITS)) & mask,
            (packed >> (6 * NBITS)) & mask,
            (packed >> (7 * NBITS)) & mask);
    return _mm256_cvtepi32_ps(values);
}

template <size_t NBITS>
float ip_dense_avx2(
        const uint8_t* __restrict code,
        const float* __restrict query,
        size_t d,
        float cb) {
    __m256 acc = _mm256_setzero_ps();
    const __m256 bias = _mm256_set1_ps(cb);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256 values =
                dense_decode_8_avx2<NBITS>(code + (i * NBITS) / 8);
        acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(query + i), _mm256_add_ps(values, bias), acc);
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
            lanes[5] + lanes[6] + lanes[7];
    return result +
            compute_inner_product_dense<SIMDLevel::NONE>(
                    code + (i * NBITS) / 8, query + i, d - i, NBITS, cb);
}

template <size_t NBITS>
void ip_dense_batch_4_avx2(
        const uint8_t* const codes[4],
        const float* __restrict query,
        size_t d,
        float cb,
        float out[4]) {
    __m256 acc[4] = {
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps()};
    const __m256 bias = _mm256_set1_ps(cb);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256 q = _mm256_loadu_ps(query + i);
        for (size_t j = 0; j < 4; j++) {
            const __m256 values =
                    dense_decode_8_avx2<NBITS>(codes[j] + (i * NBITS) / 8);
            acc[j] = _mm256_fmadd_ps(q, _mm256_add_ps(values, bias), acc[j]);
        }
    }
    for (size_t j = 0; j < 4; j++) {
        float lanes[8];
        _mm256_storeu_ps(lanes, acc[j]);
        out[j] = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
                lanes[5] + lanes[6] + lanes[7] +
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
float compute_inner_product_dense<SIMDLevel::AVX2>(
        const uint8_t* __restrict code,
        const float* __restrict query,
        size_t d,
        size_t nbits,
        float cb) {
#define FAISS_RABITQ_DENSE_CASE(N) \
    case N:                        \
        return ip_dense_avx2<N>(code, query, d, cb)
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
void compute_inner_product_dense_batch_4<SIMDLevel::AVX2>(
        const uint8_t* const codes[4],
        const float* query,
        size_t d,
        size_t nbits,
        float cb,
        float out[4]) {
#define FAISS_RABITQ_DENSE_CASE(N) \
    case N:                        \
        return ip_dense_batch_4_avx2<N>(codes, query, d, cb, out)
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
float compute_inner_product_byte<SIMDLevel::AVX2>(
        const uint8_t* __restrict code,
        const float* __restrict query,
        size_t d,
        float cb) {
    __m256 acc = _mm256_setzero_ps();
    const __m256 bias = _mm256_set1_ps(cb);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m128i bytes =
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + i));
        const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
        acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(query + i), _mm256_add_ps(values, bias), acc);
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
            lanes[5] + lanes[6] + lanes[7];
    return result + ip_byte_scalar(code, query, i, d, cb);
}

template <>
void compute_inner_product_byte_batch_4<SIMDLevel::AVX2>(
        const uint8_t* const codes[4],
        const float* __restrict query,
        size_t d,
        float cb,
        float out[4]) {
    __m256 acc[4] = {
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps()};
    const __m256 bias = _mm256_set1_ps(cb);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256 q = _mm256_loadu_ps(query + i);
        for (size_t j = 0; j < 4; j++) {
            const __m128i bytes = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(codes[j] + i));
            const __m256 values =
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
            acc[j] = _mm256_fmadd_ps(q, _mm256_add_ps(values, bias), acc[j]);
        }
    }
    for (size_t j = 0; j < 4; j++) {
        float lanes[8];
        _mm256_storeu_ps(lanes, acc[j]);
        out[j] = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
                lanes[5] + lanes[6] + lanes[7] +
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

} // namespace

template <>
float compute_inner_product<SIMDLevel::AVX2>(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    if (ex_bits == 1) {
        return ip_1exbit_avx2(sign_bits, ex_code, rotated_q, d, cb);
    }

#if FAISS_RABITQ_HAS_BMI2_TARGET
    if (ex_bits <= 7 && cpu_supports_fast_bmi2()) {
        return ip_bitplane_avx2(sign_bits, ex_code, rotated_q, d, ex_bits, cb);
    }
#endif
    return ip_scalar(sign_bits, ex_code, rotated_q, 0, d, ex_bits, cb);
}

template <>
void compute_inner_product_batch_4<SIMDLevel::AVX2>(
        const uint8_t* const sign_bits[4],
        const uint8_t* const ex_codes[4],
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb,
        float out[4]) {
#if FAISS_RABITQ_HAS_BMI2_TARGET
    if (ex_bits <= 7 && cpu_supports_fast_bmi2()) {
        ip_bitplane_batch_4_avx2(
                sign_bits, ex_codes, rotated_q, d, ex_bits, cb, out);
        return;
    }
#endif
    for (size_t i = 0; i < 4; i++) {
        out[i] = compute_inner_product<SIMDLevel::AVX2>(
                sign_bits[i], ex_codes[i], rotated_q, d, ex_bits, cb);
    }
}

#undef FAISS_RABITQ_TARGET_BMI2
#undef FAISS_RABITQ_HAS_BMI2_TARGET

} // namespace faiss::rabitq::multibit

#endif // COMPILE_SIMD_AVX2
