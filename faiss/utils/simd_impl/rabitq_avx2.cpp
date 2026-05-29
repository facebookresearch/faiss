/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <faiss/utils/rabitq_simd.h>
#include <immintrin.h>

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

} // namespace

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

#ifdef __BMI2__
    if (ex_bits <= 7) {
        return ip_bitplane_avx2(sign_bits, ex_code, rotated_q, d, ex_bits, cb);
    }
#endif
    return ip_scalar(sign_bits, ex_code, rotated_q, 0, d, ex_bits, cb);
}

} // namespace faiss::rabitq::multibit

#endif // COMPILE_SIMD_AVX2
