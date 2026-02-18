/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <immintrin.h>

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=78782
#if defined(__GNUC__) && __GNUC__ < 9
#define _mm_loadu_si64(x) (_mm_loadl_epi64((__m128i_u*)x))
#endif

namespace {

inline float horizontal_sum(const __m128 v) {
    const __m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));
    const __m128 v1 = _mm_add_ps(v, v0);
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    const __m128 v3 = _mm_add_ps(v1, v2);
    return _mm_cvtss_f32(v3);
}

// Computes a horizontal sum over an __m256 register.
inline float horizontal_sum(const __m256 v) {
    const __m128 v0 =
            _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    return horizontal_sum(v0);
}

// Processes a single code for M=4, ksub=256, nbits=8.
float inline distance_single_code_avx2_pqdecoder8_m4(
        const float* sim_table,
        const uint8_t* code) {
    const float* tab = sim_table;
    constexpr size_t ksub = 1 << 8;

    const __m128i vksub = _mm_set1_epi32(ksub);
    __m128i offsets_0 = _mm_setr_epi32(0, 1, 2, 3);
    offsets_0 = _mm_mullo_epi32(offsets_0, vksub);

    const __m128i mm1 = _mm_cvtsi32_si128(*((const int32_t*)code));
    const __m128i idx1 = _mm_cvtepu8_epi32(mm1);
    const __m128i indices_to_read_from = _mm_add_epi32(idx1, offsets_0);
    __m128 collected =
            _mm_i32gather_ps(tab, indices_to_read_from, sizeof(float));

    return horizontal_sum(collected);
}

// Processes a single code for M=8, ksub=256, nbits=8.
float inline distance_single_code_avx2_pqdecoder8_m8(
        const float* sim_table,
        const uint8_t* code) {
    const float* tab = sim_table;
    constexpr size_t ksub = 1 << 8;

    const __m256i vksub = _mm256_set1_epi32(ksub);
    __m256i offsets_0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    offsets_0 = _mm256_mullo_epi32(offsets_0, vksub);

    const __m128i mm1 = _mm_loadu_si64((const __m128i_u*)code);
    const __m256i idx1 = _mm256_cvtepu8_epi32(mm1);
    const __m256i indices_to_read_from = _mm256_add_epi32(idx1, offsets_0);
    __m256 collected =
            _mm256_i32gather_ps(tab, indices_to_read_from, sizeof(float));

    return horizontal_sum(collected);
}

inline void distance_four_codes_avx2_pqdecoder8_m4(
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    constexpr intptr_t N = 4;

    const float* tab = sim_table;
    constexpr size_t ksub = 1 << 8;

    const __m128i vksub = _mm_set1_epi32(ksub);
    __m128i offsets_0 = _mm_setr_epi32(0, 1, 2, 3);
    offsets_0 = _mm_mullo_epi32(offsets_0, vksub);

    __m128 partialSums[N];

    __m128i mm1[N];
    mm1[0] = _mm_cvtsi32_si128(*((const int32_t*)code0));
    mm1[1] = _mm_cvtsi32_si128(*((const int32_t*)code1));
    mm1[2] = _mm_cvtsi32_si128(*((const int32_t*)code2));
    mm1[3] = _mm_cvtsi32_si128(*((const int32_t*)code3));

    for (intptr_t j = 0; j < N; j++) {
        const __m128i idx1 = _mm_cvtepu8_epi32(mm1[j]);
        const __m128i indices_to_read_from = _mm_add_epi32(idx1, offsets_0);
        __m128 collected =
                _mm_i32gather_ps(tab, indices_to_read_from, sizeof(float));
        partialSums[j] = collected;
    }

    result0 = horizontal_sum(partialSums[0]);
    result1 = horizontal_sum(partialSums[1]);
    result2 = horizontal_sum(partialSums[2]);
    result3 = horizontal_sum(partialSums[3]);
}

inline void distance_four_codes_avx2_pqdecoder8_m8(
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    constexpr intptr_t N = 4;

    const float* tab = sim_table;
    constexpr size_t ksub = 1 << 8;

    const __m256i vksub = _mm256_set1_epi32(ksub);
    __m256i offsets_0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    offsets_0 = _mm256_mullo_epi32(offsets_0, vksub);

    __m256 partialSums[N];

    __m128i mm1[N];
    mm1[0] = _mm_loadu_si64((const __m128i_u*)code0);
    mm1[1] = _mm_loadu_si64((const __m128i_u*)code1);
    mm1[2] = _mm_loadu_si64((const __m128i_u*)code2);
    mm1[3] = _mm_loadu_si64((const __m128i_u*)code3);

    for (intptr_t j = 0; j < N; j++) {
        const __m256i idx1 = _mm256_cvtepu8_epi32(mm1[j]);
        const __m256i indices_to_read_from = _mm256_add_epi32(idx1, offsets_0);
        __m256 collected =
                _mm256_i32gather_ps(tab, indices_to_read_from, sizeof(float));
        partialSums[j] = collected;
    }

    result0 = horizontal_sum(partialSums[0]);
    result1 = horizontal_sum(partialSums[1]);
    result2 = horizontal_sum(partialSums[2]);
    result3 = horizontal_sum(partialSums[3]);
}

} // namespace

namespace faiss {
namespace pq_code_distance {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
float pq_code_distance_single_impl<SIMDLevel::AVX2>(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code) {
    if (M == 4) {
        return distance_single_code_avx2_pqdecoder8_m4(sim_table, code);
    }
    if (M == 8) {
        return distance_single_code_avx2_pqdecoder8_m8(sim_table, code);
    }

    // Precomputed distances, layout (M, ksub).
    float result = 0;
    constexpr size_t ksub = 1 << 8;

    size_t m = 0;
    const size_t pqM16 = M / 16;

    const float* tab = sim_table;

    if (pqM16 > 0) {
        const __m256i vksub = _mm256_set1_epi32(ksub);
        __m256i offsets_0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        offsets_0 = _mm256_mullo_epi32(offsets_0, vksub);

        __m256 partialSum = _mm256_setzero_ps();

        // Process 16 values per loop iteration.
        for (m = 0; m < pqM16 * 16; m += 16) {
            const __m128i mm1 = _mm_loadu_si128((const __m128i_u*)(code + m));
            // Process first 8 codes.
            {
                const __m256i idx1 = _mm256_cvtepu8_epi32(mm1);
                const __m256i indices_to_read_from =
                        _mm256_add_epi32(idx1, offsets_0);
                __m256 collected = _mm256_i32gather_ps(
                        tab, indices_to_read_from, sizeof(float));
                tab += ksub * 8;
                partialSum = _mm256_add_ps(partialSum, collected);
            }

            // Process next 8 codes.
            const __m128i mm2 = _mm_unpackhi_epi64(mm1, _mm_setzero_si128());
            {
                const __m256i idx1 = _mm256_cvtepu8_epi32(mm2);
                const __m256i indices_to_read_from =
                        _mm256_add_epi32(idx1, offsets_0);
                __m256 collected = _mm256_i32gather_ps(
                        tab, indices_to_read_from, sizeof(float));
                tab += ksub * 8;
                partialSum = _mm256_add_ps(partialSum, collected);
            }
        }

        // Horizontal sum for partialSum.
        result += horizontal_sum(partialSum);
    }

    // Process leftovers.
    if (m < M) {
        PQDecoder8 decoder(code + m, nbits);
        for (; m < M; m++) {
            result += tab[decoder.decode()];
            tab += ksub;
        }
    }

    return result;
}

// Combines 4 operations of pq_code_distance_single_impl().
// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void pq_code_distance_four_impl<SIMDLevel::AVX2>(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    if (M == 4) {
        distance_four_codes_avx2_pqdecoder8_m4(
                sim_table,
                code0,
                code1,
                code2,
                code3,
                result0,
                result1,
                result2,
                result3);
        return;
    }
    if (M == 8) {
        distance_four_codes_avx2_pqdecoder8_m8(
                sim_table,
                code0,
                code1,
                code2,
                code3,
                result0,
                result1,
                result2,
                result3);
        return;
    }

    // Precomputed distances, layout (M, ksub).
    result0 = 0;
    result1 = 0;
    result2 = 0;
    result3 = 0;
    constexpr size_t ksub = 1 << 8;

    size_t m = 0;
    const size_t pqM16 = M / 16;

    constexpr intptr_t N = 4;

    const float* tab = sim_table;

    if (pqM16 > 0) {
        const __m256i vksub = _mm256_set1_epi32(ksub);
        __m256i offsets_0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        offsets_0 = _mm256_mullo_epi32(offsets_0, vksub);

        __m256 partialSums[N];
        for (intptr_t j = 0; j < N; j++) {
            partialSums[j] = _mm256_setzero_ps();
        }

        // Process 16 values per loop iteration.
        for (m = 0; m < pqM16 * 16; m += 16) {
            __m128i mm1[N];
            mm1[0] = _mm_loadu_si128((const __m128i_u*)(code0 + m));
            mm1[1] = _mm_loadu_si128((const __m128i_u*)(code1 + m));
            mm1[2] = _mm_loadu_si128((const __m128i_u*)(code2 + m));
            mm1[3] = _mm_loadu_si128((const __m128i_u*)(code3 + m));

            // Process first 8 codes.
            for (intptr_t j = 0; j < N; j++) {
                const __m256i idx1 = _mm256_cvtepu8_epi32(mm1[j]);
                const __m256i indices_to_read_from =
                        _mm256_add_epi32(idx1, offsets_0);
                __m256 collected = _mm256_i32gather_ps(
                        tab, indices_to_read_from, sizeof(float));
                partialSums[j] = _mm256_add_ps(partialSums[j], collected);
            }
            tab += ksub * 8;

            // Process next 8 codes.
            for (intptr_t j = 0; j < N; j++) {
                const __m128i mm2 =
                        _mm_unpackhi_epi64(mm1[j], _mm_setzero_si128());
                const __m256i idx1 = _mm256_cvtepu8_epi32(mm2);
                const __m256i indices_to_read_from =
                        _mm256_add_epi32(idx1, offsets_0);
                __m256 collected = _mm256_i32gather_ps(
                        tab, indices_to_read_from, sizeof(float));
                partialSums[j] = _mm256_add_ps(partialSums[j], collected);
            }

            tab += ksub * 8;
        }

        result0 += horizontal_sum(partialSums[0]);
        result1 += horizontal_sum(partialSums[1]);
        result2 += horizontal_sum(partialSums[2]);
        result3 += horizontal_sum(partialSums[3]);
    }

    // Process leftovers.
    if (m < M) {
        PQDecoder8 decoder0(code0 + m, nbits);
        PQDecoder8 decoder1(code1 + m, nbits);
        PQDecoder8 decoder2(code2 + m, nbits);
        PQDecoder8 decoder3(code3 + m, nbits);
        for (; m < M; m++) {
            result0 += tab[decoder0.decode()];
            result1 += tab[decoder1.decode()];
            result2 += tab[decoder2.decode()];
            result3 += tab[decoder3.decode()];
            tab += ksub;
        }
    }
}

} // namespace pq_code_distance
} // namespace faiss

#endif // COMPILE_SIMD_AVX2
