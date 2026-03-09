/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <immintrin.h>

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

namespace faiss {
namespace pq_code_distance {

// According to experiments, the AVX-512 version may be SLOWER than
// the AVX2 version, which is somewhat unexpected.
// This version is kept for completeness.
//
// TODO: test for AMD CPUs.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
float pq_code_distance_single_impl<SIMDLevel::AVX512>(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code0) {
    float result0 = 0;
    constexpr size_t ksub = 1 << 8;

    size_t m = 0;
    const size_t pqM16 = M / 16;

    constexpr intptr_t N = 1;

    const float* tab = sim_table;

    if (pqM16 > 0) {
        const __m512i vksub = _mm512_set1_epi32(ksub);
        __m512i offsets_0 = _mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        offsets_0 = _mm512_mullo_epi32(offsets_0, vksub);

        __m512 partialSums[N];
        for (intptr_t j = 0; j < N; j++) {
            partialSums[j] = _mm512_setzero_ps();
        }

        // Process 16 values per loop iteration.
        for (m = 0; m < pqM16 * 16; m += 16) {
            __m128i mm1[N];
            mm1[0] = _mm_loadu_si128((const __m128i_u*)(code0 + m));

            for (intptr_t j = 0; j < N; j++) {
                const __m512i idx1 = _mm512_cvtepu8_epi32(mm1[j]);
                const __m512i indices_to_read_from =
                        _mm512_add_epi32(idx1, offsets_0);
                __m512 collected = _mm512_i32gather_ps(
                        indices_to_read_from, tab, sizeof(float));
                partialSums[j] = _mm512_add_ps(partialSums[j], collected);
            }
            tab += ksub * 16;
        }

        result0 += _mm512_reduce_add_ps(partialSums[0]);
    }

    // Process leftovers.
    if (m < M) {
        PQDecoder8 decoder0(code0 + m, nbits);
        for (; m < M; m++) {
            result0 += tab[decoder0.decode()];
            tab += ksub;
        }
    }

    return result0;
}

// Combines 4 operations of pq_code_distance_single_impl().
// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void pq_code_distance_four_impl<SIMDLevel::AVX512>(
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
        const __m512i vksub = _mm512_set1_epi32(ksub);
        __m512i offsets_0 = _mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        offsets_0 = _mm512_mullo_epi32(offsets_0, vksub);

        __m512 partialSums[N];
        for (intptr_t j = 0; j < N; j++) {
            partialSums[j] = _mm512_setzero_ps();
        }

        // Process 16 values per loop iteration.
        for (m = 0; m < pqM16 * 16; m += 16) {
            __m128i mm1[N];
            mm1[0] = _mm_loadu_si128((const __m128i_u*)(code0 + m));
            mm1[1] = _mm_loadu_si128((const __m128i_u*)(code1 + m));
            mm1[2] = _mm_loadu_si128((const __m128i_u*)(code2 + m));
            mm1[3] = _mm_loadu_si128((const __m128i_u*)(code3 + m));

            for (intptr_t j = 0; j < N; j++) {
                const __m512i idx1 = _mm512_cvtepu8_epi32(mm1[j]);
                const __m512i indices_to_read_from =
                        _mm512_add_epi32(idx1, offsets_0);
                __m512 collected = _mm512_i32gather_ps(
                        indices_to_read_from, tab, sizeof(float));
                partialSums[j] = _mm512_add_ps(partialSums[j], collected);
            }
            tab += ksub * 16;
        }

        result0 += _mm512_reduce_add_ps(partialSums[0]);
        result1 += _mm512_reduce_add_ps(partialSums[1]);
        result2 += _mm512_reduce_add_ps(partialSums[2]);
        result3 += _mm512_reduce_add_ps(partialSums[3]);
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

#ifdef COMPILE_SIMD_AVX512_SPR
// AVX512_SPR: Sapphire Rapids is a superset of AVX512. Reuse the
// AVX512 implementation until a dedicated SPR specialization is written.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
float pq_code_distance_single_impl<SIMDLevel::AVX512_SPR>(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code) {
    return pq_code_distance_single_impl<SIMDLevel::AVX512>(
            M, nbits, sim_table, code);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void pq_code_distance_four_impl<SIMDLevel::AVX512_SPR>(
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
    pq_code_distance_four_impl<SIMDLevel::AVX512>(
            M,
            nbits,
            sim_table,
            code0,
            code1,
            code2,
            code3,
            result0,
            result1,
            result2,
            result3);
}
#endif // COMPILE_SIMD_AVX512_SPR

} // namespace pq_code_distance
} // namespace faiss

#endif // COMPILE_SIMD_AVX512
