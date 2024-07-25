/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __AVX512F__

#include <immintrin.h>

#include <type_traits>

#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/code_distance/code_distance-generic.h>

namespace faiss {

// According to experiments, the AVX-512 version may be SLOWER than
//   the AVX2 version, which is somewhat unexpected.
// This version is not used for now, but it may be used later.
//
// TODO: test for AMD CPUs.

template <typename PQDecoderT>
typename std::enable_if<!std::is_same<PQDecoderT, PQDecoder8>::value, float>::
        type inline distance_single_code_avx512(
                // number of subquantizers
                const size_t M,
                // number of bits per quantization index
                const size_t nbits,
                // precomputed distances, layout (M, ksub)
                const float* sim_table,
                const uint8_t* code) {
    // default implementation
    return distance_single_code_generic<PQDecoderT>(M, nbits, sim_table, code);
}

template <typename PQDecoderT>
typename std::enable_if<std::is_same<PQDecoderT, PQDecoder8>::value, float>::
        type inline distance_single_code_avx512(
                // number of subquantizers
                const size_t M,
                // number of bits per quantization index
                const size_t nbits,
                // precomputed distances, layout (M, ksub)
                const float* sim_table,
                const uint8_t* code0) {
    float result0 = 0;
    constexpr size_t ksub = 1 << 8;

    size_t m = 0;
    const size_t pqM16 = M / 16;

    constexpr intptr_t N = 1;

    const float* tab = sim_table;

    if (pqM16 > 0) {
        // process 16 values per loop
        const __m512i vksub = _mm512_set1_epi32(ksub);
        __m512i offsets_0 = _mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        offsets_0 = _mm512_mullo_epi32(offsets_0, vksub);

        // accumulators of partial sums
        __m512 partialSums[N];
        for (intptr_t j = 0; j < N; j++) {
            partialSums[j] = _mm512_setzero_ps();
        }

        // loop
        for (m = 0; m < pqM16 * 16; m += 16) {
            // load 16 uint8 values
            __m128i mm1[N];
            mm1[0] = _mm_loadu_si128((const __m128i_u*)(code0 + m));

            // process first 8 codes
            for (intptr_t j = 0; j < N; j++) {
                const __m512i idx1 = _mm512_cvtepu8_epi32(mm1[j]);

                // add offsets
                const __m512i indices_to_read_from =
                        _mm512_add_epi32(idx1, offsets_0);

                // gather 16 values, similar to 16 operations of tab[idx]
                __m512 collected = _mm512_i32gather_ps(
                        indices_to_read_from, tab, sizeof(float));

                // collect partial sums
                partialSums[j] = _mm512_add_ps(partialSums[j], collected);
            }
            tab += ksub * 16;
        }

        // horizontal sum for partialSum
        result0 += _mm512_reduce_add_ps(partialSums[0]);
    }

    //
    if (m < M) {
        // process leftovers
        PQDecoder8 decoder0(code0 + m, nbits);
        for (; m < M; m++) {
            result0 += tab[decoder0.decode()];
            tab += ksub;
        }
    }

    return result0;
}

template <typename PQDecoderT>
typename std::enable_if<!std::is_same<PQDecoderT, PQDecoder8>::value, void>::
        type
        distance_four_codes_avx512(
                // number of subquantizers
                const size_t M,
                // number of bits per quantization index
                const size_t nbits,
                // precomputed distances, layout (M, ksub)
                const float* sim_table,
                // codes
                const uint8_t* __restrict code0,
                const uint8_t* __restrict code1,
                const uint8_t* __restrict code2,
                const uint8_t* __restrict code3,
                // computed distances
                float& result0,
                float& result1,
                float& result2,
                float& result3) {
    distance_four_codes_generic<PQDecoderT>(
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

// Combines 4 operations of distance_single_code()
template <typename PQDecoderT>
typename std::enable_if<std::is_same<PQDecoderT, PQDecoder8>::value, void>::type
distance_four_codes_avx512(
        // number of subquantizers
        const size_t M,
        // number of bits per quantization index
        const size_t nbits,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        // codes
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        // computed distances
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
        // process 16 values per loop
        const __m512i vksub = _mm512_set1_epi32(ksub);
        __m512i offsets_0 = _mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        offsets_0 = _mm512_mullo_epi32(offsets_0, vksub);

        // accumulators of partial sums
        __m512 partialSums[N];
        for (intptr_t j = 0; j < N; j++) {
            partialSums[j] = _mm512_setzero_ps();
        }

        // loop
        for (m = 0; m < pqM16 * 16; m += 16) {
            // load 16 uint8 values
            __m128i mm1[N];
            mm1[0] = _mm_loadu_si128((const __m128i_u*)(code0 + m));
            mm1[1] = _mm_loadu_si128((const __m128i_u*)(code1 + m));
            mm1[2] = _mm_loadu_si128((const __m128i_u*)(code2 + m));
            mm1[3] = _mm_loadu_si128((const __m128i_u*)(code3 + m));

            // process first 8 codes
            for (intptr_t j = 0; j < N; j++) {
                const __m512i idx1 = _mm512_cvtepu8_epi32(mm1[j]);

                // add offsets
                const __m512i indices_to_read_from =
                        _mm512_add_epi32(idx1, offsets_0);

                // gather 16 values, similar to 16 operations of tab[idx]
                __m512 collected = _mm512_i32gather_ps(
                        indices_to_read_from, tab, sizeof(float));

                // collect partial sums
                partialSums[j] = _mm512_add_ps(partialSums[j], collected);
            }
            tab += ksub * 16;
        }

        // horizontal sum for partialSum
        result0 += _mm512_reduce_add_ps(partialSums[0]);
        result1 += _mm512_reduce_add_ps(partialSums[1]);
        result2 += _mm512_reduce_add_ps(partialSums[2]);
        result3 += _mm512_reduce_add_ps(partialSums[3]);
    }

    //
    if (m < M) {
        // process leftovers
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

} // namespace faiss

#endif
