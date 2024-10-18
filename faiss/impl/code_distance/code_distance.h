/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/platform_macros.h>

// This directory contains functions to compute a distance
// from a given PQ code to a query vector, given that the
// distances to a query vector for pq.M codebooks are precomputed.
//
// The code was originally the part of IndexIVFPQ.cpp.
// The baseline implementation can be found in
//   code_distance-generic.h, distance_single_code_generic().

// The reason for this somewhat unusual structure is that
// custom implementations may need to fall off to generic
// implementation in certain cases. So, say, avx2 header file
// needs to reference the generic header file. This is
// why the names of the functions for custom implementations
// have this _generic or _avx2 suffix.

#ifdef __AVX2__

#include <faiss/impl/code_distance/code_distance-avx2.h>

namespace faiss {

template <typename PQDecoderT>
inline float distance_single_code(
        // number of subquantizers
        const size_t M,
        // number of bits per quantization index
        const size_t nbits,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        // the code
        const uint8_t* code) {
    return distance_single_code_avx2<PQDecoderT>(M, nbits, sim_table, code);
}

template <typename PQDecoderT>
inline void distance_four_codes(
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
    distance_four_codes_avx2<PQDecoderT>(
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

} // namespace faiss

#elif defined(__ARM_FEATURE_SVE)

#include <faiss/impl/code_distance/code_distance-sve.h>

namespace faiss {

template <typename PQDecoderT>
inline float distance_single_code(
        // the product quantizer
        const size_t M,
        // number of bits per quantization index
        const size_t nbits,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        // the code
        const uint8_t* code) {
    return distance_single_code_sve<PQDecoderT>(M, nbits, sim_table, code);
}

template <typename PQDecoderT>
inline void distance_four_codes(
        // the product quantizer
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
    distance_four_codes_sve<PQDecoderT>(
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

} // namespace faiss

#else

#include <faiss/impl/code_distance/code_distance-generic.h>

namespace faiss {

template <typename PQDecoderT>
inline float distance_single_code(
        // number of subquantizers
        const size_t M,
        // number of bits per quantization index
        const size_t nbits,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        // the code
        const uint8_t* code) {
    return distance_single_code_generic<PQDecoderT>(M, nbits, sim_table, code);
}

template <typename PQDecoderT>
inline void distance_four_codes(
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

} // namespace faiss

#endif
