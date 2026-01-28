/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/platform_macros.h>

#include <faiss/utils/simd_levels.h>

#include <faiss/impl/ProductQuantizer.h>

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

namespace faiss {

// definiton and default implementation
template <typename PQDecoderT, SIMDLevel>
struct PQCodeDistance {
    using PQDecoder = PQDecoderT;

    /// Returns the distance to a single code.
    static float distance_single_code(
            // number of subquantizers
            const size_t M,
            // number of bits per quantization index
            const size_t nbits,
            // precomputed distances, layout (M, ksub)
            const float* sim_table,
            // the code
            const uint8_t* code) {
        PQDecoderT decoder(code, static_cast<int>(nbits));
        const size_t ksub = 1 << nbits;

        const float* tab = sim_table;
        float result = 0;

        for (size_t m = 0; m < M; m++) {
            result += tab[decoder.decode()];
            tab += ksub;
        }

        return result;
    }

    /// Combines 4 operations of distance_single_code()
    /// General-purpose version.
    static void distance_four_codes(
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
        PQDecoderT decoder0(code0, static_cast<int>(nbits));
        PQDecoderT decoder1(code1, static_cast<int>(nbits));
        PQDecoderT decoder2(code2, static_cast<int>(nbits));
        PQDecoderT decoder3(code3, static_cast<int>(nbits));
        const size_t ksub = 1 << nbits;

        const float* tab = sim_table;
        result0 = 0;
        result1 = 0;
        result2 = 0;
        result3 = 0;

        for (size_t m = 0; m < M; m++) {
            result0 += tab[decoder0.decode()];
            result1 += tab[decoder1.decode()];
            result2 += tab[decoder2.decode()];
            result3 += tab[decoder3.decode()];
            tab += ksub;
        }
    }
};

} // namespace faiss
