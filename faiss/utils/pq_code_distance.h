/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {
namespace pq_code_distance {

/*********************************************************************
 * PQCodeDistance — SIMD-dispatched PQ code distance
 *
 * Computes the distance from a PQ-encoded vector to a query vector,
 * given a precomputed table of sub-distances (one per subquantizer
 * per centroid). Originally extracted from IndexIVFPQ.cpp.
 *
 * DESIGN:
 *
 * PQCodeDistance<PQDecoderT, SL> computes PQ code distances at a given
 * SIMD level. The dispatch site (IndexIVFPQ.cpp, IndexPQ.cpp) uses
 * DISPATCH_SIMDLevel to select SL at runtime, which instantiates
 * PQCodeDistance for ALL decoder types (PQDecoder8, PQDecoder16,
 * PQDecoderGeneric) at the chosen level.
 *
 * Only PQDecoder8 has SIMD-optimized implementations (AVX2, AVX512,
 * ARM_SVE). The other decoders always use scalar code — their decode()
 * method is inherently sequential, so SIMD doesn't help.
 *
 * The primary template is always complete (no forward declarations
 * needed). For PQDecoder8, it delegates to _impl dispatch bridge
 * functions whose specializations are defined in per-SIMD .cpp files
 * and resolved at link time. For other decoders, it uses scalar.
 *
 * ADDING A NEW SIMD LEVEL:
 *
 *   1. Add the level to SIMDLevel enum (simd_levels.h)
 *   2. Add dispatch_config entry (simd_dispatch.bzl)
 *   3. Define pq_code_distance_single_impl<NEW_LEVEL> and
 *      pq_code_distance_four_impl<NEW_LEVEL> specializations in a
 *      new .cpp file compiled with appropriate SIMD flags
 *   4. Add the .cpp to the build (CMakeLists.txt, xplat.bzl)
 *********************************************************************/

/// Scalar PQ code distance implementation.
/// Templated only on decoder type, independent of SIMD level.
/// Used directly by non-PQDecoder8 decoders (PQDecoder16,
/// PQDecoderGeneric) and as fallback for PQDecoder8 at NONE/NEON.
template <typename PQDecoderT>
struct PQCodeDistanceScalar {
    using PQDecoder = PQDecoderT;

    static float distance_single_code(
            // number of subquantizers
            size_t M,
            size_t nbits,
            // precomputed distances, layout (M, ksub)
            const float* sim_table,
            const uint8_t* code) {
        PQDecoderT decoder(code, nbits);
        const size_t ksub = 1 << nbits;

        const float* tab = sim_table;
        float result = 0;

        for (size_t m = 0; m < M; m++) {
            result += tab[decoder.decode()];
            tab += ksub;
        }

        return result;
    }

    static void distance_four_codes(
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
        PQDecoderT decoder0(code0, nbits);
        PQDecoderT decoder1(code1, nbits);
        PQDecoderT decoder2(code2, nbits);
        PQDecoderT decoder3(code3, nbits);
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

/*********************************************************************
 * Dispatch bridge — function templates for PQDecoder8 SIMD dispatch.
 *
 * Primary declarations only; specializations are defined in per-SIMD
 * .cpp files (AVX2, AVX512, ARM_SVE) and pq_code_distance-generic.cpp
 * (NONE, ARM_NEON). Same pattern as fvec_L2sqr et al. in distances.h.
 *********************************************************************/

template <SIMDLevel SL>
float pq_code_distance_single_impl(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code);

template <SIMDLevel SL>
void pq_code_distance_four_impl(
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
        float& result3);

/// Primary template — always complete.
/// For PQDecoder8, delegates to _impl dispatch bridges (resolved at
/// link time to per-SIMD implementations). For other decoders, uses
/// scalar — their sequential decode() methods don't benefit from SIMD.
template <typename PQDecoderT, SIMDLevel SL>
struct PQCodeDistance {
    using PQDecoder = PQDecoderT;

    static float distance_single_code(
            size_t M,
            size_t nbits,
            const float* sim_table,
            const uint8_t* code) {
        if constexpr (std::is_same_v<PQDecoderT, PQDecoder8>) {
            return pq_code_distance_single_impl<SL>(M, nbits, sim_table, code);
        } else {
            return PQCodeDistanceScalar<PQDecoderT>::distance_single_code(
                    M, nbits, sim_table, code);
        }
    }

    static void distance_four_codes(
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
        if constexpr (std::is_same_v<PQDecoderT, PQDecoder8>) {
            pq_code_distance_four_impl<SL>(
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
        } else {
            PQCodeDistanceScalar<PQDecoderT>::distance_four_codes(
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
    }
};

/*********************************************************************
 * Non-templated PQ code distance dispatch (PQDecoder8 only).
 *
 * These follow the same pattern as distances.h: the caller does not
 * name a SIMDLevel. Internally they dispatch via DISPATCH_SIMDLevel
 * to the best available SIMD implementation (DD: runtime detection,
 * static: compile-time selection). Definitions are in
 * pq_code_distance-generic.cpp.
 *********************************************************************/

/// Compute PQ distance for a single code, dispatching to the best
/// available SIMD level.
FAISS_API float pq_code_distance_single(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code);

/// Compute PQ distances for four codes simultaneously, dispatching
/// to the best available SIMD level.
FAISS_API void pq_code_distance_four(
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
        float& result3);

} // namespace pq_code_distance

// Re-export public API into namespace faiss for convenience
using pq_code_distance::pq_code_distance_four;
using pq_code_distance::pq_code_distance_single;
using pq_code_distance::PQCodeDistance;
using pq_code_distance::PQCodeDistanceScalar;

} // namespace faiss
