/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>

#include <tuple>
#include <type_traits>

#include <faiss/impl/code_distance/code_distance-generic.h>

namespace faiss {

template <typename PQDecoderT>
std::enable_if_t<!std::is_same_v<PQDecoderT, PQDecoder8>, float> inline distance_single_code_sve(
        // the product quantizer
        const size_t M,
        // number of bits per quantization index
        const size_t nbits,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        const uint8_t* code) {
    // default implementation
    return distance_single_code_generic<PQDecoderT>(M, nbits, sim_table, code);
}

static inline void distance_codes_kernel(
        svbool_t pg,
        svuint32_t idx1,
        svuint32_t offsets_0,
        const float* tab,
        svfloat32_t& partialSum) {
    // add offset
    const auto indices_to_read_from = svadd_u32_x(pg, idx1, offsets_0);

    // gather values, similar to some operations of tab[index]
    const auto collected =
            svld1_gather_u32index_f32(pg, tab, indices_to_read_from);

    // collect partial sum
    partialSum = svadd_f32_m(pg, partialSum, collected);
}

static float distance_single_code_sve_for_small_m(
        // the product quantizer
        const size_t M,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        // codes
        const uint8_t* __restrict code) {
    constexpr size_t nbits = 8u;

    const size_t ksub = 1 << nbits;

    const auto offsets_0 = svindex_u32(0, static_cast<uint32_t>(ksub));

    // loop
    const auto pg = svwhilelt_b32_u64(0, M);

    auto mm1 = svld1ub_u32(pg, code);
    mm1 = svadd_u32_x(pg, mm1, offsets_0);
    const auto collected0 = svld1_gather_u32index_f32(pg, sim_table, mm1);
    return svaddv_f32(pg, collected0);
}

template <typename PQDecoderT>
std::enable_if_t<std::is_same_v<PQDecoderT, PQDecoder8>, float> inline distance_single_code_sve(
        // the product quantizer
        const size_t M,
        // number of bits per quantization index
        const size_t nbits,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        const uint8_t* code) {
    if (M <= svcntw())
        return distance_single_code_sve_for_small_m(M, sim_table, code);

    const float* tab = sim_table;

    const size_t ksub = 1 << nbits;

    const auto offsets_0 = svindex_u32(0, static_cast<uint32_t>(ksub));

    // accumulators of partial sums
    auto partialSum = svdup_n_f32(0.f);

    const auto lanes = svcntb();
    const auto quad_lanes = lanes / 4;

    // loop
    for (std::size_t m = 0; m < M;) {
        const auto pg = svwhilelt_b8_u64(m, M);

        const auto mm1 = svld1_u8(pg, code + m);
        {
            const auto mm1lo = svunpklo_u16(mm1);
            const auto pglo = svunpklo_b(pg);

            {
                // convert uint8 values to uint32 values
                const auto idx1 = svunpklo_u32(mm1lo);
                const auto pglolo = svunpklo_b(pglo);

                distance_codes_kernel(pglolo, idx1, offsets_0, tab, partialSum);
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
            if (m >= M)
                break;

            {
                // convert uint8 values to uint32 values
                const auto idx1 = svunpkhi_u32(mm1lo);
                const auto pglohi = svunpkhi_b(pglo);

                distance_codes_kernel(pglohi, idx1, offsets_0, tab, partialSum);
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
            if (m >= M)
                break;
        }

        {
            const auto mm1hi = svunpkhi_u16(mm1);
            const auto pghi = svunpkhi_b(pg);

            {
                // convert uint8 values to uint32 values
                const auto idx1 = svunpklo_u32(mm1hi);
                const auto pghilo = svunpklo_b(pghi);

                distance_codes_kernel(pghilo, idx1, offsets_0, tab, partialSum);
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
            if (m >= M)
                break;

            {
                // convert uint8 values to uint32 values
                const auto idx1 = svunpkhi_u32(mm1hi);
                const auto pghihi = svunpkhi_b(pghi);

                distance_codes_kernel(pghihi, idx1, offsets_0, tab, partialSum);
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
        }
    }

    return svaddv_f32(svptrue_b32(), partialSum);
}

template <typename PQDecoderT>
std::enable_if_t<!std::is_same_v<PQDecoderT, PQDecoder8>, void>
distance_four_codes_sve(
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

static void distance_four_codes_sve_for_small_m(
        // the product quantizer
        const size_t M,
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
    constexpr size_t nbits = 8u;

    const size_t ksub = 1 << nbits;

    const auto offsets_0 = svindex_u32(0, static_cast<uint32_t>(ksub));

    const auto quad_lanes = svcntw();

    // loop
    const auto pg = svwhilelt_b32_u64(0, M);

    auto mm10 = svld1ub_u32(pg, code0);
    auto mm11 = svld1ub_u32(pg, code1);
    auto mm12 = svld1ub_u32(pg, code2);
    auto mm13 = svld1ub_u32(pg, code3);
    mm10 = svadd_u32_x(pg, mm10, offsets_0);
    mm11 = svadd_u32_x(pg, mm11, offsets_0);
    mm12 = svadd_u32_x(pg, mm12, offsets_0);
    mm13 = svadd_u32_x(pg, mm13, offsets_0);
    const auto collected0 = svld1_gather_u32index_f32(pg, sim_table, mm10);
    const auto collected1 = svld1_gather_u32index_f32(pg, sim_table, mm11);
    const auto collected2 = svld1_gather_u32index_f32(pg, sim_table, mm12);
    const auto collected3 = svld1_gather_u32index_f32(pg, sim_table, mm13);
    result0 = svaddv_f32(pg, collected0);
    result1 = svaddv_f32(pg, collected1);
    result2 = svaddv_f32(pg, collected2);
    result3 = svaddv_f32(pg, collected3);
}

// Combines 4 operations of distance_single_code()
template <typename PQDecoderT>
std::enable_if_t<std::is_same_v<PQDecoderT, PQDecoder8>, void>
distance_four_codes_sve(
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
    if (M <= svcntw()) {
        distance_four_codes_sve_for_small_m(
                M,
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

    const float* tab = sim_table;

    const size_t ksub = 1 << nbits;

    const auto offsets_0 = svindex_u32(0, static_cast<uint32_t>(ksub));

    // accumulators of partial sums
    auto partialSum0 = svdup_n_f32(0.f);
    auto partialSum1 = svdup_n_f32(0.f);
    auto partialSum2 = svdup_n_f32(0.f);
    auto partialSum3 = svdup_n_f32(0.f);

    const auto lanes = svcntb();
    const auto quad_lanes = lanes / 4;

    // loop
    for (std::size_t m = 0; m < M;) {
        const auto pg = svwhilelt_b8_u64(m, M);

        const auto mm10 = svld1_u8(pg, code0 + m);
        const auto mm11 = svld1_u8(pg, code1 + m);
        const auto mm12 = svld1_u8(pg, code2 + m);
        const auto mm13 = svld1_u8(pg, code3 + m);
        {
            const auto mm10lo = svunpklo_u16(mm10);
            const auto mm11lo = svunpklo_u16(mm11);
            const auto mm12lo = svunpklo_u16(mm12);
            const auto mm13lo = svunpklo_u16(mm13);
            const auto pglo = svunpklo_b(pg);

            {
                const auto pglolo = svunpklo_b(pglo);
                {
                    const auto idx1 = svunpklo_u32(mm10lo);
                    distance_codes_kernel(
                            pglolo, idx1, offsets_0, tab, partialSum0);
                }
                {
                    const auto idx1 = svunpklo_u32(mm11lo);
                    distance_codes_kernel(
                            pglolo, idx1, offsets_0, tab, partialSum1);
                }
                {
                    const auto idx1 = svunpklo_u32(mm12lo);
                    distance_codes_kernel(
                            pglolo, idx1, offsets_0, tab, partialSum2);
                }
                {
                    const auto idx1 = svunpklo_u32(mm13lo);
                    distance_codes_kernel(
                            pglolo, idx1, offsets_0, tab, partialSum3);
                }
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
            if (m >= M)
                break;

            {
                const auto pglohi = svunpkhi_b(pglo);
                {
                    const auto idx1 = svunpkhi_u32(mm10lo);
                    distance_codes_kernel(
                            pglohi, idx1, offsets_0, tab, partialSum0);
                }
                {
                    const auto idx1 = svunpkhi_u32(mm11lo);
                    distance_codes_kernel(
                            pglohi, idx1, offsets_0, tab, partialSum1);
                }
                {
                    const auto idx1 = svunpkhi_u32(mm12lo);
                    distance_codes_kernel(
                            pglohi, idx1, offsets_0, tab, partialSum2);
                }
                {
                    const auto idx1 = svunpkhi_u32(mm13lo);
                    distance_codes_kernel(
                            pglohi, idx1, offsets_0, tab, partialSum3);
                }
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
            if (m >= M)
                break;
        }

        {
            const auto mm10hi = svunpkhi_u16(mm10);
            const auto mm11hi = svunpkhi_u16(mm11);
            const auto mm12hi = svunpkhi_u16(mm12);
            const auto mm13hi = svunpkhi_u16(mm13);
            const auto pghi = svunpkhi_b(pg);

            {
                const auto pghilo = svunpklo_b(pghi);
                {
                    const auto idx1 = svunpklo_u32(mm10hi);
                    distance_codes_kernel(
                            pghilo, idx1, offsets_0, tab, partialSum0);
                }
                {
                    const auto idx1 = svunpklo_u32(mm11hi);
                    distance_codes_kernel(
                            pghilo, idx1, offsets_0, tab, partialSum1);
                }
                {
                    const auto idx1 = svunpklo_u32(mm12hi);
                    distance_codes_kernel(
                            pghilo, idx1, offsets_0, tab, partialSum2);
                }
                {
                    const auto idx1 = svunpklo_u32(mm13hi);
                    distance_codes_kernel(
                            pghilo, idx1, offsets_0, tab, partialSum3);
                }
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
            if (m >= M)
                break;

            {
                const auto pghihi = svunpkhi_b(pghi);
                {
                    const auto idx1 = svunpkhi_u32(mm10hi);
                    distance_codes_kernel(
                            pghihi, idx1, offsets_0, tab, partialSum0);
                }
                {
                    const auto idx1 = svunpkhi_u32(mm11hi);
                    distance_codes_kernel(
                            pghihi, idx1, offsets_0, tab, partialSum1);
                }
                {
                    const auto idx1 = svunpkhi_u32(mm12hi);
                    distance_codes_kernel(
                            pghihi, idx1, offsets_0, tab, partialSum2);
                }
                {
                    const auto idx1 = svunpkhi_u32(mm13hi);
                    distance_codes_kernel(
                            pghihi, idx1, offsets_0, tab, partialSum3);
                }
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
        }
    }

    result0 = svaddv_f32(svptrue_b32(), partialSum0);
    result1 = svaddv_f32(svptrue_b32(), partialSum1);
    result2 = svaddv_f32(svptrue_b32(), partialSum2);
    result3 = svaddv_f32(svptrue_b32(), partialSum3);
}

} // namespace faiss

#endif
