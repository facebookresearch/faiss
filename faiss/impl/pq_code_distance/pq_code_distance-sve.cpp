/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_SVE

#include <arm_sve.h>

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

namespace {

inline void distance_codes_kernel(
        svbool_t pg,
        svuint32_t idx1,
        svuint32_t offsets_0,
        const float* tab,
        svfloat32_t& partialSum) {
    const auto indices_to_read_from = svadd_u32_x(pg, idx1, offsets_0);
    const auto collected =
            svld1_gather_u32index_f32(pg, tab, indices_to_read_from);
    partialSum = svadd_f32_m(pg, partialSum, collected);
}

inline float distance_single_code_sve_for_small_m(
        const size_t M,
        const float* sim_table,
        const uint8_t* __restrict code) {
    constexpr size_t nbits = 8u;
    const size_t ksub = 1 << nbits;

    const auto offsets_0 = svindex_u32(0, static_cast<uint32_t>(ksub));
    const auto pg = svwhilelt_b32_u64(0, M);

    auto mm1 = svld1ub_u32(pg, code);
    mm1 = svadd_u32_x(pg, mm1, offsets_0);
    const auto collected0 = svld1_gather_u32index_f32(pg, sim_table, mm1);
    return svaddv_f32(pg, collected0);
}

inline void distance_four_codes_sve_for_small_m(
        const size_t M,
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    constexpr size_t nbits = 8u;
    const size_t ksub = 1 << nbits;

    const auto offsets_0 = svindex_u32(0, static_cast<uint32_t>(ksub));
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

} // namespace

namespace faiss {
namespace pq_code_distance {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
float pq_code_distance_single_impl<SIMDLevel::ARM_SVE>(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code) {
    if (M <= svcntw())
        return distance_single_code_sve_for_small_m(M, sim_table, code);

    const float* tab = sim_table;
    const size_t ksub = 1 << nbits;

    const auto offsets_0 = svindex_u32(0, static_cast<uint32_t>(ksub));
    auto partialSum = svdup_n_f32(0.f);

    const auto lanes = svcntb();
    const auto quad_lanes = lanes / 4;

    for (std::size_t m = 0; m < M;) {
        const auto pg = svwhilelt_b8_u64(m, M);
        const auto mm1 = svld1_u8(pg, code + m);
        {
            const auto mm1lo = svunpklo_u16(mm1);
            const auto pglo = svunpklo_b(pg);

            {
                const auto idx1 = svunpklo_u32(mm1lo);
                const auto pglolo = svunpklo_b(pglo);
                distance_codes_kernel(pglolo, idx1, offsets_0, tab, partialSum);
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
            if (m >= M)
                break;

            {
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
                const auto idx1 = svunpklo_u32(mm1hi);
                const auto pghilo = svunpklo_b(pghi);
                distance_codes_kernel(pghilo, idx1, offsets_0, tab, partialSum);
                tab += ksub * quad_lanes;
            }

            m += quad_lanes;
            if (m >= M)
                break;

            {
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

// Combines 4 operations of pq_code_distance_single_impl().
// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void pq_code_distance_four_impl<SIMDLevel::ARM_SVE>(
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

    auto partialSum0 = svdup_n_f32(0.f);
    auto partialSum1 = svdup_n_f32(0.f);
    auto partialSum2 = svdup_n_f32(0.f);
    auto partialSum3 = svdup_n_f32(0.f);

    const auto lanes = svcntb();
    const auto quad_lanes = lanes / 4;

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

} // namespace pq_code_distance
} // namespace faiss

#endif // COMPILE_SIMD_ARM_SVE
