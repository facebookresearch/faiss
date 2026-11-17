/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

#if !defined(__ARM_FEATURE_SVE_BITS) || __ARM_FEATURE_SVE_BITS != 256
#error "pq_code_distance-sve.cpp requires -msve-vector-bits=256 (FAISS_SIMD_SVE256_FLAGS)"
#endif

namespace {

constexpr size_t ksub = 1u << 8;

constexpr size_t kLanes = 8;
constexpr size_t kTile = 32;

FAISS_ALWAYS_INLINE void build_lanes_codes_tiles(
        const uint8_t* __restrict codes,
        size_t M,
        size_t m0,
        size_t mlen,
        size_t R,
        uint8_t* __restrict scratch) {
    if (mlen == kTile) {
        const svbool_t pgb = svptrue_b8();
        for (size_t r = 0; r < R; r++) {
            const uint8_t* c = codes + r * kLanes * M + m0;
            uint8_t* ir = scratch + r * kTile * kLanes;
            const svuint8_t v0 = svld1_u8(pgb, c + 0 * M);
            const svuint8_t v1 = svld1_u8(pgb, c + 1 * M);
            const svuint8_t v2 = svld1_u8(pgb, c + 2 * M);
            const svuint8_t v3 = svld1_u8(pgb, c + 3 * M);
            const svuint8_t v4 = svld1_u8(pgb, c + 4 * M);
            const svuint8_t v5 = svld1_u8(pgb, c + 5 * M);
            const svuint8_t v6 = svld1_u8(pgb, c + 6 * M);
            const svuint8_t v7 = svld1_u8(pgb, c + 7 * M);

            const svuint8_t t0 = svzip1_u8(v0, v4);
            const svuint8_t t4 = svzip2_u8(v0, v4);
            const svuint8_t t1 = svzip1_u8(v1, v5);
            const svuint8_t t5 = svzip2_u8(v1, v5);
            const svuint8_t t2 = svzip1_u8(v2, v6);
            const svuint8_t t6 = svzip2_u8(v2, v6);
            const svuint8_t t3 = svzip1_u8(v3, v7);
            const svuint8_t t7 = svzip2_u8(v3, v7);

            const svuint8_t u0 = svzip1_u8(t0, t2);
            const svuint8_t u2 = svzip2_u8(t0, t2);
            const svuint8_t u1 = svzip1_u8(t1, t3);
            const svuint8_t u3 = svzip2_u8(t1, t3);
            const svuint8_t u4 = svzip1_u8(t4, t6);
            const svuint8_t u6 = svzip2_u8(t4, t6);
            const svuint8_t u5 = svzip1_u8(t5, t7);
            const svuint8_t u7 = svzip2_u8(t5, t7);

            svst1_u8(pgb, ir + 0 * kLanes, svzip1_u8(u0, u1));
            svst1_u8(pgb, ir + 4 * kLanes, svzip2_u8(u0, u1));
            svst1_u8(pgb, ir + 8 * kLanes, svzip1_u8(u2, u3));
            svst1_u8(pgb, ir + 12 * kLanes, svzip2_u8(u2, u3));
            svst1_u8(pgb, ir + 16 * kLanes, svzip1_u8(u4, u5));
            svst1_u8(pgb, ir + 20 * kLanes, svzip2_u8(u4, u5));
            svst1_u8(pgb, ir + 24 * kLanes, svzip1_u8(u6, u7));
            svst1_u8(pgb, ir + 28 * kLanes, svzip2_u8(u6, u7));
        }
        return;
    }

    for (size_t r = 0; r < R; r++) {
        const uint8_t* cr = codes + r * kLanes * M;
        uint8_t* ir = scratch + r * kTile * kLanes;
        for (size_t m = 0; m < mlen; m++) {
            for (size_t k = 0; k < kLanes; k++) {
                ir[m * kLanes + k] = cr[k * M + m0 + m];
            }
        }
    }
}

#define FAISS_PQ_SVE_DECL(r) svfloat32_t acc##r = svdup_n_f32(0.f);
#define FAISS_PQ_SVE_PTR(r) \
    const uint8_t* tile##r = scratch + (size_t)(r) * kTile * kLanes;
#define FAISS_PQ_SVE_STEP(r) \
    acc##r = svadd_f32_x(    \
            pg,              \
            acc##r,          \
            svld1_gather_u32index_f32(pg, row, svld1ub_u32(pg, tile##r)));
#define FAISS_PQ_SVE_ADV(r) tile##r += kLanes;
#define FAISS_PQ_SVE_STORE(r) \
    svst1_f32(pg, dist + (size_t)(r) * kLanes, acc##r);

#define FAISS_PQ_SVE_LIST_1(X) X(0)
#define FAISS_PQ_SVE_LIST_2(X) FAISS_PQ_SVE_LIST_1(X) X(1)
#define FAISS_PQ_SVE_LIST_4(X) FAISS_PQ_SVE_LIST_2(X) X(2) X(3)
#define FAISS_PQ_SVE_LIST_8(X) FAISS_PQ_SVE_LIST_4(X) X(4) X(5) X(6) X(7)
#define FAISS_PQ_SVE_LIST_16(X) \
    FAISS_PQ_SVE_LIST_8(X) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)

#define FAISS_PQ_SVE_BODY(LIST)                                  \
    LIST(FAISS_PQ_SVE_DECL)                                      \
    alignas(64) uint8_t scratch[R * kTile * kLanes];             \
    for (size_t m0 = 0; m0 < M; m0 += kTile) {                   \
        size_t mlen = M - m0;                                    \
        if (mlen > kTile) {                                      \
            mlen = kTile;                                        \
        }                                                        \
        build_lanes_codes_tiles(codes, M, m0, mlen, R, scratch); \
        LIST(FAISS_PQ_SVE_PTR)                                   \
        for (size_t t = 0; t < mlen; t++) {                      \
            const float* row = sim_table + (m0 + t) * ksub;      \
            LIST(FAISS_PQ_SVE_STEP)                              \
            LIST(FAISS_PQ_SVE_ADV)                               \
        }                                                        \
    }                                                            \
    LIST(FAISS_PQ_SVE_STORE)

template <int R>
inline void lanes_codes_group(
        const float* __restrict sim_table,
        const uint8_t* __restrict codes,
        size_t M,
        float* __restrict dist) {
    const svbool_t pg = svptrue_b32();

    if constexpr (R == 16) {
        FAISS_PQ_SVE_BODY(FAISS_PQ_SVE_LIST_16)
    } else if constexpr (R == 8) {
        FAISS_PQ_SVE_BODY(FAISS_PQ_SVE_LIST_8)
    } else if constexpr (R == 4) {
        FAISS_PQ_SVE_BODY(FAISS_PQ_SVE_LIST_4)
    } else if constexpr (R == 2) {
        FAISS_PQ_SVE_BODY(FAISS_PQ_SVE_LIST_2)
    } else {
        static_assert(R == 1, "unsupported lanes=codes group width");
        FAISS_PQ_SVE_BODY(FAISS_PQ_SVE_LIST_1)
    }
}

inline void lanes_codes_dispatch(
        size_t R,
        const float* __restrict sim_table,
        const uint8_t* __restrict codes,
        size_t M,
        float* __restrict dist) {
    switch (R) {
        case 16:
            lanes_codes_group<16>(sim_table, codes, M, dist);
            break;
        case 8:
            lanes_codes_group<8>(sim_table, codes, M, dist);
            break;
        case 4:
            lanes_codes_group<4>(sim_table, codes, M, dist);
            break;
        case 2:
            lanes_codes_group<2>(sim_table, codes, M, dist);
            break;
        default:
            lanes_codes_group<1>(sim_table, codes, M, dist);
            break;
    }
}

#undef FAISS_PQ_SVE_LIST_1
#undef FAISS_PQ_SVE_LIST_2
#undef FAISS_PQ_SVE_LIST_4
#undef FAISS_PQ_SVE_LIST_8
#undef FAISS_PQ_SVE_LIST_16
#undef FAISS_PQ_SVE_BODY
#undef FAISS_PQ_SVE_DECL
#undef FAISS_PQ_SVE_PTR
#undef FAISS_PQ_SVE_STEP
#undef FAISS_PQ_SVE_ADV
#undef FAISS_PQ_SVE_STORE

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

size_t pq_code_distance_8bit_sve_lanes_codes_scan(
        size_t M,
        const float* sim_table,
        const uint8_t* codes,
        size_t ncode,
        float* dist) {
    size_t R = 0;
    if (ncode >= 16 * kLanes) {
        R = 16;
    } else if (ncode >= 8 * kLanes) {
        R = 8;
    } else if (ncode >= 4 * kLanes) {
        R = 4;
    } else if (ncode >= 2 * kLanes) {
        R = 2;
    } else if (ncode >= kLanes) {
        R = 1;
    }
    if (R == 0) {
        return 0;
    }

    const size_t group = R * kLanes;
    lanes_codes_dispatch(R, sim_table, codes, M, dist);
    return group;
}

template <>
float pq_code_distance_8bit_single_impl<SIMDLevel::ARM_SVE>(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    if (M <= svcntw())
        return distance_single_code_sve_for_small_m(M, sim_table, code);

    const float* tab = sim_table;
    constexpr size_t ksub = 1 << 8;

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

template <>
void pq_code_distance_8bit_four_impl<SIMDLevel::ARM_SVE>(
        size_t M,
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
    constexpr size_t ksub = 1 << 8;

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

#define THE_SIMD_LEVEL SIMDLevel::ARM_SVE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/pq_scan_impl.h>
#undef THE_SIMD_LEVEL
