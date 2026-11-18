/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/simdlib/simdlib_dispatch.h>

namespace faiss {

template <SIMDLevel SL>
inline simd32uint8_tpl<SL> high_nibbles(const simd32uint8_tpl<SL>& c) {
    if constexpr (SL == SIMDLevel::ARM_SVE) {
        return c.shr4();
    } else {
        return simd32uint8_tpl<SL>(simd16uint16_tpl<SL>(c) >> 4) &
                simd32uint8_tpl<SL>(0xf);
    }
}

/*
 * Multi-BB variant: accumulates NQ queries x BB*32 database elements.
 * Used by the search_1 path (bbs > 32).
 *
 * KernelSL selects the SIMD type width used for the inner accumulation
 * loop.  In DD mode the caller passes THE_LEVEL_TO_DISPATCH so the
 * kernel uses real AVX2/AVX512 types rather than the emulated-scalar
 * fallback that SINGLE_SIMD_LEVEL_256 would give.
 */
template <
        int NQ,
        int BB,
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void kernel_accumulate_block(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    constexpr SIMDLevel SL256 = simd256_level_selector<KernelSL>::value;
    using simd16uint16 = simd16uint16_tpl<SL256>;
    using simd32uint8 = simd32uint8_tpl<SL256>;

    // distance accumulators
    simd16uint16 accu[NQ][BB][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < BB; b++) {
            accu[q][b][0].clear();
            accu[q][b][1].clear();
            accu[q][b][2].clear();
            accu[q][b][3].clear();
        }
    }

    for (int sq = 0; sq < nsq - scaler.nscale; sq += 2) {
        simd32uint8 lut_cache[NQ];
        for (int q = 0; q < NQ; q++) {
            lut_cache[q] = simd32uint8(LUT);
            LUT += 32;
        }

        for (int b = 0; b < BB; b++) {
            simd32uint8 c = simd32uint8(codes);
            codes += 32;
            simd32uint8 mask(0xf);
            simd32uint8 chi = high_nibbles(c);
            simd32uint8 clo = c & mask;

            for (int q = 0; q < NQ; q++) {
                simd32uint8 lut = lut_cache[q];
                simd32uint8 res0 = lut.lookup_2_lanes(clo);
                simd32uint8 res1 = lut.lookup_2_lanes(chi);

                accu[q][b][0] += simd16uint16(res0);
                accu[q][b][1] += simd16uint16(res0) >> 8;

                accu[q][b][2] += simd16uint16(res1);
                accu[q][b][3] += simd16uint16(res1) >> 8;
            }
        }
    }

    for (int sq = 0; sq < scaler.nscale; sq += 2) {
        simd32uint8 lut_cache[NQ];
        for (int q = 0; q < NQ; q++) {
            lut_cache[q] = simd32uint8(LUT);
            LUT += 32;
        }

        for (int b = 0; b < BB; b++) {
            simd32uint8 c = simd32uint8(codes);
            codes += 32;
            simd32uint8 mask(0xf);
            simd32uint8 chi = high_nibbles(c);
            simd32uint8 clo = c & mask;

            for (int q = 0; q < NQ; q++) {
                simd32uint8 lut = lut_cache[q];

                simd32uint8 res0 = scaler.lookup(lut, clo);
                accu[q][b][0] += scaler.scale_lo(res0); // handle vectors 0..7
                accu[q][b][1] += scaler.scale_hi(res0); // handle vectors 8..15

                simd32uint8 res1 = scaler.lookup(lut, chi);
                accu[q][b][2] += scaler.scale_lo(res1); // handle vectors 16..23
                accu[q][b][3] +=
                        scaler.scale_hi(res1); //  handle vectors 24..31
            }
        }
    }

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < BB; b++) {
            accu[q][b][0] -= accu[q][b][1] << 8;
            simd16uint16 dis0 = combine2x2(accu[q][b][0], accu[q][b][1]);

            accu[q][b][2] -= accu[q][b][3] << 8;
            simd16uint16 dis1 = combine2x2(accu[q][b][2], accu[q][b][3]);

            res.handle(q, b, dis0, dis1);
        }
    }
}

/*
 * Single-BB QBS variant: accumulates NQ queries x 32 db elements (BB=1).
 * Used by the decompose_qbs layer for non-AVX512 paths.
 */
template <
        int NQ,
        bool ROW_LUT,
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void pq4_kernel_qbs_256_impl(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,         // ROW_LUT=false 时使用
        const uint8_t* const* rows, // ROW_LUT=true 时使用
        int q0,
        ResultHandler& res,
        const Scaler& scaler) {
    constexpr SIMDLevel SL256 = simd256_level_selector<KernelSL>::value;
    using simd16uint16 = simd16uint16_tpl<SL256>;
    using simd32uint8 = simd32uint8_tpl<SL256>;

    constexpr bool sve_wide_accumulate =
            SL256 == SIMDLevel::ARM_SVE && Scaler::nscale == 0;

    // dummy alloc to keep the windows compiler happy
    constexpr int NQA = NQ > 0 ? NQ : 1;
    // distance accumulators
    // layout: accu[q][b]: distance accumulator for vectors 8*b..8*b+7
    simd16uint16 accu[NQA][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
        }
    }

    // first sub-quantizer pair handled by the scaler tail
    const int sq_scale0 = nsq - scaler.nscale;

    for (int sq = 0; sq < sq_scale0; sq += 2) {
        simd32uint8 c;
        c.loadu(codes);
        codes += 32;

        simd32uint8 mask(0xf);
        simd32uint8 chi = high_nibbles(c);
        simd32uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            const uint8_t* lut_ptr;
            if constexpr (ROW_LUT) {
                lut_ptr = rows[q] + sq * 16;
            } else {
                lut_ptr = LUT;
                LUT += 32;
            }
            simd32uint8 lut(lut_ptr);

            simd32uint8 res0 = lut.lookup_2_lanes(clo);
            simd32uint8 res1 = lut.lookup_2_lanes(chi);

            if constexpr (sve_wide_accumulate) {
                accu[q][0].pq4_accumulate_low_bytes(res0);
                accu[q][1].pq4_accumulate_high_bytes(res0);
                accu[q][2].pq4_accumulate_low_bytes(res1);
                accu[q][3].pq4_accumulate_high_bytes(res1);
            } else {
                accu[q][0] += simd16uint16(res0);
                accu[q][1] += simd16uint16(res0) >> 8;

                accu[q][2] += simd16uint16(res1);
                accu[q][3] += simd16uint16(res1) >> 8;
            }
        }
    }

    for (int sq = 0; sq < scaler.nscale; sq += 2) {
        simd32uint8 c;
        c.loadu(codes);
        codes += 32;

        simd32uint8 mask(0xf);
        simd32uint8 chi = high_nibbles(c);
        simd32uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // tail continues where the main loop stopped, like the packed
            // variant continues its LUT pointer
            const uint8_t* lut_ptr;
            if constexpr (ROW_LUT) {
                lut_ptr = rows[q] + (sq_scale0 + sq) * 16;
            } else {
                lut_ptr = LUT;
                LUT += 32;
            }
            simd32uint8 lut(lut_ptr);

            simd32uint8 res0 = scaler.lookup(lut, clo);
            accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
            accu[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

            simd32uint8 res1 = scaler.lookup(lut, chi);
            accu[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
            accu[q][3] += scaler.scale_hi(res1); //  handle vectors 24..31
        }
    }

    if constexpr (sve_wide_accumulate) {
        for (int q = 0; q < NQ; q++) {
            simd16uint16 dis0 = combine2x2(accu[q][0], accu[q][1]);
            simd16uint16 dis1 = combine2x2(accu[q][2], accu[q][3]);
            res.handle(q + q0, 0, dis0, dis1);
        }
    } else {
        for (int q = 0; q < NQ; q++) {
            accu[q][0] -= accu[q][1] << 8;
            simd16uint16 dis0 = combine2x2(accu[q][0], accu[q][1]);
            accu[q][2] -= accu[q][3] << 8;
            simd16uint16 dis1 = combine2x2(accu[q][2], accu[q][3]);
            res.handle(q + q0, 0, dis0, dis1);
        }
    }
}

// packed LUT layout; see pq4_kernel_qbs_256_impl
template <
        int NQ,
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void pq4_kernel_qbs_256(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    pq4_kernel_qbs_256_impl<NQ, false, KernelSL>(
            nsq, codes, LUT, nullptr, 0, res, scaler);
}

// per-query LUT rows; see pq4_kernel_qbs_256_impl
template <
        int NQ,
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void pq4_kernel_qbs_rows_256(
        int nsq,
        const uint8_t* codes,
        const uint8_t* const* rows,
        int q0,
        ResultHandler& res,
        const Scaler& scaler) {
    pq4_kernel_qbs_256_impl<NQ, true, KernelSL>(
            nsq, codes, nullptr, rows, q0, res, scaler);
}

template <SIMDLevel SL>
struct PQ4QBSKernel {
    template <int NQ, class ResultHandler, class Scaler>
    FAISS_ALWAYS_INLINE static void run(
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            ResultHandler& res,
            const Scaler& scaler) {
        pq4_kernel_qbs_256<NQ, SL>(nsq, codes, LUT, res, scaler);
    }
};

} // namespace faiss
