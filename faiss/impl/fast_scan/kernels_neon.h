/*
 * Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arm_neon.h>

#include <faiss/impl/fast_scan/kernels_simd256.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>

namespace faiss {

/*
 * Single-BB QBS variant: accumulates NQ queries x 32 db elements (BB=1).
 * Uses native 128-bit AdvSIMD registers instead of logical 256-bit values.
 */
template <int NQ, class ResultHandler, class Scaler>
FAISS_ALWAYS_INLINE void pq4_kernel_qbs_neon(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    // Avoid a zero-length array for NQ=0 instantiations.
    constexpr int NQA = NQ > 0 ? NQ : 1;
    const uint16x8_t zero = vdupq_n_u16(0);
    // accu[q][b] holds distances for vectors 8*b..8*b+7.
    uint16x8_t accu[NQA][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b] = zero;
        }
    }

    // Process two subquantizers per iteration.
    for (int sq = 0; sq < nsq - scaler.nscale; sq += 2) {
        const uint8x16_t code0 = vld1q_u8(codes);
        const uint8x16_t code1 = vld1q_u8(codes + 16);
        codes += 32;

        const uint8x16_t mask = vdupq_n_u8(0x0f);
        const uint8x16_t code0_lo = vandq_u8(code0, mask);
        const uint8x16_t code1_lo = vandq_u8(code1, mask);
        const uint8x16_t code0_hi = vshrq_n_u8(code0, 4);
        const uint8x16_t code1_hi = vshrq_n_u8(code1, 4);

        for (int q = 0; q < NQ; q++) {
            // Load LUTs for two subquantizers.
            const uint8x16_t lut0 = vld1q_u8(LUT);
            const uint8x16_t lut1 = vld1q_u8(LUT + 16);
            LUT += 32;

            const uint8x16_t d0_lo = vqtbl1q_u8(lut0, code0_lo);
            const uint8x16_t d0_hi = vqtbl1q_u8(lut0, code0_hi);
            const uint8x16_t d1_lo = vqtbl1q_u8(lut1, code1_lo);
            const uint8x16_t d1_hi = vqtbl1q_u8(lut1, code1_hi);

            accu[q][0] = vaddq_u16(
                    accu[q][0],
                    vaddl_u8(vget_low_u8(d0_lo), vget_low_u8(d1_lo)));
            accu[q][1] = vaddq_u16(accu[q][1], vaddl_high_u8(d0_lo, d1_lo));
            accu[q][2] = vaddq_u16(
                    accu[q][2],
                    vaddl_u8(vget_low_u8(d0_hi), vget_low_u8(d1_hi)));
            accu[q][3] = vaddq_u16(accu[q][3], vaddl_high_u8(d0_hi, d1_hi));
        }
    }

    // Scaled tail used by additive-quantizer FastScan indexes.
    // (this is compiled out for DummyScaler)
    for (int sq = 0; sq < scaler.nscale; sq += 2) {
        const uint16_t scale = scaler.scale_one(uint16_t{1});
        const uint8x16_t code0 = vld1q_u8(codes);
        const uint8x16_t code1 = vld1q_u8(codes + 16);
        codes += 32;

        const uint8x16_t mask = vdupq_n_u8(0x0f);
        const uint8x16_t code0_lo = vandq_u8(code0, mask);
        const uint8x16_t code1_lo = vandq_u8(code1, mask);
        const uint8x16_t code0_hi = vshrq_n_u8(code0, 4);
        const uint8x16_t code1_hi = vshrq_n_u8(code1, 4);

        for (int q = 0; q < NQ; q++) {
            // Load LUTs for two subquantizers.
            const uint8x16_t lut0 = vld1q_u8(LUT);
            const uint8x16_t lut1 = vld1q_u8(LUT + 16);
            LUT += 32;

            const uint8x16_t d0_lo = vqtbl1q_u8(lut0, code0_lo);
            const uint8x16_t d0_hi = vqtbl1q_u8(lut0, code0_hi);
            const uint8x16_t d1_lo = vqtbl1q_u8(lut1, code1_lo);
            const uint8x16_t d1_hi = vqtbl1q_u8(lut1, code1_hi);

            accu[q][0] = vaddq_u16(
                    accu[q][0],
                    vmulq_n_u16(
                            vaddl_u8(vget_low_u8(d0_lo), vget_low_u8(d1_lo)),
                            scale));
            accu[q][1] = vaddq_u16(
                    accu[q][1],
                    vmulq_n_u16(vaddl_high_u8(d0_lo, d1_lo), scale));
            accu[q][2] = vaddq_u16(
                    accu[q][2],
                    vmulq_n_u16(
                            vaddl_u8(vget_low_u8(d0_hi), vget_low_u8(d1_hi)),
                            scale));
            accu[q][3] = vaddq_u16(
                    accu[q][3],
                    vmulq_n_u16(vaddl_high_u8(d0_hi, d1_hi), scale));
        }
    }

    // Reorder the four 8-lane accumulators for the result handler.
    using simd16uint16 = simd16uint16_tpl<SIMDLevel::ARM_NEON>;
    for (int q = 0; q < NQ; q++) {
        const uint16x8_t d0 = vuzp1q_u16(accu[q][0], accu[q][1]);
        const uint16x8_t d1 = vuzp2q_u16(accu[q][0], accu[q][1]);
        const uint16x8_t d2 = vuzp1q_u16(accu[q][2], accu[q][3]);
        const uint16x8_t d3 = vuzp2q_u16(accu[q][2], accu[q][3]);
        res.handle(
                q,
                0,
                simd16uint16(uint16x8x2_t{d0, d1}),
                simd16uint16(uint16x8x2_t{d2, d3}));
    }
}

template <>
struct PQ4QBSKernel<SIMDLevel::ARM_NEON> {
    template <int NQ, class ResultHandler, class Scaler>
    FAISS_ALWAYS_INLINE static void run(
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            ResultHandler& res,
            const Scaler& scaler) {
        pq4_kernel_qbs_neon<NQ>(nsq, codes, LUT, res, scaler);
    }
};

} // namespace faiss
