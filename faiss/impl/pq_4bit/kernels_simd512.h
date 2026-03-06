/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/simdlib.h>

namespace faiss {

// NQ=1 specialization: processes 512-bit chunks aggressively.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <SIMDLevel SL = SINGLE_SIMD_LEVEL, class ResultHandler, class Scaler>
void kernel_accumulate_block_avx512_nq1(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;
    constexpr int NQ = 1;
    // distance accumulators
    simd32uint16<SL> accu[NQ][4];
    simd32uint16<SL> accu1[NQ][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
            accu1[q][b].clear();
        }
    }

    const int nsq_minus_nscale = nsq - scaler.nscale;
    const int nsq_minus_nscale_8 = (nsq_minus_nscale / 8) * 8;
    const int nsq_minus_nscale_4 = (nsq_minus_nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nsq_minus_nscale_8; sq += 8) {
        simd64uint8<SL> c(codes);
        codes += 64;

        simd64uint8<SL> c1(codes);
        codes += 64;

        simd64uint8<SL> mask(0xf);
        simd64uint8<SL> chi = simd64uint8<SL>(simd32uint16<SL>(c) >> 4) & mask;
        simd64uint8<SL> clo = c & mask;

        simd64uint8<SL> c1hi =
                simd64uint8<SL>(simd32uint16<SL>(c1) >> 4) & mask;
        simd64uint8<SL> c1lo = c1 & mask;

        for (int q = 0; q < NQ; q++) {
            simd64uint8<SL> lut(LUT);
            LUT += 64;

            {
                simd64uint8<SL> res0 = lut.lookup_4_lanes(clo);
                simd64uint8<SL> res1 = lut.lookup_4_lanes(chi);

                accu[q][0] += simd32uint16<SL>(res0);
                accu[q][1] += simd32uint16<SL>(res0) >> 8;
                accu[q][2] += simd32uint16<SL>(res1);
                accu[q][3] += simd32uint16<SL>(res1) >> 8;
            }
        }

        for (int q = 0; q < NQ; q++) {
            simd64uint8<SL> lut(LUT);
            LUT += 64;

            {
                simd64uint8<SL> res0 = lut.lookup_4_lanes(c1lo);
                simd64uint8<SL> res1 = lut.lookup_4_lanes(c1hi);

                accu1[q][0] += simd32uint16<SL>(res0);
                accu1[q][1] += simd32uint16<SL>(res0) >> 8;
                accu1[q][2] += simd32uint16<SL>(res1);
                accu1[q][3] += simd32uint16<SL>(res1) >> 8;
            }
        }
    }

    // process leftovers: a single chunk of size 4
    if (nsq_minus_nscale_8 != nsq_minus_nscale_4) {
        simd64uint8<SL> c(codes);
        codes += 64;

        simd64uint8<SL> mask(0xf);
        simd64uint8<SL> chi = simd64uint8<SL>(simd32uint16<SL>(c) >> 4) & mask;
        simd64uint8<SL> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd64uint8<SL> lut(LUT);
            LUT += 64;

            simd64uint8<SL> res0 = lut.lookup_4_lanes(clo);
            simd64uint8<SL> res1 = lut.lookup_4_lanes(chi);

            accu[q][0] += simd32uint16<SL>(res0);
            accu[q][1] += simd32uint16<SL>(res0) >> 8;
            accu[q][2] += simd32uint16<SL>(res1);
            accu[q][3] += simd32uint16<SL>(res1) >> 8;
        }
    }

    // process leftovers: a single chunk of size 2
    if (nsq_minus_nscale_4 != nsq_minus_nscale) {
        simd32uint8<SL256> c(codes);
        codes += 32;

        simd32uint8<SL256> mask(0xf);
        simd32uint8<SL256> chi =
                simd32uint8<SL256>(simd16uint16<SL256>(c) >> 4) & mask;
        simd32uint8<SL256> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd32uint8<SL256> lut(LUT);
            LUT += 32;

            simd32uint8<SL256> res0 = lut.lookup_2_lanes(clo);
            simd32uint8<SL256> res1 = lut.lookup_2_lanes(chi);

            accu[q][0] += simd32uint16<SL>(simd16uint16<SL256>(res0));
            accu[q][1] += simd32uint16<SL>(simd16uint16<SL256>(res0) >> 8);
            accu[q][2] += simd32uint16<SL>(simd16uint16<SL256>(res1));
            accu[q][3] += simd32uint16<SL>(simd16uint16<SL256>(res1) >> 8);
        }
    }

    // process "sq" part
    const int nscale = scaler.nscale;
    const int nscale_8 = (nscale / 8) * 8;
    const int nscale_4 = (nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nscale_8; sq += 8) {
        simd64uint8<SL> c(codes);
        codes += 64;

        simd64uint8<SL> c1(codes);
        codes += 64;

        simd64uint8<SL> mask(0xf);
        simd64uint8<SL> chi = simd64uint8<SL>(simd32uint16<SL>(c) >> 4) & mask;
        simd64uint8<SL> clo = c & mask;

        simd64uint8<SL> c1hi =
                simd64uint8<SL>(simd32uint16<SL>(c1) >> 4) & mask;
        simd64uint8<SL> c1lo = c1 & mask;

        for (int q = 0; q < NQ; q++) {
            simd64uint8<SL> lut(LUT);
            LUT += 64;

            {
                simd64uint8<SL> res0 = scaler.lookup(lut, clo);
                accu[q][0] += scaler.scale_lo(res0);
                accu[q][1] += scaler.scale_hi(res0);

                simd64uint8<SL> res1 = scaler.lookup(lut, chi);
                accu[q][2] += scaler.scale_lo(res1);
                accu[q][3] += scaler.scale_hi(res1);
            }
        }

        for (int q = 0; q < NQ; q++) {
            simd64uint8<SL> lut(LUT);
            LUT += 64;

            {
                simd64uint8<SL> res0 = scaler.lookup(lut, c1lo);
                accu1[q][0] += scaler.scale_lo(res0);
                accu1[q][1] += scaler.scale_hi(res0);

                simd64uint8<SL> res1 = scaler.lookup(lut, c1hi);
                accu1[q][2] += scaler.scale_lo(res1);
                accu1[q][3] += scaler.scale_hi(res1);
            }
        }
    }

    // process leftovers: a single chunk of size 4
    if (nscale_8 != nscale_4) {
        simd64uint8<SL> c(codes);
        codes += 64;

        simd64uint8<SL> mask(0xf);
        simd64uint8<SL> chi = simd64uint8<SL>(simd32uint16<SL>(c) >> 4) & mask;
        simd64uint8<SL> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd64uint8<SL> lut(LUT);
            LUT += 64;

            simd64uint8<SL> res0 = scaler.lookup(lut, clo);
            accu[q][0] += scaler.scale_lo(res0);
            accu[q][1] += scaler.scale_hi(res0);

            simd64uint8<SL> res1 = scaler.lookup(lut, chi);
            accu[q][2] += scaler.scale_lo(res1);
            accu[q][3] += scaler.scale_hi(res1);
        }
    }

    // process leftovers: a single chunk of size 2
    if (nscale_4 != nscale) {
        simd32uint8<SL256> c(codes);
        codes += 32;

        simd32uint8<SL256> mask(0xf);
        simd32uint8<SL256> chi =
                simd32uint8<SL256>(simd16uint16<SL256>(c) >> 4) & mask;
        simd32uint8<SL256> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd32uint8<SL256> lut(LUT);
            LUT += 32;

            simd32uint8<SL256> res0 = scaler.lookup(lut, clo);
            accu[q][0] += simd32uint16<SL>(scaler.scale_lo(res0));
            accu[q][1] += simd32uint16<SL>(scaler.scale_hi(res0));

            simd32uint8<SL256> res1 = scaler.lookup(lut, chi);
            accu[q][2] += simd32uint16<SL>(scaler.scale_lo(res1));
            accu[q][3] += simd32uint16<SL>(scaler.scale_hi(res1));
        }
    }

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b] += accu1[q][b];
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16<SL256> dis0 = combine4x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16<SL256> dis1 = combine4x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

// General NQ case for AVX512.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <
        int NQ,
        SIMDLevel SL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void kernel_accumulate_block_avx512_nqx(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;
    constexpr int NQA = NQ > 0 ? NQ : 1;
    simd32uint16<SL> accu[NQA][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
        }
    }

    const int nsq_minus_nscale = nsq - scaler.nscale;
    const int nsq_minus_nscale_4 = (nsq_minus_nscale / 4) * 4;

    for (int sq = 0; sq < nsq_minus_nscale_4; sq += 4) {
        simd64uint8<SL> c(codes);
        codes += 64;

        simd64uint8<SL> mask(0xf);
        simd64uint8<SL> chi = simd64uint8<SL>(simd32uint16<SL>(c) >> 4) & mask;
        simd64uint8<SL> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd32uint8<SL256> lut_a(LUT);
            simd32uint8<SL256> lut_b(LUT + NQ * 32);

            simd64uint8<SL> lut(lut_a, lut_b);
            LUT += 32;

            {
                simd64uint8<SL> res0 = lut.lookup_4_lanes(clo);
                simd64uint8<SL> res1 = lut.lookup_4_lanes(chi);

                accu[q][0] += simd32uint16<SL>(res0);
                accu[q][1] += simd32uint16<SL>(res0) >> 8;
                accu[q][2] += simd32uint16<SL>(res1);
                accu[q][3] += simd32uint16<SL>(res1) >> 8;
            }
        }

        LUT += NQ * 32;
    }

    // process leftovers: a single chunk of size 2
    if (nsq_minus_nscale_4 != nsq_minus_nscale) {
        simd32uint8<SL256> c(codes);
        codes += 32;

        simd32uint8<SL256> mask(0xf);
        simd32uint8<SL256> chi =
                simd32uint8<SL256>(simd16uint16<SL256>(c) >> 4) & mask;
        simd32uint8<SL256> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd32uint8<SL256> lut(LUT);
            LUT += 32;

            simd32uint8<SL256> res0 = lut.lookup_2_lanes(clo);
            simd32uint8<SL256> res1 = lut.lookup_2_lanes(chi);

            accu[q][0] += simd32uint16<SL>(simd16uint16<SL256>(res0));
            accu[q][1] += simd32uint16<SL>(simd16uint16<SL256>(res0) >> 8);
            accu[q][2] += simd32uint16<SL>(simd16uint16<SL256>(res1));
            accu[q][3] += simd32uint16<SL>(simd16uint16<SL256>(res1) >> 8);
        }
    }

    // process "sq" part
    const int nscale = scaler.nscale;
    const int nscale_4 = (nscale / 4) * 4;

    for (int sq = 0; sq < nscale_4; sq += 4) {
        simd64uint8<SL> c(codes);
        codes += 64;

        simd64uint8<SL> mask(0xf);
        simd64uint8<SL> chi = simd64uint8<SL>(simd32uint16<SL>(c) >> 4) & mask;
        simd64uint8<SL> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd32uint8<SL256> lut_a(LUT);
            simd32uint8<SL256> lut_b(LUT + NQ * 32);

            simd64uint8<SL> lut(lut_a, lut_b);
            LUT += 32;

            {
                simd64uint8<SL> res0 = scaler.lookup(lut, clo);
                accu[q][0] += scaler.scale_lo(res0);
                accu[q][1] += scaler.scale_hi(res0);

                simd64uint8<SL> res1 = scaler.lookup(lut, chi);
                accu[q][2] += scaler.scale_lo(res1);
                accu[q][3] += scaler.scale_hi(res1);
            }
        }

        LUT += NQ * 32;
    }

    // process leftovers: a single chunk of size 2
    if (nscale_4 != nscale) {
        simd32uint8<SL256> c(codes);
        codes += 32;

        simd32uint8<SL256> mask(0xf);
        simd32uint8<SL256> chi =
                simd32uint8<SL256>(simd16uint16<SL256>(c) >> 4) & mask;
        simd32uint8<SL256> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd32uint8<SL256> lut(LUT);
            LUT += 32;

            simd32uint8<SL256> res0 = scaler.lookup(lut, clo);
            accu[q][0] += simd32uint16<SL>(scaler.scale_lo(res0));
            accu[q][1] += simd32uint16<SL>(scaler.scale_hi(res0));

            simd32uint8<SL256> res1 = scaler.lookup(lut, chi);
            accu[q][2] += simd32uint16<SL>(scaler.scale_lo(res1));
            accu[q][3] += simd32uint16<SL>(scaler.scale_hi(res1));
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16<SL256> dis0 = combine4x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16<SL256> dis1 = combine4x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

// Dispatcher: selects NQ=1 vs general case.
template <
        int NQ,
        SIMDLevel SL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void pq4_kernel_qbs_512(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    if constexpr (NQ == 1) {
        kernel_accumulate_block_avx512_nq1<SL>(nsq, codes, LUT, res, scaler);
    } else {
        kernel_accumulate_block_avx512_nqx<NQ, SL>(
                nsq, codes, LUT, res, scaler);
    }
}

} // namespace faiss
