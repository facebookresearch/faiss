/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/utils/simdlib.h>

namespace faiss {

/*
 * Multi-BB variant: accumulates NQ queries × BB*32 database elements.
 * Used by the search_1 path (bbs > 32).
 */
template <
        int NQ,
        int BB,
        SIMDLevel SL = SINGLE_SIMD_LEVEL_256,
        class ResultHandler,
        class Scaler>
void kernel_accumulate_block(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;
    // distance accumulators
    simd16uint16<SL256> accu[NQ][BB][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < BB; b++) {
            accu[q][b][0].clear();
            accu[q][b][1].clear();
            accu[q][b][2].clear();
            accu[q][b][3].clear();
        }
    }

    for (int sq = 0; sq < nsq - scaler.nscale; sq += 2) {
        simd32uint8<SL256> lut_cache[NQ];
        for (int q = 0; q < NQ; q++) {
            lut_cache[q] = simd32uint8<SL256>(LUT);
            LUT += 32;
        }

        for (int b = 0; b < BB; b++) {
            simd32uint8<SL256> c = simd32uint8<SL256>(codes);
            codes += 32;
            simd32uint8<SL256> mask(15);
            simd32uint8<SL256> chi =
                    simd32uint8<SL256>(simd16uint16<SL256>(c) >> 4) & mask;
            simd32uint8<SL256> clo = c & mask;

            for (int q = 0; q < NQ; q++) {
                simd32uint8<SL256> lut = lut_cache[q];
                simd32uint8<SL256> res0 = lut.lookup_2_lanes(clo);
                simd32uint8<SL256> res1 = lut.lookup_2_lanes(chi);

                accu[q][b][0] += simd16uint16<SL256>(res0);
                accu[q][b][1] += simd16uint16<SL256>(res0) >> 8;

                accu[q][b][2] += simd16uint16<SL256>(res1);
                accu[q][b][3] += simd16uint16<SL256>(res1) >> 8;
            }
        }
    }

    for (int sq = 0; sq < scaler.nscale; sq += 2) {
        simd32uint8<SL256> lut_cache[NQ];
        for (int q = 0; q < NQ; q++) {
            lut_cache[q] = simd32uint8<SL256>(LUT);
            LUT += 32;
        }

        for (int b = 0; b < BB; b++) {
            simd32uint8<SL256> c = simd32uint8<SL256>(codes);
            codes += 32;
            simd32uint8<SL256> mask(15);
            simd32uint8<SL256> chi =
                    simd32uint8<SL256>(simd16uint16<SL256>(c) >> 4) & mask;
            simd32uint8<SL256> clo = c & mask;

            for (int q = 0; q < NQ; q++) {
                simd32uint8<SL256> lut = lut_cache[q];

                simd32uint8<SL256> res0 = scaler.lookup(lut, clo);
                accu[q][b][0] += scaler.scale_lo(res0);
                accu[q][b][1] += scaler.scale_hi(res0);

                simd32uint8<SL256> res1 = scaler.lookup(lut, chi);
                accu[q][b][2] += scaler.scale_lo(res1);
                accu[q][b][3] += scaler.scale_hi(res1);
            }
        }
    }

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < BB; b++) {
            accu[q][b][0] -= accu[q][b][1] << 8;
            simd16uint16<SL256> dis0 = combine2x2(accu[q][b][0], accu[q][b][1]);

            accu[q][b][2] -= accu[q][b][3] << 8;
            simd16uint16<SL256> dis1 = combine2x2(accu[q][b][2], accu[q][b][3]);

            res.handle(q, b, dis0, dis1);
        }
    }
}

/*
 * Single-BB QBS variant: accumulates NQ queries × 32 db elements (BB=1).
 * Used by the decompose_qbs layer for non-AVX512 paths.
 */
template <
        int NQ,
        SIMDLevel SL = SINGLE_SIMD_LEVEL_256,
        class ResultHandler,
        class Scaler>
void pq4_kernel_qbs_256(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;
    // dummy alloc to keep the windows compiler happy
    constexpr int NQA = NQ > 0 ? NQ : 1;
    // distance accumulators
    // layout: accu[q][b]: distance accumulator for vectors 8*b..8*b+7
    simd16uint16<SL256> accu[NQA][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
        }
    }

    for (int sq = 0; sq < nsq - scaler.nscale; sq += 2) {
        simd32uint8<SL256> c;
        c.loadu(codes);
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

            accu[q][0] += simd16uint16<SL256>(res0);
            accu[q][1] += simd16uint16<SL256>(res0) >> 8;

            accu[q][2] += simd16uint16<SL256>(res1);
            accu[q][3] += simd16uint16<SL256>(res1) >> 8;
        }
    }

    for (int sq = 0; sq < scaler.nscale; sq += 2) {
        simd32uint8<SL256> c;
        c.loadu(codes);
        codes += 32;

        simd32uint8<SL256> mask(0xf);
        simd32uint8<SL256> chi =
                simd32uint8<SL256>(simd16uint16<SL256>(c) >> 4) & mask;
        simd32uint8<SL256> clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            simd32uint8<SL256> lut(LUT);
            LUT += 32;

            simd32uint8<SL256> res0 = scaler.lookup(lut, clo);
            accu[q][0] += scaler.scale_lo(res0);
            accu[q][1] += scaler.scale_hi(res0);

            simd32uint8<SL256> res1 = scaler.lookup(lut, chi);
            accu[q][2] += scaler.scale_lo(res1);
            accu[q][3] += scaler.scale_hi(res1);
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16<SL256> dis0 = combine2x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16<SL256> dis1 = combine2x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

} // namespace faiss
