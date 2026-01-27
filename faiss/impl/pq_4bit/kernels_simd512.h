/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/pq_4bit/pq4_fast_scan.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/pq_4bit/decompose_qbs.h>
#include <faiss/impl/pq_4bit/LookupTableScaler.h>
#include <faiss/impl/pq_4bit/kernels_common.h>
#include <faiss/impl/pq_4bit/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

// Type aliases for explicit SIMD levels in AVX512 kernel
// 512-bit types (native AVX512)
using simd64uint8_avx512 = simd64uint8<SIMDLevel::AVX512>;
using simd32uint16_avx512 = simd32uint16<SIMDLevel::AVX512>;
// 256-bit types (AVX2, used in fallback paths)
using simd32uint8_avx2 = simd32uint8<SIMDLevel::AVX2>;
using simd16uint16_avx2 = simd16uint16<SIMDLevel::AVX2>;

/***************************************************************
 * accumulation functions -- simplified for bbs=32
 * (AVX512-optimized versions)
 ***************************************************************/

// a special version for NQ=1.
// Despite the function being large in the text form, it compiles to a very
//    compact assembler code.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <class ResultHandler, class Scaler>
void kernel_accumulate_block_avx512_nq1(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    // NQ is kept in order to match the similarity to baseline function
    constexpr int NQ = 1;
    // distance accumulators. We can accept more for NQ=1
    // layout: accu[q][b]: distance accumulator for vectors 32*b..32*b+15
    simd32uint16_avx512 accu[NQ][4];
    // layout: accu[q][b]: distance accumulator for vectors 32*b+16..32*b+31
    simd32uint16_avx512 accu1[NQ][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
            accu1[q][b].clear();
        }
    }

    // process "nsq - scaler.nscale" part
    const int nsq_minus_nscale = nsq - scaler.nscale;
    const int nsq_minus_nscale_8 = (nsq_minus_nscale / 8) * 8;
    const int nsq_minus_nscale_4 = (nsq_minus_nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nsq_minus_nscale_8; sq += 8) {
        // prefetch
        simd64uint8_avx512 c(codes);
        codes += 64;

        simd64uint8_avx512 c1(codes);
        codes += 64;

        simd64uint8_avx512 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8_avx512 chi =
                simd64uint8_avx512(simd32uint16_avx512(c) >> 4) & mask;
        simd64uint8_avx512 clo = c & mask;

        simd64uint8_avx512 c1hi =
                simd64uint8_avx512(simd32uint16_avx512(c1) >> 4) & mask;
        simd64uint8_avx512 c1lo = c1 & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8_avx512 lut(LUT);
            LUT += 64;

            {
                simd64uint8_avx512 res0 = lut.lookup_4_lanes(clo);
                simd64uint8_avx512 res1 = lut.lookup_4_lanes(chi);

                accu[q][0] += simd32uint16_avx512(res0);
                accu[q][1] += simd32uint16_avx512(res0) >> 8;

                accu[q][2] += simd32uint16_avx512(res1);
                accu[q][3] += simd32uint16_avx512(res1) >> 8;
            }
        }

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8_avx512 lut(LUT);
            LUT += 64;

            {
                simd64uint8_avx512 res0 = lut.lookup_4_lanes(c1lo);
                simd64uint8_avx512 res1 = lut.lookup_4_lanes(c1hi);

                accu1[q][0] += simd32uint16_avx512(res0);
                accu1[q][1] += simd32uint16_avx512(res0) >> 8;

                accu1[q][2] += simd32uint16_avx512(res1);
                accu1[q][3] += simd32uint16_avx512(res1) >> 8;
            }
        }
    }

    // process leftovers: a single chunk of size 4
    if (nsq_minus_nscale_8 != nsq_minus_nscale_4) {
        // prefetch
        simd64uint8_avx512 c(codes);
        codes += 64;

        simd64uint8_avx512 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8_avx512 chi =
                simd64uint8_avx512(simd32uint16_avx512(c) >> 4) & mask;
        simd64uint8_avx512 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8_avx512 lut(LUT);
            LUT += 64;

            simd64uint8_avx512 res0 = lut.lookup_4_lanes(clo);
            simd64uint8_avx512 res1 = lut.lookup_4_lanes(chi);

            accu[q][0] += simd32uint16_avx512(res0);
            accu[q][1] += simd32uint16_avx512(res0) >> 8;

            accu[q][2] += simd32uint16_avx512(res1);
            accu[q][3] += simd32uint16_avx512(res1) >> 8;
        }
    }

    // process leftovers: a single chunk of size 2
    if (nsq_minus_nscale_4 != nsq_minus_nscale) {
        // prefetch
        simd32uint8_avx2 c(codes);
        codes += 32;

        simd32uint8_avx2 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8_avx2 chi =
                simd32uint8_avx2(simd16uint16_avx2(c) >> 4) & mask;
        simd32uint8_avx2 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8_avx2 lut(LUT);
            LUT += 32;

            simd32uint8_avx2 res0 = lut.lookup_2_lanes(clo);
            simd32uint8_avx2 res1 = lut.lookup_2_lanes(chi);

            accu[q][0] += simd32uint16_avx512(simd16uint16_avx2(res0));
            accu[q][1] += simd32uint16_avx512(simd16uint16_avx2(res0) >> 8);

            accu[q][2] += simd32uint16_avx512(simd16uint16_avx2(res1));
            accu[q][3] += simd32uint16_avx512(simd16uint16_avx2(res1) >> 8);
        }
    }

    // process "sq" part
    const int nscale = scaler.nscale;
    const int nscale_8 = (nscale / 8) * 8;
    const int nscale_4 = (nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nscale_8; sq += 8) {
        // prefetch
        simd64uint8_avx512 c(codes);
        codes += 64;

        simd64uint8_avx512 c1(codes);
        codes += 64;

        simd64uint8_avx512 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8_avx512 chi =
                simd64uint8_avx512(simd32uint16_avx512(c) >> 4) & mask;
        simd64uint8_avx512 clo = c & mask;

        simd64uint8_avx512 c1hi =
                simd64uint8_avx512(simd32uint16_avx512(c1) >> 4) & mask;
        simd64uint8_avx512 c1lo = c1 & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8_avx512 lut(LUT);
            LUT += 64;

            {
                simd64uint8_avx512 res0 = scaler.lookup(lut, clo);
                accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..15
                accu[q][1] += scaler.scale_hi(res0); // handle vectors 16..31

                simd64uint8_avx512 res1 = scaler.lookup(lut, chi);
                accu[q][2] += scaler.scale_lo(res1); // handle vectors 32..47
                accu[q][3] += scaler.scale_hi(res1); //  handle vectors 48..63
            }
        }

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8_avx512 lut(LUT);
            LUT += 64;

            {
                simd64uint8_avx512 res0 = scaler.lookup(lut, c1lo);
                accu1[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
                accu1[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

                simd64uint8_avx512 res1 = scaler.lookup(lut, c1hi);
                accu1[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
                accu1[q][3] += scaler.scale_hi(res1); //  handle vectors 24..31
            }
        }
    }

    // process leftovers: a single chunk of size 4
    if (nscale_8 != nscale_4) {
        // prefetch
        simd64uint8_avx512 c(codes);
        codes += 64;

        simd64uint8_avx512 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8_avx512 chi =
                simd64uint8_avx512(simd32uint16_avx512(c) >> 4) & mask;
        simd64uint8_avx512 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8_avx512 lut(LUT);
            LUT += 64;

            simd64uint8_avx512 res0 = scaler.lookup(lut, clo);
            accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..15
            accu[q][1] += scaler.scale_hi(res0); // handle vectors 16..31

            simd64uint8_avx512 res1 = scaler.lookup(lut, chi);
            accu[q][2] += scaler.scale_lo(res1); // handle vectors 32..47
            accu[q][3] += scaler.scale_hi(res1); //  handle vectors 48..63
        }
    }

    // process leftovers: a single chunk of size 2
    if (nscale_4 != nscale) {
        // prefetch
        simd32uint8_avx2 c(codes);
        codes += 32;

        simd32uint8_avx2 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8_avx2 chi =
                simd32uint8_avx2(simd16uint16_avx2(c) >> 4) & mask;
        simd32uint8_avx2 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8_avx2 lut(LUT);
            LUT += 32;

            simd32uint8_avx2 res0 = scaler.lookup(lut, clo);
            accu[q][0] += simd32uint16_avx512(
                    scaler.scale_lo(res0)); // handle vectors 0..7
            accu[q][1] += simd32uint16_avx512(
                    scaler.scale_hi(res0)); // handle vectors 8..15

            simd32uint8_avx2 res1 = scaler.lookup(lut, chi);
            accu[q][2] += simd32uint16_avx512(
                    scaler.scale_lo(res1)); // handle vectors 16..23
            accu[q][3] += simd32uint16_avx512(
                    scaler.scale_hi(res1)); //  handle vectors 24..31
        }
    }

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b] += accu1[q][b];
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16_avx2 dis0 = combine4x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16_avx2 dis1 = combine4x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

// general-purpose case
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <int NQ, class ResultHandler, class Scaler>
void kernel_accumulate_block_avx512_nqx(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    // dummy alloc to keep the windows compiler happy
    constexpr int NQA = NQ > 0 ? NQ : 1;
    // distance accumulators
    // layout: accu[q][b]: distance accumulator for vectors 8*b..8*b+7
    simd32uint16_avx512 accu[NQA][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
        }
    }

    // process "nsq - scaler.nscale" part
    const int nsq_minus_nscale = nsq - scaler.nscale;
    const int nsq_minus_nscale_4 = (nsq_minus_nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nsq_minus_nscale_4; sq += 4) {
        // prefetch
        simd64uint8_avx512 c(codes);
        codes += 64;

        simd64uint8_avx512 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8_avx512 chi =
                simd64uint8_avx512(simd32uint16_avx512(c) >> 4) & mask;
        simd64uint8_avx512 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd32uint8_avx2 lut_a(LUT);
            simd32uint8_avx2 lut_b(LUT + NQ * 32);

            simd64uint8_avx512 lut(lut_a, lut_b);
            LUT += 32;

            {
                simd64uint8_avx512 res0 = lut.lookup_4_lanes(clo);
                simd64uint8_avx512 res1 = lut.lookup_4_lanes(chi);

                accu[q][0] += simd32uint16_avx512(res0);
                accu[q][1] += simd32uint16_avx512(res0) >> 8;

                accu[q][2] += simd32uint16_avx512(res1);
                accu[q][3] += simd32uint16_avx512(res1) >> 8;
            }
        }

        LUT += NQ * 32;
    }

    // process leftovers: a single chunk of size 2
    if (nsq_minus_nscale_4 != nsq_minus_nscale) {
        // prefetch
        simd32uint8_avx2 c(codes);
        codes += 32;

        simd32uint8_avx2 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8_avx2 chi =
                simd32uint8_avx2(simd16uint16_avx2(c) >> 4) & mask;
        simd32uint8_avx2 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8_avx2 lut(LUT);
            LUT += 32;

            simd32uint8_avx2 res0 = lut.lookup_2_lanes(clo);
            simd32uint8_avx2 res1 = lut.lookup_2_lanes(chi);

            accu[q][0] += simd32uint16_avx512(simd16uint16_avx2(res0));
            accu[q][1] += simd32uint16_avx512(simd16uint16_avx2(res0) >> 8);

            accu[q][2] += simd32uint16_avx512(simd16uint16_avx2(res1));
            accu[q][3] += simd32uint16_avx512(simd16uint16_avx2(res1) >> 8);
        }
    }

    // process "sq" part
    const int nscale = scaler.nscale;
    const int nscale_4 = (nscale / 4) * 4;

    // process in chunks of 4
    for (int sq = 0; sq < nscale_4; sq += 4) {
        // prefetch
        simd64uint8_avx512 c(codes);
        codes += 64;

        simd64uint8_avx512 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8_avx512 chi =
                simd64uint8_avx512(simd32uint16_avx512(c) >> 4) & mask;
        simd64uint8_avx512 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd32uint8_avx2 lut_a(LUT);
            simd32uint8_avx2 lut_b(LUT + NQ * 32);

            simd64uint8_avx512 lut(lut_a, lut_b);
            LUT += 32;

            {
                simd64uint8_avx512 res0 = scaler.lookup(lut, clo);
                accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
                accu[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

                simd64uint8_avx512 res1 = scaler.lookup(lut, chi);
                accu[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
                accu[q][3] += scaler.scale_hi(res1); //  handle vectors 24..31
            }
        }

        LUT += NQ * 32;
    }

    // process leftovers: a single chunk of size 2
    if (nscale_4 != nscale) {
        // prefetch
        simd32uint8_avx2 c(codes);
        codes += 32;

        simd32uint8_avx2 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8_avx2 chi =
                simd32uint8_avx2(simd16uint16_avx2(c) >> 4) & mask;
        simd32uint8_avx2 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8_avx2 lut(LUT);
            LUT += 32;

            simd32uint8_avx2 res0 = scaler.lookup(lut, clo);
            accu[q][0] += simd32uint16_avx512(
                    scaler.scale_lo(res0)); // handle vectors 0..7
            accu[q][1] += simd32uint16_avx512(
                    scaler.scale_hi(res0)); // handle vectors 8..15

            simd32uint8_avx2 res1 = scaler.lookup(lut, chi);
            accu[q][2] += simd32uint16_avx512(
                    scaler.scale_lo(res1)); // handle vectors 16..23
            accu[q][3] += simd32uint16_avx512(
                    scaler.scale_hi(res1)); //  handle vectors 24..31
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16_avx2 dis0 = combine4x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16_avx2 dis1 = combine4x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

template <int NQ, class ResultHandler, class Scaler>
void kernel_accumulate_block_avx512(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    if constexpr (NQ == 1) {
        kernel_accumulate_block_avx512_nq1<ResultHandler, Scaler>(
                nsq, codes, LUT, res, scaler);
    } else {
        kernel_accumulate_block_avx512_nqx<NQ, ResultHandler, Scaler>(
                nsq, codes, LUT, res, scaler);
    }
}

/*
kernel_accumulate_block_avx512_nq1 --- is the only specialized avx512 kernel and
others functions can be routed to avx2 kernel.

    to remove:
        kernel_accumulate_block_bb
        accumulate_fixed_blocks_bb
        pq4_accumulate_loop_fixed_scaler
*/
} // namespace faiss
