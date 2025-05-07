/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <faiss/impl/pq_4bit/pq4_fast_scan.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/pq_4bit/LookupTableScaler.h>
#include <faiss/impl/pq_4bit/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

/***************************************************************
 * accumulation functions -- for bbs not necessarily 32
 * These functions are not specialized for 512 bit SIMD so the code is just
 * copied from kernels_simd256.h
 ***************************************************************/

/*
 * The computation kernel
 * It accumulates results for NQ queries and BB * 32 database elements
 * writes results in a ResultHandler.
 */

template <int NQ, int BB, class ResultHandler, class Scaler>
void kernel_accumulate_block_bb(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
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
            simd32uint8 mask(15);
            simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
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
            simd32uint8 mask(15);
            simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
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

template <int NQ, int BB, SIMDLevel SL, class ResultHandler, class Scaler>
void accumulate_fixed_blocks_bb(
        size_t nb,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    constexpr int bbs = 32 * BB;
    for (size_t j0 = 0; j0 < nb; j0 += bbs) {
        FixedStorageHandler<NQ, 2 * BB, SL> res2;
        kernel_accumulate_block_bb<NQ, BB>(nsq, codes, LUT, res2, scaler);
        res.set_block_origin(0, j0);
        res2.to_other_handler(res);
        codes += bbs * nsq / 2;
    }
}

template <SIMDLevel SL, class ResultHandler, class Scaler>
void pq4_accumulate_loop_fixed_scaler(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    FAISS_THROW_IF_NOT(is_aligned_pointer(codes));
    FAISS_THROW_IF_NOT(is_aligned_pointer(LUT));
    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    FAISS_THROW_IF_NOT(nb % bbs == 0);

#define DISPATCH(NQ, BB)                           \
    case NQ * 1000 + BB:                           \
        accumulate_fixed_blocks_bb<NQ, BB, SL>(    \
                nb, nsq, codes, LUT, res, scaler); \
        break

    switch (nq * 1000 + bbs / 32) {
        DISPATCH(1, 1);
        DISPATCH(1, 2);
        DISPATCH(1, 3);
        DISPATCH(1, 4);
        DISPATCH(1, 5);
        DISPATCH(2, 1);
        DISPATCH(2, 2);
        DISPATCH(3, 1);
        DISPATCH(4, 1);
        default:
            FAISS_THROW_FMT("nq=%d bbs=%d not instantiated", nq, bbs);
    }
#undef DISPATCH
}

/***************************************************************
 * accumulation functions -- simplified for bbs=32
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
    simd32uint16 accu[NQ][4];
    // layout: accu[q][b]: distance accumulator for vectors 32*b+16..32*b+31
    simd32uint16 accu1[NQ][4];

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
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 c1(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        simd64uint8 c1hi = simd64uint8(simd32uint16(c1) >> 4) & mask;
        simd64uint8 c1lo = c1 & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            {
                simd64uint8 res0 = lut.lookup_4_lanes(clo);
                simd64uint8 res1 = lut.lookup_4_lanes(chi);

                accu[q][0] += simd32uint16(res0);
                accu[q][1] += simd32uint16(res0) >> 8;

                accu[q][2] += simd32uint16(res1);
                accu[q][3] += simd32uint16(res1) >> 8;
            }
        }

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            {
                simd64uint8 res0 = lut.lookup_4_lanes(c1lo);
                simd64uint8 res1 = lut.lookup_4_lanes(c1hi);

                accu1[q][0] += simd32uint16(res0);
                accu1[q][1] += simd32uint16(res0) >> 8;

                accu1[q][2] += simd32uint16(res1);
                accu1[q][3] += simd32uint16(res1) >> 8;
            }
        }
    }

    // process leftovers: a single chunk of size 4
    if (nsq_minus_nscale_8 != nsq_minus_nscale_4) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            simd64uint8 res0 = lut.lookup_4_lanes(clo);
            simd64uint8 res1 = lut.lookup_4_lanes(chi);

            accu[q][0] += simd32uint16(res0);
            accu[q][1] += simd32uint16(res0) >> 8;

            accu[q][2] += simd32uint16(res1);
            accu[q][3] += simd32uint16(res1) >> 8;
        }
    }

    // process leftovers: a single chunk of size 2
    if (nsq_minus_nscale_4 != nsq_minus_nscale) {
        // prefetch
        simd32uint8 c(codes);
        codes += 32;

        simd32uint8 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
        simd32uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8 lut(LUT);
            LUT += 32;

            simd32uint8 res0 = lut.lookup_2_lanes(clo);
            simd32uint8 res1 = lut.lookup_2_lanes(chi);

            accu[q][0] += simd32uint16(simd16uint16(res0));
            accu[q][1] += simd32uint16(simd16uint16(res0) >> 8);

            accu[q][2] += simd32uint16(simd16uint16(res1));
            accu[q][3] += simd32uint16(simd16uint16(res1) >> 8);
        }
    }

    // process "sq" part
    const int nscale = scaler.nscale;
    const int nscale_8 = (nscale / 8) * 8;
    const int nscale_4 = (nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nscale_8; sq += 8) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 c1(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        simd64uint8 c1hi = simd64uint8(simd32uint16(c1) >> 4) & mask;
        simd64uint8 c1lo = c1 & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            {
                simd64uint8 res0 = scaler.lookup(lut, clo);
                accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..15
                accu[q][1] += scaler.scale_hi(res0); // handle vectors 16..31

                simd64uint8 res1 = scaler.lookup(lut, chi);
                accu[q][2] += scaler.scale_lo(res1); // handle vectors 32..47
                accu[q][3] += scaler.scale_hi(res1); //  handle vectors 48..63
            }
        }

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            {
                simd64uint8 res0 = scaler.lookup(lut, c1lo);
                accu1[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
                accu1[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

                simd64uint8 res1 = scaler.lookup(lut, c1hi);
                accu1[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
                accu1[q][3] += scaler.scale_hi(res1); //  handle vectors 24..31
            }
        }
    }

    // process leftovers: a single chunk of size 4
    if (nscale_8 != nscale_4) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            simd64uint8 res0 = scaler.lookup(lut, clo);
            accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..15
            accu[q][1] += scaler.scale_hi(res0); // handle vectors 16..31

            simd64uint8 res1 = scaler.lookup(lut, chi);
            accu[q][2] += scaler.scale_lo(res1); // handle vectors 32..47
            accu[q][3] += scaler.scale_hi(res1); //  handle vectors 48..63
        }
    }

    // process leftovers: a single chunk of size 2
    if (nscale_4 != nscale) {
        // prefetch
        simd32uint8 c(codes);
        codes += 32;

        simd32uint8 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
        simd32uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8 lut(LUT);
            LUT += 32;

            simd32uint8 res0 = scaler.lookup(lut, clo);
            accu[q][0] +=
                    simd32uint16(scaler.scale_lo(res0)); // handle vectors 0..7
            accu[q][1] +=
                    simd32uint16(scaler.scale_hi(res0)); // handle vectors 8..15

            simd32uint8 res1 = scaler.lookup(lut, chi);
            accu[q][2] += simd32uint16(
                    scaler.scale_lo(res1)); // handle vectors 16..23
            accu[q][3] += simd32uint16(
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
        simd16uint16 dis0 = combine4x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16 dis1 = combine4x2(accu[q][2], accu[q][3]);
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
    simd32uint16 accu[NQA][4];

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
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd32uint8 lut_a(LUT);
            simd32uint8 lut_b(LUT + NQ * 32);

            simd64uint8 lut(lut_a, lut_b);
            LUT += 32;

            {
                simd64uint8 res0 = lut.lookup_4_lanes(clo);
                simd64uint8 res1 = lut.lookup_4_lanes(chi);

                accu[q][0] += simd32uint16(res0);
                accu[q][1] += simd32uint16(res0) >> 8;

                accu[q][2] += simd32uint16(res1);
                accu[q][3] += simd32uint16(res1) >> 8;
            }
        }

        LUT += NQ * 32;
    }

    // process leftovers: a single chunk of size 2
    if (nsq_minus_nscale_4 != nsq_minus_nscale) {
        // prefetch
        simd32uint8 c(codes);
        codes += 32;

        simd32uint8 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
        simd32uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8 lut(LUT);
            LUT += 32;

            simd32uint8 res0 = lut.lookup_2_lanes(clo);
            simd32uint8 res1 = lut.lookup_2_lanes(chi);

            accu[q][0] += simd32uint16(simd16uint16(res0));
            accu[q][1] += simd32uint16(simd16uint16(res0) >> 8);

            accu[q][2] += simd32uint16(simd16uint16(res1));
            accu[q][3] += simd32uint16(simd16uint16(res1) >> 8);
        }
    }

    // process "sq" part
    const int nscale = scaler.nscale;
    const int nscale_4 = (nscale / 4) * 4;

    // process in chunks of 4
    for (int sq = 0; sq < nscale_4; sq += 4) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd32uint8 lut_a(LUT);
            simd32uint8 lut_b(LUT + NQ * 32);

            simd64uint8 lut(lut_a, lut_b);
            LUT += 32;

            {
                simd64uint8 res0 = scaler.lookup(lut, clo);
                accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
                accu[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

                simd64uint8 res1 = scaler.lookup(lut, chi);
                accu[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
                accu[q][3] += scaler.scale_hi(res1); //  handle vectors 24..31
            }
        }

        LUT += NQ * 32;
    }

    // process leftovers: a single chunk of size 2
    if (nscale_4 != nscale) {
        // prefetch
        simd32uint8 c(codes);
        codes += 32;

        simd32uint8 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
        simd32uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8 lut(LUT);
            LUT += 32;

            simd32uint8 res0 = scaler.lookup(lut, clo);
            accu[q][0] +=
                    simd32uint16(scaler.scale_lo(res0)); // handle vectors 0..7
            accu[q][1] +=
                    simd32uint16(scaler.scale_hi(res0)); // handle vectors 8..15

            simd32uint8 res1 = scaler.lookup(lut, chi);
            accu[q][2] += simd32uint16(
                    scaler.scale_lo(res1)); // handle vectors 16..23
            accu[q][3] += simd32uint16(
                    scaler.scale_hi(res1)); //  handle vectors 24..31
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16 dis0 = combine4x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16 dis1 = combine4x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

template <int NQ, class ResultHandler, class Scaler>
void kernel_accumulate_block(
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

#include <faiss/impl/pq_4bit/decompose_qbs.h>

} // namespace faiss
