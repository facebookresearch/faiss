/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/aq4_fast_scan.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/simdlib.h>

namespace faiss {

using namespace simd_result_handlers;

/************************************************************
 * Accumulation functions
 ************************************************************/

namespace {

/*
 * The computation kernel
 * It accumulates results for NQ queries and 2 * 16 database elements
 * writes results in a ResultHandler
 */

template <int NQ, class ResultHandler>
void kernel_accumulate_block(
        int nsq,
        int nscale,
        int scale,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res) {
    // dummy alloc to keep the windows compiler happy
    constexpr int NQA = NQ > 0 ? NQ : 1;
    // distance accumulators
    simd16uint16 accu[NQA][4];

    simd16uint16 scale_factor(scale);

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
        }
    }

    // _mm_prefetch(codes + 768, 0);
    // sq: sub-quantizer
    for (int sq = 0; sq < nsq; sq += 2) {
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

            if (sq < nsq - nscale) {
                // 16 base elements
                accu[q][0] += simd16uint16(res0);
                accu[q][1] += simd16uint16(res0) >> 8;

                // the other 16 base elements
                accu[q][2] += simd16uint16(res1);
                accu[q][3] += simd16uint16(res1) >> 8;
            } else {
                // 16 base elements
                accu[q][0] += simd16uint16(res0) * scale_factor;
                accu[q][1] += (simd16uint16(res0) >> 8) * scale_factor;

                // the other 16 base elements
                accu[q][2] += simd16uint16(res1) * scale_factor;
                accu[q][3] += (simd16uint16(res1) >> 8) * scale_factor;
            }
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16 dis0 = combine2x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16 dis1 = combine2x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

// handle at most 4 blocks of queries
template <int QBS, class ResultHandler>
void accumulate_q_4step(
        size_t ntotal2,
        int nsq,
        int nscale,
        int scale,
        const uint8_t* codes,
        const uint8_t* LUT0,
        ResultHandler& res) {
    constexpr int Q1 = QBS & 15;
    constexpr int Q2 = (QBS >> 4) & 15;
    constexpr int Q3 = (QBS >> 8) & 15;
    constexpr int Q4 = (QBS >> 12) & 15;
    constexpr int SQ = Q1 + Q2 + Q3 + Q4;

    for (int64_t j0 = 0; j0 < ntotal2; j0 += 32) {
        FixedStorageHandler<SQ, 2> res2;
        const uint8_t* LUT = LUT0;
        kernel_accumulate_block<Q1>(nsq, nscale, scale, codes, LUT, res2);
        LUT += Q1 * nsq * 16;
        if (Q2 > 0) {
            res2.set_block_origin(Q1, 0);
            kernel_accumulate_block<Q2>(nsq, nscale, scale, codes, LUT, res2);
            LUT += Q2 * nsq * 16;
        }
        if (Q3 > 0) {
            res2.set_block_origin(Q1 + Q2, 0);
            kernel_accumulate_block<Q3>(nsq, nscale, scale, codes, LUT, res2);
            LUT += Q3 * nsq * 16;
        }
        if (Q4 > 0) {
            res2.set_block_origin(Q1 + Q2 + Q3, 0);
            kernel_accumulate_block<Q4>(nsq, nscale, scale, codes, LUT, res2);
        }
        res.set_block_origin(0, j0);
        res2.to_other_handler(res);
        codes += 32 * nsq / 2;
    }
}

template <int NQ, class ResultHandler>
void kernel_accumulate_block_loop(
        size_t ntotal2,
        int nsq,
        int nscale,
        int scale,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res) {
    for (int64_t j0 = 0; j0 < ntotal2; j0 += 32) {
        res.set_block_origin(0, j0);
        kernel_accumulate_block<NQ, ResultHandler>(
                nsq, nscale, scale, codes + j0 * nsq / 2, LUT, res);
    }
}

} // namespace

template <class ResultHandler>
void aq4_accumulate_loop_qbs(
        int qbs,
        size_t ntotal2,
        int nsq,
        int nscale,
        int scale,
        const uint8_t* codes,
        const uint8_t* LUT0,
        ResultHandler& res) {
    assert(nsq % 2 == 0);
    assert(is_aligned_pointer(codes));
    assert(is_aligned_pointer(LUT0));

    // try out optimized versions
    switch (qbs) {
#define DISPATCH(QBS)                                           \
    case QBS:                                                   \
        accumulate_q_4step<QBS>(                                \
                ntotal2, nsq, nscale, scale, codes, LUT0, res); \
        return;
        DISPATCH(0x3333); // 12
        DISPATCH(0x2333); // 11
        DISPATCH(0x2233); // 10
        DISPATCH(0x333);  // 9
        DISPATCH(0x2223); // 9
        DISPATCH(0x233);  // 8
        DISPATCH(0x1223); // 8
        DISPATCH(0x223);  // 7
        DISPATCH(0x34);   // 7
        DISPATCH(0x133);  // 7
        DISPATCH(0x6);    // 6
        DISPATCH(0x33);   // 6
        DISPATCH(0x123);  // 6
        DISPATCH(0x222);  // 6
        DISPATCH(0x23);   // 5
        DISPATCH(0x5);    // 5
        DISPATCH(0x13);   // 4
        DISPATCH(0x22);   // 4
        DISPATCH(0x4);    // 4
        DISPATCH(0x3);    // 3
        DISPATCH(0x21);   // 3
        DISPATCH(0x2);    // 2
        DISPATCH(0x1);    // 1
#undef DISPATCH
    }

    // default implementation where qbs is not known at compile time

    for (int64_t j0 = 0; j0 < ntotal2; j0 += 32) {
        const uint8_t* LUT = LUT0;
        int qi = qbs;
        int i0 = 0;
        while (qi) {
            int nq = qi & 15;
            qi >>= 4;
            res.set_block_origin(i0, j0);
#define DISPATCH(NQ)                                  \
    case NQ:                                          \
        kernel_accumulate_block<NQ, ResultHandler>(   \
                nsq, nscale, scale, codes, LUT, res); \
        break
            switch (nq) {
                DISPATCH(1);
                DISPATCH(2);
                DISPATCH(3);
                DISPATCH(4);
#undef DISPATCH
                default:
                    FAISS_THROW_FMT("accumulate nq=%d not instanciated", nq);
            }
            i0 += nq;
            LUT += nq * nsq * 16;
        }
        codes += 32 * nsq / 2;
    }
}

// explicit template instantiations

#define INSTANTIATE_ACCUMULATE_AQ(RH)          \
    template void aq4_accumulate_loop_qbs<RH>( \
            int, size_t, int, int, int, const uint8_t*, const uint8_t*, RH&);

using Csi = CMax<uint16_t, int>;
INSTANTIATE_ACCUMULATE_AQ(SingleResultHandler<Csi>)
INSTANTIATE_ACCUMULATE_AQ(HeapHandler<Csi>)
INSTANTIATE_ACCUMULATE_AQ(ReservoirHandler<Csi>)
using Csi2 = CMin<uint16_t, int>;
INSTANTIATE_ACCUMULATE_AQ(SingleResultHandler<Csi2>)
INSTANTIATE_ACCUMULATE_AQ(HeapHandler<Csi2>)
INSTANTIATE_ACCUMULATE_AQ(ReservoirHandler<Csi2>)

using Cfl = CMax<uint16_t, int64_t>;
using HHCsl = HeapHandler<Cfl, true>;
using RHCsl = ReservoirHandler<Cfl, true>;
using SHCsl = SingleResultHandler<Cfl, true>;
INSTANTIATE_ACCUMULATE_AQ(HHCsl)
INSTANTIATE_ACCUMULATE_AQ(RHCsl)
INSTANTIATE_ACCUMULATE_AQ(SHCsl)
using Cfl2 = CMin<uint16_t, int64_t>;
using HHCsl2 = HeapHandler<Cfl2, true>;
using RHCsl2 = ReservoirHandler<Cfl2, true>;
using SHCsl2 = SingleResultHandler<Cfl2, true>;
INSTANTIATE_ACCUMULATE_AQ(HHCsl2)
INSTANTIATE_ACCUMULATE_AQ(RHCsl2)
INSTANTIATE_ACCUMULATE_AQ(SHCsl2)

} // namespace faiss
