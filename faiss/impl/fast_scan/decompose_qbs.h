/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/fast_scan/kernels_simd256.h>
#include <faiss/impl/fast_scan/kernels_simd512.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

/*
 * Unified kernel: selects 256-bit vs 512-bit path based on
 * compile-time __AVX512F__ guard.
 */
template <int NQ, class ResultHandler, class Scaler>
void kernel_accumulate_block(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
#ifdef __AVX512F__
    pq4_kernel_qbs_512<NQ>(nsq, codes, LUT, res, scaler);
#else
    pq4_kernel_qbs_256<NQ>(nsq, codes, LUT, res, scaler);
#endif
}

// handle at most 4 blocks of queries
template <int QBS, class ResultHandler, class Scaler>
void accumulate_q_4step(
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT0,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    constexpr int Q1 = QBS & 15;
    constexpr int Q2 = (QBS >> 4) & 15;
    constexpr int Q3 = (QBS >> 8) & 15;
    constexpr int Q4 = (QBS >> 12) & 15;
    constexpr int SQ = Q1 + Q2 + Q3 + Q4;

    for (size_t j0 = 0; j0 < ntotal2; j0 += 32) {
        FixedStorageHandler<SQ, 2> res2;
        const uint8_t* LUT = LUT0;
        kernel_accumulate_block<Q1>(nsq, codes, LUT, res2, scaler);
        LUT += Q1 * nsq * 16;
        if (Q2 > 0) {
            res2.set_block_origin(Q1, 0);
            kernel_accumulate_block<Q2>(nsq, codes, LUT, res2, scaler);
            LUT += Q2 * nsq * 16;
        }
        if (Q3 > 0) {
            res2.set_block_origin(Q1 + Q2, 0);
            kernel_accumulate_block<Q3>(nsq, codes, LUT, res2, scaler);
            LUT += Q3 * nsq * 16;
        }
        if (Q4 > 0) {
            res2.set_block_origin(Q1 + Q2 + Q3, 0);
            kernel_accumulate_block<Q4>(nsq, codes, LUT, res2, scaler);
        }
        res.set_block_origin(0, j0);
        res2.to_other_handler(res);
        codes += block_stride;
    }
}

template <int NQ, class ResultHandler, class Scaler>
void kernel_accumulate_block_loop(
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    for (size_t j0 = 0; j0 < ntotal2; j0 += 32) {
        res.set_block_origin(0, j0);
        kernel_accumulate_block<NQ, ResultHandler>(
                nsq, codes, LUT, res, scaler);
        codes += block_stride;
    }
}

// non-template version of accumulate kernel -- dispatches dynamically
template <class ResultHandler, class Scaler>
void accumulate(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    assert(nsq % 2 == 0);
    assert(is_aligned_pointer(LUT));

#define DISPATCH(NQ)                                                  \
    case NQ:                                                          \
        kernel_accumulate_block_loop<NQ, ResultHandler>(              \
                ntotal2, nsq, codes, LUT, res, scaler, block_stride); \
        return

    switch (nq) {
        DISPATCH(1);
        DISPATCH(2);
        DISPATCH(3);
        DISPATCH(4);
    }
    FAISS_THROW_FMT("accumulate nq=%d not instantiated", nq);

#undef DISPATCH
}

template <class ResultHandler, class Scaler>
void pq4_accumulate_loop_qbs_fixed_scaler(
        int qbs,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT0,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride = 0) {
    assert(nsq % 2 == 0);
    assert(is_aligned_pointer(codes));
    assert(is_aligned_pointer(LUT0));

    // try out optimized versions
    switch (qbs) {
#define DISPATCH(QBS)                                                  \
    case QBS:                                                          \
        accumulate_q_4step<QBS>(                                       \
                ntotal2, nsq, codes, LUT0, res, scaler, block_stride); \
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
    for (size_t j0 = 0; j0 < ntotal2; j0 += 32) {
        const uint8_t* LUT = LUT0;
        int qi = qbs;
        int i0 = 0;
        while (qi) {
            int nq = qi & 15;
            qi >>= 4;
            res.set_block_origin(i0, j0);
#define DISPATCH(NQ)                                \
    case NQ:                                        \
        kernel_accumulate_block<NQ, ResultHandler>( \
                nsq, codes, LUT, res, scaler);      \
        break
            switch (nq) {
                DISPATCH(1);
                DISPATCH(2);
                DISPATCH(3);
                DISPATCH(4);
#undef DISPATCH
                default:
                    FAISS_THROW_FMT("accumulate nq=%d not instantiated", nq);
            }
            i0 += nq;
            LUT += nq * nsq * 16;
        }
        codes += block_stride;
    }
}

} // namespace faiss
