/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file accumulate_loops.h
 * @brief Shared accumulation loop helpers for fast-scan search paths.
 *
 * Contains:
 *   - accumulate_fixed_blocks / pq4_accumulate_loop_fixed_scaler
 *     (search_1 multi-BB path, bbs > 32)
 *   - accumulate_q_4step_256 / pq4_accumulate_loop_qbs_fixed_scaler_256
 *     (QBS path, bbs == 32, 256-bit kernel only)
 *
 * The QBS helpers use pq4_kernel_qbs_256 exclusively (not decompose_qbs.h)
 * because decompose_qbs.h includes kernels_simd512.h which uses 512-bit
 * types that are empty primary templates when SINGLE_SIMD_LEVEL=NONE
 * (DD mode). SL-parameterizing the 512-bit kernels is future work.
 *
 * All functions live in `namespace faiss` (not anonymous) so they can be
 * shared by both the per-SIMD TU dispatcher (dispatching.h) and the old
 * free-function search paths (pq4_fast_scan_search_1.cpp).
 *
 * The QBS helpers here always use pq4_kernel_qbs_256 (never 512-bit).
 * This is required for the per-SIMD DD TUs where SINGLE_SIMD_LEVEL=NONE
 * leaves 512-bit types empty.  The old pq4_fast_scan_search_qbs.cpp
 * continues to use decompose_qbs.h which includes both 256 and 512 paths.
 */

#include <cassert>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/fast_scan/LookupTableScaler.h>
#include <faiss/impl/fast_scan/kernels_simd256.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

/***************************************************************
 * Search_1 path helpers (multi-BB kernel, bbs > 32)
 ***************************************************************/

template <int NQ, int BB, class ResultHandler, class Scaler>
void accumulate_fixed_blocks(
        size_t nb,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    constexpr int bbs = 32 * BB;
    for_each_block<bbs>(nb, codes, block_stride, res, [&](size_t) {
        FixedStorageHandler<NQ, 2 * BB> res2;
        kernel_accumulate_block<NQ, BB>(nsq, codes, LUT, res2, scaler);
        res2.to_other_handler(res);
    });
}

template <class ResultHandler, class Scaler>
void pq4_accumulate_loop_fixed_scaler(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    FAISS_THROW_IF_NOT(is_aligned_pointer(codes));
    FAISS_THROW_IF_NOT(is_aligned_pointer(LUT));
    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    FAISS_THROW_IF_NOT(nb % bbs == 0);

#define FAISS_ACCLOOP_DISPATCH(NQ, BB)                           \
    case NQ * 1000 + BB:                                         \
        accumulate_fixed_blocks<NQ, BB>(                         \
                nb, nsq, codes, LUT, res, scaler, block_stride); \
        break

    switch (nq * 1000 + bbs / 32) {
        FAISS_ACCLOOP_DISPATCH(1, 1);
        FAISS_ACCLOOP_DISPATCH(1, 2);
        FAISS_ACCLOOP_DISPATCH(1, 3);
        FAISS_ACCLOOP_DISPATCH(1, 4);
        FAISS_ACCLOOP_DISPATCH(1, 5);
        FAISS_ACCLOOP_DISPATCH(2, 1);
        FAISS_ACCLOOP_DISPATCH(2, 2);
        FAISS_ACCLOOP_DISPATCH(3, 1);
        FAISS_ACCLOOP_DISPATCH(4, 1);
        default:
            FAISS_THROW_FMT("nq=%d bbs=%d not instantiated", nq, bbs);
    }
#undef FAISS_ACCLOOP_DISPATCH
}

/***************************************************************
 * QBS path helpers (bbs == 32, 256-bit kernel only)
 ***************************************************************/

template <int QBS, class ResultHandler, class Scaler>
void accumulate_q_4step_256(
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

    for_each_block<32>(ntotal2, codes, block_stride, res, [&](size_t) {
        FixedStorageHandler<SQ, 2> res2;
        const uint8_t* LUT = LUT0;
        pq4_kernel_qbs_256<Q1>(nsq, codes, LUT, res2, scaler);
        LUT += Q1 * nsq * 16;
        if (Q2 > 0) {
            res2.set_block_origin(Q1, 0);
            pq4_kernel_qbs_256<Q2>(nsq, codes, LUT, res2, scaler);
            LUT += Q2 * nsq * 16;
        }
        if (Q3 > 0) {
            res2.set_block_origin(Q1 + Q2, 0);
            pq4_kernel_qbs_256<Q3>(nsq, codes, LUT, res2, scaler);
            LUT += Q3 * nsq * 16;
        }
        if (Q4 > 0) {
            res2.set_block_origin(Q1 + Q2 + Q3, 0);
            pq4_kernel_qbs_256<Q4>(nsq, codes, LUT, res2, scaler);
        }
        res2.to_other_handler(res);
    });
}

template <class ResultHandler, class Scaler>
void pq4_accumulate_loop_qbs_fixed_scaler_256(
        int qbs,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT0,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    assert(nsq % 2 == 0);
    assert(is_aligned_pointer(codes));
    assert(is_aligned_pointer(LUT0));

    switch (qbs) {
#define FAISS_QBS256_DISPATCH(QBS)                                     \
    case QBS:                                                          \
        accumulate_q_4step_256<QBS>(                                   \
                ntotal2, nsq, codes, LUT0, res, scaler, block_stride); \
        return;
        FAISS_QBS256_DISPATCH(0x3333); // 12
        FAISS_QBS256_DISPATCH(0x2333); // 11
        FAISS_QBS256_DISPATCH(0x2233); // 10
        FAISS_QBS256_DISPATCH(0x333);  // 9
        FAISS_QBS256_DISPATCH(0x2223); // 9
        FAISS_QBS256_DISPATCH(0x233);  // 8
        FAISS_QBS256_DISPATCH(0x1223); // 8
        FAISS_QBS256_DISPATCH(0x223);  // 7
        FAISS_QBS256_DISPATCH(0x34);   // 7
        FAISS_QBS256_DISPATCH(0x133);  // 7
        FAISS_QBS256_DISPATCH(0x6);    // 6
        FAISS_QBS256_DISPATCH(0x33);   // 6
        FAISS_QBS256_DISPATCH(0x123);  // 6
        FAISS_QBS256_DISPATCH(0x222);  // 6
        FAISS_QBS256_DISPATCH(0x23);   // 5
        FAISS_QBS256_DISPATCH(0x5);    // 5
        FAISS_QBS256_DISPATCH(0x13);   // 4
        FAISS_QBS256_DISPATCH(0x22);   // 4
        FAISS_QBS256_DISPATCH(0x4);    // 4
        FAISS_QBS256_DISPATCH(0x3);    // 3
        FAISS_QBS256_DISPATCH(0x21);   // 3
        FAISS_QBS256_DISPATCH(0x2);    // 2
        FAISS_QBS256_DISPATCH(0x1);    // 1
#undef FAISS_QBS256_DISPATCH
    }

    // Default: qbs not known at compile time
    for_each_block<32>(ntotal2, codes, block_stride, res, [&](size_t j0) {
        const uint8_t* LUT = LUT0;
        int qi = qbs;
        int i0 = 0;
        while (qi) {
            int nq = qi & 15;
            qi >>= 4;
            res.set_block_origin(i0, j0);
#define FAISS_NQ256_DISPATCH(NQ)                              \
    case NQ:                                                  \
        pq4_kernel_qbs_256<NQ>(nsq, codes, LUT, res, scaler); \
        break
            switch (nq) {
                FAISS_NQ256_DISPATCH(1);
                FAISS_NQ256_DISPATCH(2);
                FAISS_NQ256_DISPATCH(3);
                FAISS_NQ256_DISPATCH(4);
#undef FAISS_NQ256_DISPATCH
                default:
                    FAISS_THROW_FMT("accumulate nq=%d not instantiated", nq);
            }
            i0 += nq;
            LUT += nq * nsq * 16;
        }
    });
}

} // namespace faiss
