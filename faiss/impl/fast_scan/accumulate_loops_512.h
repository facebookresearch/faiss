/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file accumulate_loops_512.h
 * @brief 512-bit QBS accumulation loop for AVX512 per-ISA TUs.
 *
 * Mirrors accumulate_loops.h's QBS path but uses pq4_kernel_qbs_512
 * (from kernels_simd512.h) instead of pq4_kernel_qbs_256.
 *
 * The 512-bit kernels produce simd16uint16_tpl<AVX2> results (via
 * combine4x2). The virtual SIMDResultHandler::handle() expects
 * simd16uint16_tpl<NONE> in DD mode. FixedStorage512 bridges this gap:
 * it stores AVX2-level results internally, then converts to the handler's
 * level via storeu/load in to_other_handler().
 *
 * Only included from the AVX512 per-ISA TU (impl-avx512.cpp) via
 * dispatching.h's conditional include.
 */

#if defined(COMPILE_SIMD_AVX512) && defined(__AVX512F__)

#include <cassert>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/fast_scan/accumulate_loops.h>
#include <faiss/impl/fast_scan/kernels_simd512.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

/***************************************************************
 * FixedStorage512: non-virtual intermediate result storage
 * for 512-bit kernels.
 *
 * Does NOT inherit from SIMDResultHandler — the virtual handle()
 * signature is pinned to simd16uint16_tpl<NONE> in DD mode, but
 * 512-bit kernels produce simd16uint16_tpl<AVX2>. By avoiding
 * inheritance, handle() can accept AVX2-level types directly.
 *
 * The conversion to the outer handler's type happens in
 * to_other_handler() via a store-to-memory roundtrip.
 ***************************************************************/

template <int NQ, int BB>
struct FixedStorage512 {
    using simd16uint16_avx2 = simd16uint16_tpl<SIMDLevel::AVX2>;

    simd16uint16_avx2 dis[NQ][BB];
    int i0 = 0;

    void handle(
            size_t q,
            size_t b,
            simd16uint16_avx2 d0,
            simd16uint16_avx2 d1) {
        dis[q + i0][2 * b] = d0;
        dis[q + i0][2 * b + 1] = d1;
    }

    void set_block_origin(size_t i0_in, size_t) {
        this->i0 = i0_in;
    }

    template <class OtherResultHandler>
    void to_other_handler(OtherResultHandler& other) const {
        using handler_simd16 = simd16uint16_tpl<SINGLE_SIMD_LEVEL_256>;
        for (int q = 0; q < NQ; q++) {
            for (int b = 0; b < BB; b += 2) {
                // Convert AVX2 → handler level (NONE in DD mode)
                ALIGNED(32) uint16_t buf0[16], buf1[16];
                dis[q][b].storeu(buf0);
                dis[q][b + 1].storeu(buf1);
                handler_simd16 h0, h1;
                h0.loadu(buf0);
                h1.loadu(buf1);
                other.handle(q, b / 2, h0, h1);
            }
        }
    }
};

/***************************************************************
 * QBS path: 512-bit kernel variants
 ***************************************************************/

template <int QBS, class ResultHandler, class Scaler>
void accumulate_q_4step_512(
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
        FixedStorage512<SQ, 2> res2;
        const uint8_t* LUT = LUT0;
        pq4_kernel_qbs_512<Q1>(nsq, codes, LUT, res2, scaler);
        LUT += Q1 * nsq * 16;
        if (Q2 > 0) {
            res2.set_block_origin(Q1, 0);
            pq4_kernel_qbs_512<Q2>(nsq, codes, LUT, res2, scaler);
            LUT += Q2 * nsq * 16;
        }
        if (Q3 > 0) {
            res2.set_block_origin(Q1 + Q2, 0);
            pq4_kernel_qbs_512<Q3>(nsq, codes, LUT, res2, scaler);
            LUT += Q3 * nsq * 16;
        }
        if (Q4 > 0) {
            res2.set_block_origin(Q1 + Q2 + Q3, 0);
            pq4_kernel_qbs_512<Q4>(nsq, codes, LUT, res2, scaler);
        }
        res2.to_other_handler(res);
    });
}

template <class ResultHandler, class Scaler>
void pq4_accumulate_loop_qbs_fixed_scaler_512(
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
#define FAISS_QBS512_DISPATCH(QBS)                                     \
    case QBS:                                                          \
        accumulate_q_4step_512<QBS>(                                   \
                ntotal2, nsq, codes, LUT0, res, scaler, block_stride); \
        return;
        FAISS_QBS512_DISPATCH(0x3333); // 12
        FAISS_QBS512_DISPATCH(0x2333); // 11
        FAISS_QBS512_DISPATCH(0x2233); // 10
        FAISS_QBS512_DISPATCH(0x333);  // 9
        FAISS_QBS512_DISPATCH(0x2223); // 9
        FAISS_QBS512_DISPATCH(0x233);  // 8
        FAISS_QBS512_DISPATCH(0x1223); // 8
        FAISS_QBS512_DISPATCH(0x223);  // 7
        FAISS_QBS512_DISPATCH(0x34);   // 7
        FAISS_QBS512_DISPATCH(0x133);  // 7
        FAISS_QBS512_DISPATCH(0x6);    // 6
        FAISS_QBS512_DISPATCH(0x33);   // 6
        FAISS_QBS512_DISPATCH(0x123);  // 6
        FAISS_QBS512_DISPATCH(0x222);  // 6
        FAISS_QBS512_DISPATCH(0x23);   // 5
        FAISS_QBS512_DISPATCH(0x5);    // 5
        FAISS_QBS512_DISPATCH(0x13);   // 4
        FAISS_QBS512_DISPATCH(0x22);   // 4
        FAISS_QBS512_DISPATCH(0x4);    // 4
        FAISS_QBS512_DISPATCH(0x3);    // 3
        FAISS_QBS512_DISPATCH(0x21);   // 3
        FAISS_QBS512_DISPATCH(0x2);    // 2
        FAISS_QBS512_DISPATCH(0x1);    // 1
#undef FAISS_QBS512_DISPATCH
    }

    // Fallback for unknown QBS values: use 256-bit path with NONE-level
    // scalers for type compatibility. This is rare — pq4_preferred_qbs()
    // covers all values above.
    if constexpr (Scaler::nscale == 0) {
        DummyScaler<> scaler_none;
        pq4_accumulate_loop_qbs_fixed_scaler_256(
                qbs, ntotal2, nsq, codes, LUT0, res, scaler_none, block_stride);
    } else {
        NormTableScaler<> scaler_none(scaler.scale_int);
        pq4_accumulate_loop_qbs_fixed_scaler_256(
                qbs, ntotal2, nsq, codes, LUT0, res, scaler_none, block_stride);
    }
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX512 && __AVX512F__
