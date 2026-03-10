/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file dispatching.h
 * @brief Per-SIMD TU dispatch template for fast scan.
 *
 * This header is included once per SIMD TU with THE_LEVEL_TO_DISPATCH
 * set to the desired SIMDLevel. It provides:
 *   - ScannerMixIn: wraps a handler + calls kernel at the TU's SIMD level
 *   - make_fast_scan_scanner_impl<SL>: factory specialization
 *
 * Usage (in a per-SIMD .cpp file):
 *   #define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX2
 *   #include <faiss/impl/fast_scan/dispatching.h>
 *
 * NOTE: We intentionally do NOT include decompose_qbs.h here because
 * in DD mode, the AVX512 TU sets __AVX512F__ which triggers 512-bit
 * kernel paths, but SINGLE_SIMD_LEVEL is NONE and 512-bit NONE types
 * are empty primary templates. Instead, we provide local QBS
 * accumulation using only the 256-bit kernel (pq4_kernel_qbs_256).
 */

#ifndef THE_LEVEL_TO_DISPATCH
#error "Define THE_LEVEL_TO_DISPATCH before including this header"
#endif

#include <cassert>
#include <memory>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/fast_scan/LookupTableScaler.h>
#include <faiss/impl/fast_scan/kernels_simd256.h>
#include <faiss/impl/fast_scan/pq4_fast_scan.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

namespace {

constexpr SIMDLevel FS_SL = THE_LEVEL_TO_DISPATCH;

/***************************************************************
 * Search_1 path helpers (multi-BB kernel, bbs > 32)
 * These mirror pq4_fast_scan_search_1.cpp but live here so that
 * each per-SIMD TU gets its own copy compiled with the right flags.
 ***************************************************************/

template <int NQ, int BB, class ResultHandler, class Scaler>
void fs_accumulate_fixed_blocks(
        size_t nb,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    constexpr int bbs = 32 * BB;
    for (size_t j0 = 0; j0 < nb; j0 += bbs) {
        FixedStorageHandler<NQ, 2 * BB> res2;
        kernel_accumulate_block<NQ, BB>(nsq, codes, LUT, res2, scaler);
        res.set_block_origin(0, j0);
        res2.to_other_handler(res);
        codes += block_stride;
    }
}

template <class ResultHandler, class Scaler>
void fs_pq4_accumulate_loop_fixed_scaler(
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

#define FS_DISPATCH(NQ, BB)                                      \
    case NQ * 1000 + BB:                                         \
        fs_accumulate_fixed_blocks<NQ, BB>(                      \
                nb, nsq, codes, LUT, res, scaler, block_stride); \
        break

    switch (nq * 1000 + bbs / 32) {
        FS_DISPATCH(1, 1);
        FS_DISPATCH(1, 2);
        FS_DISPATCH(1, 3);
        FS_DISPATCH(1, 4);
        FS_DISPATCH(1, 5);
        FS_DISPATCH(2, 1);
        FS_DISPATCH(2, 2);
        FS_DISPATCH(3, 1);
        FS_DISPATCH(4, 1);
        default:
            FAISS_THROW_FMT("nq=%d bbs=%d not instantiated", nq, bbs);
    }
#undef FS_DISPATCH
}

/***************************************************************
 * QBS path helpers (bbs == 32, 256-bit kernel only)
 *
 * These mirror decompose_qbs.h but always use pq4_kernel_qbs_256
 * to avoid instantiating 512-bit types with SINGLE_SIMD_LEVEL=NONE.
 * When kernels are SL-parameterized (future diff), this can switch
 * to using the full decompose_qbs.h with proper 512-bit dispatch.
 ***************************************************************/

template <int QBS, class ResultHandler, class Scaler>
void fs_accumulate_q_4step(
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
        res.set_block_origin(0, j0);
        res2.to_other_handler(res);
        codes += block_stride;
    }
}

template <int NQ, class ResultHandler, class Scaler>
void fs_kernel_accumulate_block_loop(
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    for (size_t j0 = 0; j0 < ntotal2; j0 += 32) {
        res.set_block_origin(0, j0);
        pq4_kernel_qbs_256<NQ>(nsq, codes, LUT, res, scaler);
        codes += block_stride;
    }
}

template <class ResultHandler, class Scaler>
void fs_accumulate(
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

#define FS_NQ_DISPATCH(NQ)                                            \
    case NQ:                                                          \
        fs_kernel_accumulate_block_loop<NQ>(                          \
                ntotal2, nsq, codes, LUT, res, scaler, block_stride); \
        return

    switch (nq) {
        FS_NQ_DISPATCH(1);
        FS_NQ_DISPATCH(2);
        FS_NQ_DISPATCH(3);
        FS_NQ_DISPATCH(4);
    }
    FAISS_THROW_FMT("accumulate nq=%d not instantiated", nq);

#undef FS_NQ_DISPATCH
}

template <class ResultHandler, class Scaler>
void fs_pq4_accumulate_loop_qbs_fixed_scaler(
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
#define FS_QBS_DISPATCH(QBS)                                           \
    case QBS:                                                          \
        fs_accumulate_q_4step<QBS>(                                    \
                ntotal2, nsq, codes, LUT0, res, scaler, block_stride); \
        return;
        FS_QBS_DISPATCH(0x3333); // 12
        FS_QBS_DISPATCH(0x2333); // 11
        FS_QBS_DISPATCH(0x2233); // 10
        FS_QBS_DISPATCH(0x333);  // 9
        FS_QBS_DISPATCH(0x2223); // 9
        FS_QBS_DISPATCH(0x233);  // 8
        FS_QBS_DISPATCH(0x1223); // 8
        FS_QBS_DISPATCH(0x223);  // 7
        FS_QBS_DISPATCH(0x34);   // 7
        FS_QBS_DISPATCH(0x133);  // 7
        FS_QBS_DISPATCH(0x6);    // 6
        FS_QBS_DISPATCH(0x33);   // 6
        FS_QBS_DISPATCH(0x123);  // 6
        FS_QBS_DISPATCH(0x222);  // 6
        FS_QBS_DISPATCH(0x23);   // 5
        FS_QBS_DISPATCH(0x5);    // 5
        FS_QBS_DISPATCH(0x13);   // 4
        FS_QBS_DISPATCH(0x22);   // 4
        FS_QBS_DISPATCH(0x4);    // 4
        FS_QBS_DISPATCH(0x3);    // 3
        FS_QBS_DISPATCH(0x21);   // 3
        FS_QBS_DISPATCH(0x2);    // 2
        FS_QBS_DISPATCH(0x1);    // 1
#undef FS_QBS_DISPATCH
    }

    // Default: qbs not known at compile time
    for (size_t j0 = 0; j0 < ntotal2; j0 += 32) {
        const uint8_t* LUT = LUT0;
        int qi = qbs;
        int i0 = 0;
        while (qi) {
            int nq = qi & 15;
            qi >>= 4;
            res.set_block_origin(i0, j0);
#define FS_NQ_DISPATCH2(NQ)                                   \
    case NQ:                                                  \
        pq4_kernel_qbs_256<NQ>(nsq, codes, LUT, res, scaler); \
        break
            switch (nq) {
                FS_NQ_DISPATCH2(1);
                FS_NQ_DISPATCH2(2);
                FS_NQ_DISPATCH2(3);
                FS_NQ_DISPATCH2(4);
#undef FS_NQ_DISPATCH2
                default:
                    FAISS_THROW_FMT("accumulate nq=%d not instantiated", nq);
            }
            i0 += nq;
            LUT += nq * nsq * 16;
        }
        codes += block_stride;
    }
}

} // anonymous namespace

/***************************************************************
 * ScannerMixIn: wraps a concrete handler + calls accumulation
 * kernels. Lives behind the virtual FastScanCodeScanner interface
 * so callers don't need to know the handler type.
 ***************************************************************/

template <class Handler>
struct ScannerMixIn : FastScanCodeScanner {
    Handler handler_;

    template <typename... Args>
    explicit ScannerMixIn(Args&&... args)
            : handler_(std::forward<Args>(args)...) {}

    SIMDResultHandlerToFloat* handler() override {
        return &handler_;
    }

    void accumulate_loop(
            int nq,
            size_t nb,
            int bbs,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            int pq2x4_scale,
            size_t block_stride) override {
        if (pq2x4_scale) {
            NormTableScaler<> scaler(pq2x4_scale);
            fs_pq4_accumulate_loop_fixed_scaler(
                    nq,
                    nb,
                    bbs,
                    nsq,
                    codes,
                    LUT,
                    handler_,
                    scaler,
                    block_stride);
        } else {
            DummyScaler<> dummy;
            fs_pq4_accumulate_loop_fixed_scaler(
                    nq,
                    nb,
                    bbs,
                    nsq,
                    codes,
                    LUT,
                    handler_,
                    dummy,
                    block_stride);
        }
    }

    void accumulate_loop_qbs(
            int qbs,
            size_t nb,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            int pq2x4_scale,
            size_t block_stride) override {
        if (pq2x4_scale) {
            NormTableScaler<> scaler(pq2x4_scale);
            fs_pq4_accumulate_loop_qbs_fixed_scaler(
                    qbs, nb, nsq, codes, LUT, handler_, scaler, block_stride);
        } else {
            DummyScaler<> dummy;
            fs_pq4_accumulate_loop_qbs_fixed_scaler(
                    qbs, nb, nsq, codes, LUT, handler_, dummy, block_stride);
        }
    }
};

/***************************************************************
 * Factory specialization for this SIMD level.
 *
 * Combinatorial dispatch: is_max × with_id_map × handler type
 *   k == 1:  SingleResultHandler
 *   k <= 20: HeapHandler
 *   k > 20:  ReservoirHandler (capacity = 2*k)
 ***************************************************************/

template <>
std::unique_ptr<FastScanCodeScanner>
make_fast_scan_scanner_impl<THE_LEVEL_TO_DISPATCH>(
        bool is_max,
        size_t nq,
        size_t ntotal,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        bool with_id_map) {
    // Helper lambda: given comparator C and with_id_map W, select handler
    auto make = [&]<class C, bool W>() -> std::unique_ptr<FastScanCodeScanner> {
        if (k == 1) {
            using H = SingleResultHandler<C, W>;
            return std::make_unique<ScannerMixIn<H>>(
                    nq, ntotal, distances, ids, sel);
        } else if (k <= 20) {
            using H = HeapHandler<C, W>;
            return std::make_unique<ScannerMixIn<H>>(
                    nq, ntotal, k, distances, ids, sel);
        } else {
            using H = ReservoirHandler<C, W>;
            return std::make_unique<ScannerMixIn<H>>(
                    nq, ntotal, size_t(k), size_t(2 * k), distances, ids, sel);
        }
    };

    if (is_max) {
        if (with_id_map) {
            return make.template operator()<CMax<uint16_t, int64_t>, true>();
        } else {
            return make.template operator()<CMax<uint16_t, int>, false>();
        }
    } else {
        if (with_id_map) {
            return make.template operator()<CMin<uint16_t, int64_t>, true>();
        } else {
            return make.template operator()<CMin<uint16_t, int>, false>();
        }
    }
}

} // namespace faiss
