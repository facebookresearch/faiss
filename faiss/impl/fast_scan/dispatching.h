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
 * Kernel helpers come from accumulate_loops.h (search_1 multi-BB path
 * and QBS 256-bit path) and accumulate_loops_512.h (QBS 512-bit path,
 * AVX512 TU only).
 */

#ifndef THE_LEVEL_TO_DISPATCH
#error "Define THE_LEVEL_TO_DISPATCH before including this header"
#endif

#include <memory>

#include <faiss/impl/fast_scan/accumulate_loops.h>
#include <faiss/impl/fast_scan/fast_scan.h>

#if defined(COMPILE_SIMD_AVX512) && defined(__AVX512F__)
#include <faiss/impl/fast_scan/accumulate_loops_512.h>
#endif

namespace faiss {

using namespace simd_result_handlers;

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
            pq4_accumulate_loop_fixed_scaler(
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
            pq4_accumulate_loop_fixed_scaler(
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
#if defined(COMPILE_SIMD_AVX512) && defined(__AVX512F__)
        constexpr bool use_avx512_qbs =
                (THE_LEVEL_TO_DISPATCH == SIMDLevel::AVX512 ||
                 THE_LEVEL_TO_DISPATCH == SIMDLevel::AVX512_SPR);
#else
        constexpr bool use_avx512_qbs = false;
#endif
        if constexpr (use_avx512_qbs) {
            // Use 512-bit QBS kernels with properly-leveled scalers.
            if (pq2x4_scale) {
                NormTableScaler<THE_LEVEL_TO_DISPATCH> scaler(pq2x4_scale);
                pq4_accumulate_loop_qbs_fixed_scaler_512(
                        qbs,
                        nb,
                        nsq,
                        codes,
                        LUT,
                        handler_,
                        scaler,
                        block_stride);
            } else {
                DummyScaler<THE_LEVEL_TO_DISPATCH> dummy;
                pq4_accumulate_loop_qbs_fixed_scaler_512(
                        qbs,
                        nb,
                        nsq,
                        codes,
                        LUT,
                        handler_,
                        dummy,
                        block_stride);
            }
        } else {
            if (pq2x4_scale) {
                NormTableScaler<> scaler(pq2x4_scale);
                pq4_accumulate_loop_qbs_fixed_scaler_256(
                        qbs,
                        nb,
                        nsq,
                        codes,
                        LUT,
                        handler_,
                        scaler,
                        block_stride);
            } else {
                DummyScaler<> dummy;
                pq4_accumulate_loop_qbs_fixed_scaler_256(
                        qbs,
                        nb,
                        nsq,
                        codes,
                        LUT,
                        handler_,
                        dummy,
                        block_stride);
            }
        }
    }
};

/***************************************************************
 * Factory specialization for this SIMD level.
 *
 * Combinatorial dispatch: is_max × with_id_map × handler type
 *   k == 1:  SingleResultHandler
 *   impl even: HeapHandler
 *   impl odd:  ReservoirHandler (capacity = 2*k)
 ***************************************************************/

template <>
std::unique_ptr<FastScanCodeScanner> make_fast_scan_scanner_impl<
        THE_LEVEL_TO_DISPATCH>(
        bool is_max,
        int impl,
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
        } else if (impl % 2 == 0) {
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

/***************************************************************
 * Range search scanner factories.
 ***************************************************************/

template <>
std::unique_ptr<FastScanCodeScanner> make_range_scanner_impl<
        THE_LEVEL_TO_DISPATCH>(
        bool is_max,
        RangeSearchResult& rres,
        float radius,
        size_t ntotal,
        const IDSelector* sel) {
    if (is_max) {
        using C = CMax<uint16_t, int64_t>;
        return std::make_unique<ScannerMixIn<RangeHandler<C, true>>>(
                rres, radius, ntotal, sel);
    } else {
        using C = CMin<uint16_t, int64_t>;
        return std::make_unique<ScannerMixIn<RangeHandler<C, true>>>(
                rres, radius, ntotal, sel);
    }
}

template <>
std::unique_ptr<FastScanCodeScanner> make_partial_range_scanner_impl<
        THE_LEVEL_TO_DISPATCH>(
        bool is_max,
        RangeSearchPartialResult& pres,
        float radius,
        size_t ntotal,
        size_t q0,
        size_t q1,
        const IDSelector* sel) {
    if (is_max) {
        using C = CMax<uint16_t, int64_t>;
        return std::make_unique<ScannerMixIn<PartialRangeHandler<C, true>>>(
                pres, radius, ntotal, q0, q1, sel);
    } else {
        using C = CMin<uint16_t, int64_t>;
        return std::make_unique<ScannerMixIn<PartialRangeHandler<C, true>>>(
                pres, radius, ntotal, q0, q1, sel);
    }
}

} // namespace faiss
