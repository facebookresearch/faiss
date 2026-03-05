/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef THE_LEVEL_TO_DISPATCH
#error "THE_LEVEL_TO_DISPATCH must be defined before including this header"
#endif

#include <memory>

#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/LookupTableScaler.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/pq_4bit/NormTableScalerSL.h>
#include <faiss/impl/pq_4bit/decompose_qbs.h>
#include <faiss/impl/pq_4bit/kernels_simd256.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/Heap.h>

namespace faiss {

namespace {

constexpr SIMDLevel RABITQ_SL = THE_LEVEL_TO_DISPATCH;

using namespace simd_result_handlers;

template <class Handler>
struct RaBitQScannerMixIn : PQ4CodeScanner {
    Handler handler_;

    template <typename... Args>
    explicit RaBitQScannerMixIn(Args&&... args)
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
            const NormTableScaler* scaler,
            size_t block_stride) override {
        if (scaler) {
            NormTableScalerSL<RABITQ_SL> typed(scaler->scale_int);
            pq4_accumulate_loop_fixed_scaler(
                    nq,
                    nb,
                    bbs,
                    nsq,
                    codes,
                    LUT,
                    handler_,
                    typed,
                    block_stride);
        } else {
            DummyScaler<RABITQ_SL> dummy;
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
            const NormTableScaler* scaler,
            size_t block_stride) override {
        if (scaler) {
            NormTableScalerSL<RABITQ_SL> typed(scaler->scale_int);
            pq4_accumulate_loop_qbs_fixed_scaler<RABITQ_SL>(
                    qbs, nb, nsq, codes, LUT, handler_, typed, block_stride);
        } else {
            DummyScaler<RABITQ_SL> dummy;
            pq4_accumulate_loop_qbs_fixed_scaler<RABITQ_SL>(
                    qbs, nb, nsq, codes, LUT, handler_, dummy, block_stride);
        }
    }

   private:
    template <int NQ, int BB, class Scaler>
    static void accumulate_fixed_blocks_sl(
            size_t nb,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            Handler& res,
            const Scaler& scaler,
            size_t block_stride) {
        constexpr int bbs = 32 * BB;
        for (size_t j0 = 0; j0 < nb; j0 += bbs) {
            FixedStorageHandler<NQ, 2 * BB, RABITQ_SL> res2;
            kernel_accumulate_block<NQ, BB, RABITQ_SL>(
                    nsq, codes, LUT, res2, scaler);
            res.set_block_origin(0, j0);
            res2.to_other_handler(res);
            codes += block_stride;
        }
    }

    template <class Scaler>
    static void pq4_accumulate_loop_fixed_scaler(
            int nq,
            size_t nb,
            int bbs,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            Handler& res,
            const Scaler& scaler,
            size_t block_stride) {
        FAISS_THROW_IF_NOT(is_aligned_pointer(codes));
        FAISS_THROW_IF_NOT(is_aligned_pointer(LUT));
        FAISS_THROW_IF_NOT(bbs % 32 == 0);
        FAISS_THROW_IF_NOT(nb % bbs == 0);

#define DISPATCH(NQ, BB)                                         \
    case NQ * 1000 + BB:                                         \
        accumulate_fixed_blocks_sl<NQ, BB>(                      \
                nb, nsq, codes, LUT, res, scaler, block_stride); \
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
};

} // anonymous namespace

// Flat RaBitQ scanner factory
template <>
std::unique_ptr<PQ4CodeScanner> rabitq_make_knn_scanner_impl<
        THE_LEVEL_TO_DISPATCH>(
        bool is_max,
        const IndexRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool multi_bit) {
    if (is_max) {
        using C = CMax<uint16_t, int>;
        return std::make_unique<
                RaBitQScannerMixIn<RaBitQHeapHandler<C, false, RABITQ_SL>>>(
                index, nq, k, distances, ids, sel, &context, multi_bit);
    } else {
        using C = CMin<uint16_t, int>;
        return std::make_unique<
                RaBitQScannerMixIn<RaBitQHeapHandler<C, false, RABITQ_SL>>>(
                index, nq, k, distances, ids, sel, &context, multi_bit);
    }
}

} // namespace faiss
