/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file rabitq_dispatching.h
 * @brief Per-SIMD TU dispatch for RaBitQ flat scanner.
 *
 * Included after dispatching.h in each per-SIMD TU, so that
 * ScannerMixIn from dispatching.h is visible (same TU).
 *
 * Provides the rabitq_make_knn_scanner_impl<SL> specialization
 * that wraps RaBitQHeapHandler in ScannerMixIn.
 */

#ifndef THE_LEVEL_TO_DISPATCH
#error "Define THE_LEVEL_TO_DISPATCH before including this header"
#endif

#include <faiss/IndexIVFRaBitQFastScan.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/fast_scan/fast_scan.h>

// ScannerMixIn is visible from dispatching.h (same TU)

namespace faiss {

template <>
std::unique_ptr<FastScanCodeScanner> rabitq_make_knn_scanner_impl<
        THE_LEVEL_TO_DISPATCH>(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool is_multi_bit) {
    if (is_max) {
        using H = RaBitQHeapHandler<CMax<uint16_t, int>, false>;
        return std::make_unique<ScannerMixIn<H>>(
                index, nq, k, distances, ids, sel, &context, is_multi_bit);
    } else {
        using H = RaBitQHeapHandler<CMin<uint16_t, int>, false>;
        return std::make_unique<ScannerMixIn<H>>(
                index, nq, k, distances, ids, sel, &context, is_multi_bit);
    }
}

// IVF RaBitQ scanner factory
template <>
std::unique_ptr<FastScanCodeScanner> rabitq_ivf_make_knn_scanner_impl<
        THE_LEVEL_TO_DISPATCH>(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing* context,
        bool multi_bit) {
    if (is_max) {
        using C = CMax<uint16_t, int64_t>;
        using H = simd_result_handlers::IVFRaBitQHeapHandler<C>;
        return std::make_unique<ScannerMixIn<H>>(
                index, nq, k, distances, ids, sel, context, multi_bit);
    } else {
        using C = CMin<uint16_t, int64_t>;
        using H = simd_result_handlers::IVFRaBitQHeapHandler<C>;
        return std::make_unique<ScannerMixIn<H>>(
                index, nq, k, distances, ids, sel, context, multi_bit);
    }
}

} // namespace faiss
