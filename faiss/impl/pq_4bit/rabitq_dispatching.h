/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * This header contains template definitions and declarations for RaBitQ
 * FastScan dispatch. The explicit template specializations for AVX2/AVX512
 * are in rabitq-avx2.cpp and rabitq-avx512.cpp respectively.
 */

#include <faiss/impl/pq_4bit/pq4_fast_scan.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

// Forward declarations
struct IndexRaBitQFastScan;
struct IndexIVFRaBitQFastScan;
struct IDSelector;
struct FastScanDistancePostProcessing;

// Factory function declarations for flat RaBitQ
// These are implemented in rabitq-avx2.cpp and rabitq-avx512.cpp
template <SIMDLevel SL>
PQ4CodeScanner* make_rabitq_flat_knn_handler_impl(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* labels,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool multi_bit);

// Factory function declarations for IVF RaBitQ
template <SIMDLevel SL>
PQ4CodeScanner* make_ivf_rabitq_handler_impl(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* labels,
        const FastScanDistancePostProcessing* context,
        bool multibit);

} // namespace faiss
