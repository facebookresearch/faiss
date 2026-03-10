/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_NEON

#define THE_LEVEL_TO_DISPATCH SIMDLevel::ARM_NEON
#include <faiss/impl/fast_scan/dispatching.h>        // IWYU pragma: keep
#include <faiss/impl/fast_scan/rabitq_dispatching.h> // IWYU pragma: keep

// ARM_SVE: forward to ARM_NEON implementation until a dedicated SVE
// specialization is written (same pattern as scalar_quantizer/sq-neon.cpp).
#ifdef COMPILE_SIMD_ARM_SVE

namespace faiss {

template <>
std::unique_ptr<FastScanCodeScanner> make_fast_scan_scanner_impl<
        SIMDLevel::ARM_SVE>(
        bool is_max,
        int impl,
        size_t nq,
        size_t ntotal,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        bool with_id_map) {
    return make_fast_scan_scanner_impl<SIMDLevel::ARM_NEON>(
            is_max, impl, nq, ntotal, k, distances, ids, sel, with_id_map);
}

template <>
std::unique_ptr<FastScanCodeScanner> make_range_scanner_impl<
        SIMDLevel::ARM_SVE>(
        bool is_max,
        RangeSearchResult& rres,
        float radius,
        size_t ntotal,
        const IDSelector* sel) {
    return make_range_scanner_impl<SIMDLevel::ARM_NEON>(
            is_max, rres, radius, ntotal, sel);
}

template <>
std::unique_ptr<FastScanCodeScanner> make_partial_range_scanner_impl<
        SIMDLevel::ARM_SVE>(
        bool is_max,
        RangeSearchPartialResult& pres,
        float radius,
        size_t ntotal,
        size_t q0,
        size_t q1,
        const IDSelector* sel) {
    return make_partial_range_scanner_impl<SIMDLevel::ARM_NEON>(
            is_max, pres, radius, ntotal, q0, q1, sel);
}

template <>
std::unique_ptr<FastScanCodeScanner> rabitq_make_knn_scanner_impl<
        SIMDLevel::ARM_SVE>(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool is_multi_bit) {
    return rabitq_make_knn_scanner_impl<SIMDLevel::ARM_NEON>(
            index, is_max, nq, k, distances, ids, sel, context, is_multi_bit);
}

template <>
std::unique_ptr<FastScanCodeScanner> rabitq_ivf_make_knn_scanner_impl<
        SIMDLevel::ARM_SVE>(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* ids,
        const FastScanDistancePostProcessing* context,
        bool multi_bit) {
    return rabitq_ivf_make_knn_scanner_impl<SIMDLevel::ARM_NEON>(
            is_max, index, nq, k, distances, ids, context, multi_bit);
}

} // namespace faiss

#endif // COMPILE_SIMD_ARM_SVE

#endif // COMPILE_SIMD_ARM_NEON
