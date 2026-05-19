/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// RISC-V RVV: forward all fast_scan specializations to NONE until
// dedicated RVV implementations are written.

#ifdef COMPILE_SIMD_RISCV_RVV

#include <faiss/impl/fast_scan/fast_scan.h>

namespace faiss {

template <>
void accumulate_to_mem_impl<SIMDLevel::RISCV_RVV>(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* accu) {
    accumulate_to_mem_impl<SIMDLevel::NONE>(nq, ntotal2, nsq, codes, LUT, accu);
}

template <>
std::unique_ptr<FastScanCodeScanner> make_fast_scan_scanner_impl<
        SIMDLevel::RISCV_RVV>(
        bool is_max,
        int impl,
        size_t nq,
        size_t ntotal,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        bool with_id_map) {
    return make_fast_scan_scanner_impl<SIMDLevel::NONE>(
            is_max, impl, nq, ntotal, k, distances, ids, sel, with_id_map);
}

template <>
std::unique_ptr<FastScanCodeScanner> make_range_scanner_impl<
        SIMDLevel::RISCV_RVV>(
        bool is_max,
        RangeSearchResult& rres,
        float radius,
        size_t ntotal,
        const IDSelector* sel) {
    return make_range_scanner_impl<SIMDLevel::NONE>(
            is_max, rres, radius, ntotal, sel);
}

template <>
std::unique_ptr<FastScanCodeScanner> make_partial_range_scanner_impl<
        SIMDLevel::RISCV_RVV>(
        bool is_max,
        RangeSearchPartialResult& pres,
        float radius,
        size_t ntotal,
        size_t q0,
        size_t q1,
        const IDSelector* sel) {
    return make_partial_range_scanner_impl<SIMDLevel::NONE>(
            is_max, pres, radius, ntotal, q0, q1, sel);
}

template <>
std::unique_ptr<FastScanCodeScanner> rabitq_make_knn_scanner_impl<
        SIMDLevel::RISCV_RVV>(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool is_multi_bit) {
    return rabitq_make_knn_scanner_impl<SIMDLevel::NONE>(
            index, is_max, nq, k, distances, ids, sel, context, is_multi_bit);
}

template <>
std::unique_ptr<FastScanCodeScanner> rabitq_ivf_make_knn_scanner_impl<
        SIMDLevel::RISCV_RVV>(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing* context,
        bool multi_bit) {
    return rabitq_ivf_make_knn_scanner_impl<SIMDLevel::NONE>(
            is_max, index, nq, k, distances, ids, sel, context, multi_bit);
}

} // namespace faiss

#endif // COMPILE_SIMD_RISCV_RVV
