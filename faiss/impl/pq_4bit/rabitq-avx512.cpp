/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __x86_64__
#ifndef __AVX512F__
#error "this should be compiled with AVX512"
#endif
#endif

#include <faiss/impl/pq_4bit/kernels_simd512.h>
#include <faiss/impl/pq_4bit/rabitq_dispatching.h>
#include <faiss/impl/pq_4bit/rabitq_dispatching_impl.h>

namespace faiss {

// Explicit template specialization for AVX512 - flat RaBitQ
template <>
PQ4CodeScanner* make_rabitq_flat_knn_handler_impl<SIMDLevel::AVX512>(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* labels,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool multi_bit) {
    return make_rabitq_flat_knn_handler_impl_body<SIMDLevel::AVX512>(
            index, is_max, nq, k, distances, labels, sel, context, multi_bit);
}

// Explicit template specialization for AVX512 - IVF RaBitQ
template <>
PQ4CodeScanner* make_ivf_rabitq_handler_impl<SIMDLevel::AVX512>(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* labels,
        const FastScanDistancePostProcessing* context,
        bool multibit) {
    return make_ivf_rabitq_handler_impl_body<SIMDLevel::AVX512>(
            is_max, index, nq, k, distances, labels, context, multibit);
}

} // namespace faiss
