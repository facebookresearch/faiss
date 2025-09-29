/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __x86_64__
#ifndef __AVX2__
#error "this should be compiled with AVX2"
#endif
#endif

#include <faiss/impl/pq_4bit/dispatching.h>
#include <faiss/impl/pq_4bit/kernels_simd256.h>
#include <faiss/impl/pq_4bit/pq4_fast_scan.h>
#include <faiss/impl/pq_4bit/simd_result_handlers.h>

namespace faiss {

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::AVX2, false>(KNN_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::AVX2, false>(KNN_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::AVX2, true>(KNN_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::AVX2, true>(KNN_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::AVX2, true>(RRES_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::AVX2, true>(RRES_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::AVX2, true>(PRES_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::AVX2, true>(PRES_ARGS_LIST_2);
}

} // namespace faiss
