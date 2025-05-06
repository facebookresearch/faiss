/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/pq_4bit/dispatching.h>
#include <faiss/impl/pq_4bit/kernels_simd256.h>
#include <faiss/impl/pq_4bit/pq4_fast_scan.h>
#include <faiss/impl/pq_4bit/simd_result_handlers.h>

#ifdef __x86_64__
#ifndef __AVX512F__
#error "this should be compiled with -mavx512f"
#endif
#endif

namespace faiss {

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::AVX512F, false>(KNN_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::AVX512F, false>(KNN_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::AVX512F, true>(KNN_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::AVX512F, true>(KNN_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::AVX512F, true>(RRES_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::AVX512F, true>(RRES_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::AVX512F, true>(PRES_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::AVX512F, true>(PRES_ARGS_LIST_2);
}

} // namespace faiss
