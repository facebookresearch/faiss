/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_levels.h>

#ifdef COMPILE_SIMD_NEON

#ifndef __aarch64__
#error "this can only run on aarch64"
#endif

#include <faiss/impl/pq_4bit/dispatching.h>
#include <faiss/impl/pq_4bit/kernels_simd256.h>
#include <faiss/impl/pq_4bit/pq4_fast_scan.h>
#include <faiss/impl/pq_4bit/simd_result_handlers.h>

namespace faiss {

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::ARM_NEON, false>(KNN_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::ARM_NEON, false>(KNN_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::ARM_NEON, true>(KNN_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::ARM_NEON, true>(KNN_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::ARM_NEON, true>(RRES_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::ARM_NEON, true>(RRES_ARGS_LIST_2);
}

template <>
PQ4CodeScanner* make_pq4_scanner<SIMDLevel::ARM_NEON, true>(PRES_ARGS_LIST) {
    return make_pq4_scanner_1<SIMDLevel::ARM_NEON, true>(PRES_ARGS_LIST_2);
}

} // namespace faiss

#endif // COMPILE_SIMD_NEON
