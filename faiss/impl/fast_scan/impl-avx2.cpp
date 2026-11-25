/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX2
#include <faiss/impl/fast_scan/dispatching.h>        // IWYU pragma: keep
#include <faiss/impl/fast_scan/rabitq_dispatching.h> // IWYU pragma: keep

#include <faiss/impl/fast_scan/decompose_qbs.h>
#include <faiss/impl/fast_scan/kernels_simd256.h>

#include <cstdint>

namespace faiss {

using namespace simd_result_handlers;

template <>
void accumulate_to_mem_impl<SIMDLevel::AVX2>(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* accu) {
    StoreResultHandler<SIMDLevel::AVX2> handler(accu, ntotal2);
    DummyScaler<SIMDLevel::AVX2> scaler;
    accumulate<SIMDLevel::AVX2>(
            nq, ntotal2, nsq, codes, LUT, handler, scaler, 32 * nsq / 2);
}

template <>
void accumulate_hacc_impl<SIMDLevel::AVX2>(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        int32_t* accu) {
    accumulate_hacc_simdlib<SIMDLevel::AVX2>(
            nq, ntotal2, nsq, codes, LUT, accu);
}

} // namespace faiss
