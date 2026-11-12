/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define THE_LEVEL_TO_DISPATCH SIMDLevel::ARM_NEON
#include <faiss/impl/fast_scan/dispatching.h>        // IWYU pragma: keep
#include <faiss/impl/fast_scan/rabitq_dispatching.h> // IWYU pragma: keep

#include <faiss/impl/fast_scan/decompose_qbs.h>

namespace faiss {

using namespace simd_result_handlers;

template <>
void accumulate_to_mem_impl<SIMDLevel::ARM_NEON>(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* accu) {
    StoreResultHandler<SIMDLevel::ARM_NEON> handler(accu, ntotal2);
    DummyScaler<SIMDLevel::ARM_NEON> scaler;
    accumulate<SIMDLevel::ARM_NEON>(
            nq, ntotal2, nsq, codes, LUT, handler, scaler, 32 * nsq / 2);
}

} // namespace faiss
