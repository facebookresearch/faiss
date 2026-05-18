/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512
#include <faiss/impl/fast_scan/dispatching.h>        // IWYU pragma: keep
#include <faiss/impl/fast_scan/rabitq_dispatching.h> // IWYU pragma: keep

#include <faiss/impl/fast_scan/decompose_qbs.h>

namespace faiss {

using namespace simd_result_handlers;

template <>
void accumulate_to_mem_impl<SIMDLevel::AVX512>(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* accu) {
    // Use AVX2-level handler (256-bit StoreResultHandler) since the 512-bit
    // kernels reduce to AVX2-level simd16uint16 via FixedStorage512.
    StoreResultHandler<SIMDLevel::AVX2> handler(accu, ntotal2);
    DummyScaler<SIMDLevel::AVX512> scaler;
    // kernel_accumulate_block in decompose_qbs.h selects pq4_kernel_qbs_512
    // via #ifdef __AVX512F__ (which is set for this TU).
    accumulate<SIMDLevel::AVX512>(
            nq, ntotal2, nsq, codes, LUT, handler, scaler, 32 * nsq / 2);
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX512
