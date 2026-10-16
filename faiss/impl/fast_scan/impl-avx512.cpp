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
#include <faiss/impl/fast_scan/kernels_simd512.h>

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

template <>
void accumulate_hacc_impl<SIMDLevel::AVX512>(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        int32_t* accu) {
    const size_t block_stride = 32 * size_t(nsq) / 2;
    for (int q = 0; q < nq; q++) {
        const uint8_t* lut_lo = LUT + size_t(q) * 2 * 16 * nsq;
        const uint8_t* lut_hi = lut_lo + 16 * size_t(nsq);
        for (size_t b = 0; b < ntotal2 / 32; b++) {
            accumulate_hacc_block_avx512(
                    nsq,
                    codes + b * block_stride,
                    lut_lo,
                    lut_hi,
                    accu + q * ntotal2 + b * 32);
        }
    }
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX512
