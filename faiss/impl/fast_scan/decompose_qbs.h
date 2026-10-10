/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/fast_scan/kernels_simd256.h>
#include <faiss/impl/fast_scan/kernels_simd512.h>
#ifdef COMPILE_SIMD_ARM_NEON
#include <faiss/impl/fast_scan/kernels_neon.h>
#endif
#include <faiss/impl/fast_scan/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

/*
 * Unified kernel: selects 256-bit vs 512-bit path based on
 * compile-time __AVX512F__ guard.
 *
 * KernelSL selects the SIMD type width used for the inner accumulation
 * loop.  In DD mode the caller passes the dispatch level so the kernel
 * uses real AVX2 types rather than the emulated-scalar fallback.
 */
template <
        int NQ,
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void kernel_accumulate_block(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
#ifdef __AVX512F__
    if constexpr (
            KernelSL == SIMDLevel::AVX512 ||
            KernelSL == SIMDLevel::AVX512_VPOPCNT ||
            KernelSL == SIMDLevel::AVX512_SPR) {
        pq4_kernel_qbs_512<NQ>(nsq, codes, LUT, res, scaler);
    } else {
        PQ4QBSKernel<KernelSL>::template run<NQ>(nsq, codes, LUT, res, scaler);
    }
#else
    PQ4QBSKernel<KernelSL>::template run<NQ>(nsq, codes, LUT, res, scaler);
#endif
}

template <
        int NQ,
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void kernel_accumulate_block_loop(
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    for_each_block<32>(ntotal2, codes, block_stride, res, [&](size_t) {
        kernel_accumulate_block<NQ, KernelSL>(nsq, codes, LUT, res, scaler);
    });
}

// non-template version of accumulate kernel -- dispatches dynamically
template <
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void accumulate(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    assert(nsq % 2 == 0);
    assert(is_aligned_pointer(LUT));

#define DISPATCH(NQ)                                                  \
    case NQ:                                                          \
        kernel_accumulate_block_loop<NQ, KernelSL>(                   \
                ntotal2, nsq, codes, LUT, res, scaler, block_stride); \
        return

    switch (nq) {
        DISPATCH(1);
        DISPATCH(2);
        DISPATCH(3);
        DISPATCH(4);
    }
    FAISS_THROW_FMT("accumulate nq=%d not instantiated", nq);

#undef DISPATCH
}

} // namespace faiss
