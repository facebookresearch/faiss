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

#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/simd_levels.h>

#if FAISS_SIMD_TU(RISCV_RVV)
#include <faiss/impl/fast_scan/kernels_rvv.h>
// Only used to gate the branches below; #undef'd at the end of this header so
// it does not leak into later includes in the same TU.
#define FAISS_RVV_NATIVE_KERNELS 1
#endif

#include <algorithm>
#include <cstdint>

namespace faiss {

/*
 * accumulate_hacc (see fast_scan.h) over one block of 32 codes. The low and
 * high byte tables are looked up separately, and their sums are kept in
 * 16-bit lanes over segments of 256 sub-quantizers, short enough for
 * 255 * 256 < 2^16. Each segment is then added to the 32-bit accumulators as
 * low + (high << 8).
 */
template <SIMDLevel KernelSL>
void kernel_accumulate_hacc_block(
        int nsq,
        const uint8_t* codes,
        const uint8_t* lut_lo,
        const uint8_t* lut_hi,
        int32_t* accu) {
    constexpr SIMDLevel SL256 = simd256_level_selector<KernelSL>::value;
    using simd16uint16 = simd16uint16_tpl<SL256>;
    using simd32uint8 = simd32uint8_tpl<SL256>;
    constexpr int kSegmentSq = 256;
    const simd32uint8 mask(15);
    std::fill(accu, accu + 32, 0);
    for (int seg = 0; seg < nsq; seg += kSegmentSq) {
        const int seg_end = std::min(nsq, seg + kSegmentSq);
        // [low or high byte table][as in kernel_accumulate_block]
        simd16uint16 acc[2][4];
        for (auto& a : acc) {
            for (auto& r : a) {
                r.clear();
            }
        }
        for (int sq = seg; sq < seg_end; sq += 2) {
            const simd32uint8 c(codes + 16 * sq);
            const simd32uint8 clo = c & mask;
            const simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
            const uint8_t* luts[2] = {lut_lo + 16 * sq, lut_hi + 16 * sq};
            for (int t = 0; t < 2; t++) {
                const simd32uint8 lut(luts[t]);
                const simd32uint8 res0 = lut.lookup_2_lanes(clo);
                const simd32uint8 res1 = lut.lookup_2_lanes(chi);
                acc[t][0] += simd16uint16(res0);
                acc[t][1] += simd16uint16(res0) >> 8;
                acc[t][2] += simd16uint16(res1);
                acc[t][3] += simd16uint16(res1) >> 8;
            }
        }
        uint16_t sums[2][32]; // [table][element]
        for (int t = 0; t < 2; t++) {
            acc[t][0] -= acc[t][1] << 8;
            combine2x2(acc[t][0], acc[t][1]).storeu(sums[t]);
            acc[t][2] -= acc[t][3] << 8;
            combine2x2(acc[t][2], acc[t][3]).storeu(sums[t] + 16);
        }
        for (int i = 0; i < 32; i++) {
            accu[i] += sums[0][i] + (int32_t(sums[1][i]) << 8);
        }
    }
}

/// accumulate_hacc with kernel_accumulate_hacc_block
template <SIMDLevel KernelSL>
void accumulate_hacc_simdlib(
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
            kernel_accumulate_hacc_block<KernelSL>(
                    nsq,
                    codes + b * block_stride,
                    lut_lo,
                    lut_hi,
                    accu + q * ntotal2 + b * 32);
        }
    }
}

/*
 * Multi-BB variant: accumulates NQ queries x BB*32 database elements.
 * Used by the search_1 path (bbs > 32).
 *
 * KernelSL selects the SIMD type width used for the inner accumulation
 * loop.  In DD mode the caller passes THE_LEVEL_TO_DISPATCH so the
 * kernel uses real AVX2/AVX512 types rather than the emulated-scalar
 * fallback that SINGLE_SIMD_LEVEL_256 would give.
 */
template <
        int NQ,
        int BB,
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void kernel_accumulate_block(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    constexpr SIMDLevel SL256 = simd256_level_selector<KernelSL>::value;
    using simd16uint16 = simd16uint16_tpl<SL256>;
    using simd32uint8 = simd32uint8_tpl<SL256>;

#if defined(FAISS_RVV_NATIVE_KERNELS)
    // Native RVV path (see kernels_rvv.h). KernelSL is checked, not
    // SL256: simd256_level_selector maps RISCV_RVV to NONE.
    if constexpr (KernelSL == SIMDLevel::RISCV_RVV) {
        rvv_fast_scan::kernel_accumulate_block<NQ, BB>(
                nsq, codes, LUT, res, scaler);
        return;
    }
#endif

    // Constant nibble mask, hoisted out of the sq/b loops.
    const simd32uint8 mask(0xf);

    // distance accumulators
    simd16uint16 accu[NQ][BB][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < BB; b++) {
            accu[q][b][0].clear();
            accu[q][b][1].clear();
            accu[q][b][2].clear();
            accu[q][b][3].clear();
        }
    }

    for (int sq = 0; sq < nsq - scaler.nscale; sq += 2) {
        simd32uint8 lut_cache[NQ];
        for (int q = 0; q < NQ; q++) {
            lut_cache[q] = simd32uint8(LUT);
            LUT += 32;
        }

        for (int b = 0; b < BB; b++) {
            simd32uint8 c = simd32uint8(codes);
            codes += 32;
            simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
            simd32uint8 clo = c & mask;

            for (int q = 0; q < NQ; q++) {
                simd32uint8 lut = lut_cache[q];
                simd32uint8 res0 = lut.lookup_2_lanes(clo);
                simd32uint8 res1 = lut.lookup_2_lanes(chi);

                accu[q][b][0] += simd16uint16(res0);
                accu[q][b][1] += simd16uint16(res0) >> 8;

                accu[q][b][2] += simd16uint16(res1);
                accu[q][b][3] += simd16uint16(res1) >> 8;
            }
        }
    }

    for (int sq = 0; sq < scaler.nscale; sq += 2) {
        simd32uint8 lut_cache[NQ];
        for (int q = 0; q < NQ; q++) {
            lut_cache[q] = simd32uint8(LUT);
            LUT += 32;
        }

        for (int b = 0; b < BB; b++) {
            simd32uint8 c = simd32uint8(codes);
            codes += 32;
            simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
            simd32uint8 clo = c & mask;

            for (int q = 0; q < NQ; q++) {
                simd32uint8 lut = lut_cache[q];

                simd32uint8 res0 = scaler.lookup(lut, clo);
                accu[q][b][0] += scaler.scale_lo(res0); // handle vectors 0..7
                accu[q][b][1] += scaler.scale_hi(res0); // handle vectors 8..15

                simd32uint8 res1 = scaler.lookup(lut, chi);
                accu[q][b][2] += scaler.scale_lo(res1); // handle vectors 16..23
                accu[q][b][3] +=
                        scaler.scale_hi(res1); //  handle vectors 24..31
            }
        }
    }

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < BB; b++) {
            accu[q][b][0] -= accu[q][b][1] << 8;
            simd16uint16 dis0 = combine2x2(accu[q][b][0], accu[q][b][1]);

            accu[q][b][2] -= accu[q][b][3] << 8;
            simd16uint16 dis1 = combine2x2(accu[q][b][2], accu[q][b][3]);

            res.handle(q, b, dis0, dis1);
        }
    }
}

/*
 * Single-BB QBS variant: accumulates NQ queries x 32 db elements (BB=1).
 * Used by the decompose_qbs layer for non-AVX512 paths.
 */
template <
        int NQ,
        SIMDLevel KernelSL = SINGLE_SIMD_LEVEL,
        class ResultHandler,
        class Scaler>
void pq4_kernel_qbs_256(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    constexpr SIMDLevel SL256 = simd256_level_selector<KernelSL>::value;
    using simd16uint16 = simd16uint16_tpl<SL256>;
    using simd32uint8 = simd32uint8_tpl<SL256>;

#if defined(FAISS_RVV_NATIVE_KERNELS)
    // Native RVV path: no 256-bit simdlib / GNU VLS types, fixed vl that
    // is valid for every VLEN >= 128 (see kernels_rvv.h).
    if constexpr (KernelSL == SIMDLevel::RISCV_RVV) {
        rvv_fast_scan::pq4_kernel_qbs<NQ>(nsq, codes, LUT, res, scaler);
        return;
    }
#endif

    // dummy alloc to keep the windows compiler happy
    constexpr int NQA = NQ > 0 ? NQ : 1;
    // distance accumulators
    // layout: accu[q][b]: distance accumulator for vectors 8*b..8*b+7
    simd16uint16 accu[NQA][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
        }
    }

    // _mm_prefetch(codes + 768, 0);
    for (int sq = 0; sq < nsq - scaler.nscale; sq += 2) {
        simd32uint8 c;
        c.loadu(codes);
        codes += 32;

        simd32uint8 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
        simd32uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8 lut(LUT);
            LUT += 32;

            simd32uint8 res0 = lut.lookup_2_lanes(clo);
            simd32uint8 res1 = lut.lookup_2_lanes(chi);

            accu[q][0] += simd16uint16(res0);
            accu[q][1] += simd16uint16(res0) >> 8;

            accu[q][2] += simd16uint16(res1);
            accu[q][3] += simd16uint16(res1) >> 8;
        }
    }

    for (int sq = 0; sq < scaler.nscale; sq += 2) {
        simd32uint8 c;
        c.loadu(codes);
        codes += 32;

        simd32uint8 mask(0xf);
        // shift op does not exist for int8...
        simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
        simd32uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            simd32uint8 lut(LUT);
            LUT += 32;

            simd32uint8 res0 = scaler.lookup(lut, clo);
            accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
            accu[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

            simd32uint8 res1 = scaler.lookup(lut, chi);
            accu[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
            accu[q][3] += scaler.scale_hi(res1); //  handle vectors 24..31
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16 dis0 = combine2x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16 dis1 = combine2x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

template <SIMDLevel SL>
struct PQ4QBSKernel {
    template <int NQ, class ResultHandler, class Scaler>
    FAISS_ALWAYS_INLINE static void run(
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            ResultHandler& res,
            const Scaler& scaler) {
        pq4_kernel_qbs_256<NQ, SL>(nsq, codes, LUT, res, scaler);
    }
};

} // namespace faiss

#undef FAISS_RVV_NATIVE_KERNELS
