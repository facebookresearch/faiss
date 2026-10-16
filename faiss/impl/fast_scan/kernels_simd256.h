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

#ifdef __AVX2__
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#endif

namespace faiss {

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
            simd32uint8 mask(15);
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
            simd32uint8 mask(15);
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

#ifdef __AVX2__

/* accumulate_hacc over one block of 32 codes. The low and high byte tables
 * are looked up separately, and their sums are kept in 16-bit lanes over
 * segments of 256 sub-quantizers, short enough for 255 * 256 < 2^16. Each
 * segment is then widened to 32 bits and added as low + (high << 8). */
static inline void accumulate_hacc_block_avx2(
        int nsq,
        const uint8_t* codes,
        const uint8_t* lut_lo,
        const uint8_t* lut_hi,
        int32_t* accu) {
    constexpr int kSegmentSq = 256;
    const __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i res[4];
    for (auto& r : res) {
        r = _mm256_setzero_si256();
    }

    // add the lanes holding the same elements
    auto combine2x2 = [](__m256i a, __m256i b) {
        __m256i a1b0 = _mm256_permute2f128_si256(a, b, 0x21);
        __m256i a0b1 = _mm256_blend_epi32(a, b, 0xF0);
        return _mm256_add_epi16(a1b0, a0b1);
    };
    // widen the unsigned 16-bit sums, then add low + (high << 8)
    auto add_shiftl8 = [](__m256i low, __m256i high, __m256i& r0, __m256i& r1) {
        __m256i l0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(low));
        __m256i l1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(low, 1));
        __m256i h0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(high));
        __m256i h1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(high, 1));
        r0 = _mm256_add_epi32(
                r0, _mm256_add_epi32(l0, _mm256_slli_epi32(h0, 8)));
        r1 = _mm256_add_epi32(
                r1, _mm256_add_epi32(l1, _mm256_slli_epi32(h1, 8)));
    };

    for (int seg = 0; seg < nsq; seg += kSegmentSq) {
        const int seg_end = std::min(nsq, seg + kSegmentSq);
        // [low or high byte table][elements 0-15 even / odd bytes,
        // elements 16-31 even / odd bytes]
        __m256i acc[2][4];
        for (auto& a : acc) {
            for (auto& reg : a) {
                reg = _mm256_setzero_si256();
            }
        }
        for (int sq = seg; sq < seg_end; sq += 2) {
            const __m256i c =
                    _mm256_loadu_si256((const __m256i*)(codes + 16 * sq));
            const __m256i clo = _mm256_and_si256(c, low_mask);
            const __m256i chi =
                    _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
            const uint8_t* luts[2] = {lut_lo + 16 * sq, lut_hi + 16 * sq};
            for (int t = 0; t < 2; t++) {
                const __m256i lut = _mm256_loadu_si256((const __m256i*)luts[t]);
                const __m256i res_lo = _mm256_shuffle_epi8(lut, clo);
                const __m256i res_hi = _mm256_shuffle_epi8(lut, chi);
                acc[t][0] = _mm256_add_epi16(acc[t][0], res_lo);
                acc[t][1] = _mm256_add_epi16(
                        acc[t][1], _mm256_srli_epi16(res_lo, 8));
                acc[t][2] = _mm256_add_epi16(acc[t][2], res_hi);
                acc[t][3] = _mm256_add_epi16(
                        acc[t][3], _mm256_srli_epi16(res_hi, 8));
            }
        }
        __m256i dis0[2], dis1[2];
        for (int t = 0; t < 2; t++) {
            acc[t][0] = _mm256_sub_epi16(
                    acc[t][0], _mm256_slli_epi16(acc[t][1], 8));
            dis0[t] = combine2x2(acc[t][0], acc[t][1]);
            acc[t][2] = _mm256_sub_epi16(
                    acc[t][2], _mm256_slli_epi16(acc[t][3], 8));
            dis1[t] = combine2x2(acc[t][2], acc[t][3]);
        }
        add_shiftl8(dis0[0], dis0[1], res[0], res[1]); // elements 0 to 15
        add_shiftl8(dis1[0], dis1[1], res[2], res[3]); // elements 16 to 31
    }

    _mm256_storeu_si256((__m256i*)accu, res[0]);
    _mm256_storeu_si256((__m256i*)(accu + 8), res[1]);
    _mm256_storeu_si256((__m256i*)(accu + 16), res[2]);
    _mm256_storeu_si256((__m256i*)(accu + 24), res[3]);
}

#endif // __AVX2__

} // namespace faiss
