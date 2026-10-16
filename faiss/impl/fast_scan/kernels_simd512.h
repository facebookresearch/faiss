/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>

#ifdef __AVX512F__

#include <immintrin.h>
#include <algorithm>
#include <cstdint>

namespace faiss {

// Explicit SIMD-level aliases for this file (no global bare aliases).
using simd32uint16 = simd32uint16_tpl<SINGLE_SIMD_LEVEL_512>;
using simd64uint8 = simd64uint8_tpl<SINGLE_SIMD_LEVEL_512>;

// NQ=1 specialization: processes 512-bit chunks aggressively.
// a special version for NQ=1.
// Despite the function being large in the text form, it compiles to a very
//    compact assembler code.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <class ResultHandler, class Scaler>
void kernel_accumulate_block_avx512_nq1(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    // Explicit SIMD levels for DD mode where bare aliases resolve to NONE
    // (512-bit NONE types don't exist — empty primary templates).
    using simd32uint16 = simd32uint16_tpl<SIMDLevel::AVX512>;
    using simd64uint8 = simd64uint8_tpl<SIMDLevel::AVX512>;
    using simd16uint16 = simd16uint16_tpl<SIMDLevel::AVX2>;
    using simd32uint8 = simd32uint8_tpl<SIMDLevel::AVX2>;

    // NQ is kept in order to match the similarity to baseline function
    constexpr int NQ = 1;
    // distance accumulators. We can accept more for NQ=1
    // layout: accu[q][b]: distance accumulator for vectors 32*b..32*b+15
    simd32uint16 accu[NQ][4];
    // layout: accu[q][b]: distance accumulator for vectors 32*b+16..32*b+31
    simd32uint16 accu1[NQ][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
            accu1[q][b].clear();
        }
    }

    // process "nsq - scaler.nscale" part
    const int nsq_minus_nscale = nsq - scaler.nscale;
    const int nsq_minus_nscale_8 = (nsq_minus_nscale / 8) * 8;
    const int nsq_minus_nscale_4 = (nsq_minus_nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nsq_minus_nscale_8; sq += 8) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 c1(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        simd64uint8 c1hi = simd64uint8(simd32uint16(c1) >> 4) & mask;
        simd64uint8 c1lo = c1 & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            {
                simd64uint8 res0 = lut.lookup_4_lanes(clo);
                simd64uint8 res1 = lut.lookup_4_lanes(chi);

                accu[q][0] += simd32uint16(res0);
                accu[q][1] += simd32uint16(res0) >> 8;

                accu[q][2] += simd32uint16(res1);
                accu[q][3] += simd32uint16(res1) >> 8;
            }
        }

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            {
                simd64uint8 res0 = lut.lookup_4_lanes(c1lo);
                simd64uint8 res1 = lut.lookup_4_lanes(c1hi);

                accu1[q][0] += simd32uint16(res0);
                accu1[q][1] += simd32uint16(res0) >> 8;

                accu1[q][2] += simd32uint16(res1);
                accu1[q][3] += simd32uint16(res1) >> 8;
            }
        }
    }

    // process leftovers: a single chunk of size 4
    if (nsq_minus_nscale_8 != nsq_minus_nscale_4) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            simd64uint8 res0 = lut.lookup_4_lanes(clo);
            simd64uint8 res1 = lut.lookup_4_lanes(chi);

            accu[q][0] += simd32uint16(res0);
            accu[q][1] += simd32uint16(res0) >> 8;

            accu[q][2] += simd32uint16(res1);
            accu[q][3] += simd32uint16(res1) >> 8;
        }
    }

    // process leftovers: a single chunk of size 2
    if (nsq_minus_nscale_4 != nsq_minus_nscale) {
        // prefetch
        simd32uint8 c(codes);
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

            accu[q][0] += simd32uint16(simd16uint16(res0));
            accu[q][1] += simd32uint16(simd16uint16(res0) >> 8);

            accu[q][2] += simd32uint16(simd16uint16(res1));
            accu[q][3] += simd32uint16(simd16uint16(res1) >> 8);
        }
    }

    // process "sq" part
    const int nscale = scaler.nscale;
    const int nscale_8 = (nscale / 8) * 8;
    const int nscale_4 = (nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nscale_8; sq += 8) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 c1(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        simd64uint8 c1hi = simd64uint8(simd32uint16(c1) >> 4) & mask;
        simd64uint8 c1lo = c1 & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            {
                simd64uint8 res0 = scaler.lookup(lut, clo);
                accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..15
                accu[q][1] += scaler.scale_hi(res0); // handle vectors 16..31

                simd64uint8 res1 = scaler.lookup(lut, chi);
                accu[q][2] += scaler.scale_lo(res1); // handle vectors 32..47
                accu[q][3] += scaler.scale_hi(res1); //  handle vectors 48..63
            }
        }

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            {
                simd64uint8 res0 = scaler.lookup(lut, c1lo);
                accu1[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
                accu1[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

                simd64uint8 res1 = scaler.lookup(lut, c1hi);
                accu1[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
                accu1[q][3] += scaler.scale_hi(res1); //  handle vectors 24..31
            }
        }
    }

    // process leftovers: a single chunk of size 4
    if (nscale_8 != nscale_4) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd64uint8 lut(LUT);
            LUT += 64;

            simd64uint8 res0 = scaler.lookup(lut, clo);
            accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..15
            accu[q][1] += scaler.scale_hi(res0); // handle vectors 16..31

            simd64uint8 res1 = scaler.lookup(lut, chi);
            accu[q][2] += scaler.scale_lo(res1); // handle vectors 32..47
            accu[q][3] += scaler.scale_hi(res1); //  handle vectors 48..63
        }
    }

    // process leftovers: a single chunk of size 2
    if (nscale_4 != nscale) {
        // prefetch
        simd32uint8 c(codes);
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
            accu[q][0] +=
                    simd32uint16(scaler.scale_lo(res0)); // handle vectors 0..7
            accu[q][1] +=
                    simd32uint16(scaler.scale_hi(res0)); // handle vectors 8..15

            simd32uint8 res1 = scaler.lookup(lut, chi);
            accu[q][2] += simd32uint16(
                    scaler.scale_lo(res1)); // handle vectors 16..23
            accu[q][3] += simd32uint16(
                    scaler.scale_hi(res1)); //  handle vectors 24..31
        }
    }

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b] += accu1[q][b];
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16 dis0 = combine4x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16 dis1 = combine4x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

// general-purpose case
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <int NQ, class ResultHandler, class Scaler>
void kernel_accumulate_block_avx512_nqx(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    // Explicit SIMD levels for DD mode (see nq1 variant for explanation).
    using simd32uint16 = simd32uint16_tpl<SIMDLevel::AVX512>;
    using simd64uint8 = simd64uint8_tpl<SIMDLevel::AVX512>;
    using simd16uint16 = simd16uint16_tpl<SIMDLevel::AVX2>;
    using simd32uint8 = simd32uint8_tpl<SIMDLevel::AVX2>;

    // dummy alloc to keep the windows compiler happy
    constexpr int NQA = NQ > 0 ? NQ : 1;
    // distance accumulators
    // layout: accu[q][b]: distance accumulator for vectors 8*b..8*b+7
    simd32uint16 accu[NQA][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b].clear();
        }
    }

    // process "nsq - scaler.nscale" part
    const int nsq_minus_nscale = nsq - scaler.nscale;
    const int nsq_minus_nscale_4 = (nsq_minus_nscale / 4) * 4;

    // process in chunks of 8
    for (int sq = 0; sq < nsq_minus_nscale_4; sq += 4) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd32uint8 lut_a(LUT);
            simd32uint8 lut_b(LUT + NQ * 32);

            simd64uint8 lut(lut_a, lut_b);
            LUT += 32;

            {
                simd64uint8 res0 = lut.lookup_4_lanes(clo);
                simd64uint8 res1 = lut.lookup_4_lanes(chi);

                accu[q][0] += simd32uint16(res0);
                accu[q][1] += simd32uint16(res0) >> 8;

                accu[q][2] += simd32uint16(res1);
                accu[q][3] += simd32uint16(res1) >> 8;
            }
        }

        LUT += NQ * 32;
    }

    // process leftovers: a single chunk of size 2
    if (nsq_minus_nscale_4 != nsq_minus_nscale) {
        // prefetch
        simd32uint8 c(codes);
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

            accu[q][0] += simd32uint16(simd16uint16(res0));
            accu[q][1] += simd32uint16(simd16uint16(res0) >> 8);

            accu[q][2] += simd32uint16(simd16uint16(res1));
            accu[q][3] += simd32uint16(simd16uint16(res1) >> 8);
        }
    }

    // process "sq" part
    const int nscale = scaler.nscale;
    const int nscale_4 = (nscale / 4) * 4;

    // process in chunks of 4
    for (int sq = 0; sq < nscale_4; sq += 4) {
        // prefetch
        simd64uint8 c(codes);
        codes += 64;

        simd64uint8 mask(0xf);
        // shift op does not exist for int8...
        simd64uint8 chi = simd64uint8(simd32uint16(c) >> 4) & mask;
        simd64uint8 clo = c & mask;

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 4 quantizers
            simd32uint8 lut_a(LUT);
            simd32uint8 lut_b(LUT + NQ * 32);

            simd64uint8 lut(lut_a, lut_b);
            LUT += 32;

            {
                simd64uint8 res0 = scaler.lookup(lut, clo);
                accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
                accu[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

                simd64uint8 res1 = scaler.lookup(lut, chi);
                accu[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
                accu[q][3] += scaler.scale_hi(res1); //  handle vectors 24..31
            }
        }

        LUT += NQ * 32;
    }

    // process leftovers: a single chunk of size 2
    if (nscale_4 != nscale) {
        // prefetch
        simd32uint8 c(codes);
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
            accu[q][0] +=
                    simd32uint16(scaler.scale_lo(res0)); // handle vectors 0..7
            accu[q][1] +=
                    simd32uint16(scaler.scale_hi(res0)); // handle vectors 8..15

            simd32uint8 res1 = scaler.lookup(lut, chi);
            accu[q][2] += simd32uint16(
                    scaler.scale_lo(res1)); // handle vectors 16..23
            accu[q][3] += simd32uint16(
                    scaler.scale_hi(res1)); //  handle vectors 24..31
        }
    }

    for (int q = 0; q < NQ; q++) {
        accu[q][0] -= accu[q][1] << 8;
        simd16uint16 dis0 = combine4x2(accu[q][0], accu[q][1]);
        accu[q][2] -= accu[q][3] << 8;
        simd16uint16 dis1 = combine4x2(accu[q][2], accu[q][3]);
        res.handle(q, 0, dis0, dis1);
    }
}

// Dispatcher: selects NQ=1 vs general case.
template <int NQ, class ResultHandler, class Scaler>
void pq4_kernel_qbs_512(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    if constexpr (NQ == 1) {
        kernel_accumulate_block_avx512_nq1<ResultHandler, Scaler>(
                nsq, codes, LUT, res, scaler);
    } else {
        kernel_accumulate_block_avx512_nqx<NQ, ResultHandler, Scaler>(
                nsq, codes, LUT, res, scaler);
    }
}

/* accumulate_hacc over one block of 32 codes. One 512-bit register holds 4
 * sub-quantizers, one per 128-bit lane. The sums of two lanes are added in
 * 16 bits before widening, so segments of 512 sub-quantizers keep them below
 * 255 * 512 / 2 < 2^16. */
static inline void accumulate_hacc_block_avx512(
        int nsq,
        const uint8_t* codes,
        const uint8_t* lut_lo,
        const uint8_t* lut_hi,
        int32_t* accu) {
    constexpr int kSegmentSq = 512;
    const __m512i low_mask = _mm512_set1_epi8(0xf);
    __m512i res[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};

    for (int seg = 0; seg < nsq; seg += kSegmentSq) {
        const int seg_end = std::min(nsq, seg + kSegmentSq);
        // 4 sub-quantizers from sq on; when only 2 are left, the upper
        // lanes are zero: zero codes look up zero table entries
        auto load4 = [seg_end](const uint8_t* p, int sq) {
            if (sq + 4 <= seg_end) {
                return _mm512_loadu_si512(p);
            }
            return _mm512_inserti64x4(
                    _mm512_setzero_si512(),
                    _mm256_loadu_si256((const __m256i*)p),
                    0);
        };
        // [low or high byte table][elements 0-15 even / odd bytes,
        // elements 16-31 even / odd bytes]
        __m512i acc[2][4];
        for (auto& a : acc) {
            for (auto& reg : a) {
                reg = _mm512_setzero_si512();
            }
        }
        for (int sq = seg; sq < seg_end; sq += 4) {
            const __m512i c = load4(codes + 16 * sq, sq);
            const __m512i clo = _mm512_and_si512(c, low_mask);
            const __m512i chi =
                    _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);
            const uint8_t* luts[2] = {lut_lo + 16 * sq, lut_hi + 16 * sq};
            for (int t = 0; t < 2; t++) {
                const __m512i lut = load4(luts[t], sq);
                const __m512i res_lo = _mm512_shuffle_epi8(lut, clo);
                const __m512i res_hi = _mm512_shuffle_epi8(lut, chi);
                acc[t][0] = _mm512_add_epi16(acc[t][0], res_lo);
                acc[t][1] = _mm512_add_epi16(
                        acc[t][1], _mm512_srli_epi16(res_lo, 8));
                acc[t][2] = _mm512_add_epi16(acc[t][2], res_hi);
                acc[t][3] = _mm512_add_epi16(
                        acc[t][3], _mm512_srli_epi16(res_hi, 8));
            }
        }

        // add the two 256-bit halves, then the lanes holding the same
        // elements, widening to 32 bits
        auto combine = [](__m512i even, __m512i odd) {
            __m256i e = _mm256_add_epi16(
                    _mm512_castsi512_si256(even),
                    _mm512_extracti64x4_epi64(even, 1));
            __m256i o = _mm256_add_epi16(
                    _mm512_castsi512_si256(odd),
                    _mm512_extracti64x4_epi64(odd, 1));
            e = _mm256_sub_epi16(e, _mm256_slli_epi16(o, 8));
            return _mm512_add_epi32(
                    _mm512_cvtepu16_epi32(
                            _mm256_permute2f128_si256(e, o, 0x21)),
                    _mm512_cvtepu16_epi32(_mm256_blend_epi32(e, o, 0xF0)));
        };
        __m512i dis0[2], dis1[2];
        for (int t = 0; t < 2; t++) {
            dis0[t] = combine(acc[t][0], acc[t][1]);
            dis1[t] = combine(acc[t][2], acc[t][3]);
        }
        // low + (high << 8)
        res[0] = _mm512_add_epi32(
                res[0],
                _mm512_add_epi32(dis0[0], _mm512_slli_epi32(dis0[1], 8)));
        res[1] = _mm512_add_epi32(
                res[1],
                _mm512_add_epi32(dis1[0], _mm512_slli_epi32(dis1[1], 8)));
    }

    _mm512_storeu_si512(accu, res[0]);      // elements 0 to 15
    _mm512_storeu_si512(accu + 16, res[1]); // elements 16 to 31
}

} // namespace faiss

#endif // __AVX512F__
