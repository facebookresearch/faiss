/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512
#include <faiss/impl/fast_scan/dispatching.h>        // IWYU pragma: keep
#include <faiss/impl/fast_scan/rabitq_dispatching.h> // IWYU pragma: keep

#include <faiss/impl/fast_scan/decompose_qbs.h>
#include <faiss/impl/fast_scan/kernels_simd512.h>

#include <immintrin.h>
#include <algorithm>
#include <cstdint>

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
