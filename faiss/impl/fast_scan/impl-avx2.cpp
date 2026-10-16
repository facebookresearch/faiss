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

#include <immintrin.h>
#include <algorithm>
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

template <>
void accumulate_hacc_impl<SIMDLevel::AVX2>(
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
            accumulate_hacc_block_avx2(
                    nsq,
                    codes + b * block_stride,
                    lut_lo,
                    lut_hi,
                    accu + q * ntotal2 + b * 32);
        }
    }
}

} // namespace faiss
