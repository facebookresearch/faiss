/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef LEVEL2_LSX_INL_H
#define LEVEL2_LSX_INL_H

#include <lsxintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace faiss {
namespace cppcontrib {

template <
        intptr_t DIM,
        intptr_t COARSE_SIZE,
        intptr_t FINE_SIZE,
        intptr_t COARSE_BITS = 8,
        intptr_t FINE_BITS = 8>
struct Index2LevelDecoder {
    static_assert(DIM > 0 && FINE_SIZE > 0 && DIM % FINE_SIZE == 0);
    static_assert(
            FINE_BITS == 8 || FINE_BITS == 10 || FINE_BITS == 12 ||
            FINE_BITS == 16);
    static constexpr intptr_t dim = DIM, fineSize = FINE_SIZE,
                              fineBits = FINE_BITS;
    static_assert(COARSE_SIZE > 0 && DIM % COARSE_SIZE == 0);
    static_assert(
            COARSE_BITS == 8 || COARSE_BITS == 10 || COARSE_BITS == 12 ||
            COARSE_BITS == 16);
    static constexpr intptr_t coarseSize = COARSE_SIZE,
                              coarseBits = COARSE_BITS;

   private:
    // Read exactly the bytes of a packed field; code pointers may be unaligned.
    template <int Bits>
    static uint32_t read_code(const uint8_t* code, intptr_t index) {
        const intptr_t bit = index * Bits, shift = bit % 8;
        uint32_t word = 0;
        std::memcpy(&word, code + bit / 8, (shift + Bits + 7) / 8);
        return (word >> shift) & ((1u << Bits) - 1);
    }
    template <intptr_t Size, int Bits>
    static __m128 decode_block(
            const float* table,
            const uint8_t* code,
            intptr_t pos,
            intptr_t count) {
        if (count == 4 && pos % Size + 4 <= Size) {
            intptr_t sub = pos / Size;
            return (__m128)__lsx_vld(
                    table +
                            (sub * (intptr_t(1) << Bits) +
                             read_code<Bits>(code, sub)) *
                                    Size +
                            pos % Size,
                    0);
        }
        // LSX has no gather instruction; copy contiguous centroid slices.
        float values[4] = {};
        intptr_t lane = 0;
        while (lane < count) {
            intptr_t i = pos + lane, sub = i / Size,
                     n = std::min(count - lane, Size - i % Size);
            const float* src = table +
                    (sub * (intptr_t(1) << Bits) + read_code<Bits>(code, sub)) *
                            Size +
                    i % Size;
            std::memcpy(values + lane, src, n * sizeof(float));
            lane += n;
        }
        return (__m128)__lsx_vld(values, 0);
    }
    static __m128 load_output(const float* x, intptr_t count) {
        if (count == 4)
            return (__m128)__lsx_vld(x, 0);
        float values[4] = {};
        std::memcpy(values, x, count * sizeof(float));
        return (__m128)__lsx_vld(values, 0);
    }
    static void store_output(__m128 v, float* x, intptr_t count) {
        if (count == 4) {
            __lsx_vst((__m128i)v, x, 0);
            return;
        }
        float values[4];
        __lsx_vst((__m128i)v, values, 0);
        std::memcpy(x, values, count * sizeof(float));
    }

    static constexpr intptr_t coarse_bytes =
            (DIM / COARSE_SIZE * COARSE_BITS + 7) / 8;
    static __m128 block(
            const float* coarse,
            const float* fine,
            const uint8_t* code,
            intptr_t pos,
            intptr_t count) {
        return __lsx_vfadd_s(
                decode_block<COARSE_SIZE, COARSE_BITS>(
                        coarse, code, pos, count),
                decode_block<FINE_SIZE, FINE_BITS>(
                        fine, code + coarse_bytes, pos, count));
    }
    template <int N>
    static void accumulate(
            const float* const (&coarse)[N],
            const float* const (&fine)[N],
            const uint8_t* const (&codes)[N],
            const float (&weights)[N],
            float* out) {
        for (intptr_t pos = 0; pos < DIM; pos += 4) {
            intptr_t count = std::min(intptr_t(4), DIM - pos);
            __m128 acc = load_output(out + pos, count);
            for (int j = 0; j < N; ++j) {
                const __m128 weight = (__m128)__lsx_vreplgr2vr_w(
                        std::bit_cast<int32_t>(weights[j]));
                acc = __lsx_vfmadd_s(
                        weight,
                        block(coarse[j], fine[j], codes[j], pos, count),
                        acc);
            }
            store_output(acc, out + pos, count);
        }
    }

   public:
    static void store(
            const float* coarse,
            const float* fine,
            const uint8_t* code,
            float* out) {
        for (intptr_t pos = 0; pos < DIM; pos += 4) {
            intptr_t count = std::min(intptr_t(4), DIM - pos);
            store_output(
                    block(coarse, fine, code, pos, count), out + pos, count);
        }
    }
    static void accum(
            const float* coarse0,
            const float* fine0,
            const uint8_t* code0,
            float weight0,
            float* out) {
        const float* coarse_tables[] = {coarse0};
        const float* fine_tables[] = {fine0};
        const uint8_t* codes[] = {code0};
        const float weights[] = {weight0};
        accumulate(coarse_tables, fine_tables, codes, weights, out);
    }
    static void accum(
            const float* coarse0,
            const float* fine0,
            const uint8_t* code0,
            float weight0,
            const float* coarse1,
            const float* fine1,
            const uint8_t* code1,
            float weight1,
            float* out) {
        const float* coarse_tables[] = {coarse0, coarse1};
        const float* fine_tables[] = {fine0, fine1};
        const uint8_t* codes[] = {code0, code1};
        const float weights[] = {weight0, weight1};
        accumulate(coarse_tables, fine_tables, codes, weights, out);
    }
    static void accum(
            const float* coarse,
            const float* fine,
            const uint8_t* code0,
            float weight0,
            const uint8_t* code1,
            float weight1,
            float* out) {
        const float* coarse_tables[] = {coarse, coarse};
        const float* fine_tables[] = {fine, fine};
        const uint8_t* codes[] = {code0, code1};
        const float weights[] = {weight0, weight1};
        accumulate(coarse_tables, fine_tables, codes, weights, out);
    }
    static void accum(
            const float* coarse0,
            const float* fine0,
            const uint8_t* code0,
            float weight0,
            const float* coarse1,
            const float* fine1,
            const uint8_t* code1,
            float weight1,
            const float* coarse2,
            const float* fine2,
            const uint8_t* code2,
            float weight2,
            float* out) {
        const float* coarse_tables[] = {coarse0, coarse1, coarse2};
        const float* fine_tables[] = {fine0, fine1, fine2};
        const uint8_t* codes[] = {code0, code1, code2};
        const float weights[] = {weight0, weight1, weight2};
        accumulate(coarse_tables, fine_tables, codes, weights, out);
    }
    static void accum(
            const float* coarse,
            const float* fine,
            const uint8_t* code0,
            float weight0,
            const uint8_t* code1,
            float weight1,
            const uint8_t* code2,
            float weight2,
            float* out) {
        const float* coarse_tables[] = {coarse, coarse, coarse};
        const float* fine_tables[] = {fine, fine, fine};
        const uint8_t* codes[] = {code0, code1, code2};
        const float weights[] = {weight0, weight1, weight2};
        accumulate(coarse_tables, fine_tables, codes, weights, out);
    }
};

} // namespace cppcontrib
} // namespace faiss
#endif // LEVEL2_LSX_INL_H
