/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_AVX512_VPOPCNT_H
#define HAMMING_COMPUTER_AVX512_VPOPCNT_H

// AVX512_VPOPCNT HammingComputer specializations. The 32/64/Default kernels
// use VPOPCNTDQ; the batched 20-byte kernel uses AVX512_BITALG. This gives
// a faster path than the scalar popcount fallback used in the base AVX512
// specializations when compiled without -mavx512vpopcntdq.

#include <cassert>
#include <cstdint>
#include <cstring>

#include <faiss/utils/popcount.h>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/hamming_distance/hamming_computer-avx512.h>

#include <immintrin.h>

namespace faiss {

/***************************************************************************
 * AVX512_VPOPCNT inheriting specializations without custom VPOPCNT code.
 ***************************************************************************/

#define FAISS_INHERIT_HAMMING_VPOPCNT(Class)                                   \
    template <>                                                                \
    struct Class##                                                             \
            _tpl<SIMDLevel::AVX512_VPOPCNT> : Class##_tpl<SIMDLevel::AVX512> { \
        using Class##_tpl<SIMDLevel::AVX512>::Class##_tpl;                     \
    }

FAISS_INHERIT_HAMMING_VPOPCNT(HammingComputer16);
FAISS_INHERIT_HAMMING_VPOPCNT(GenHammingComputer8);
FAISS_INHERIT_HAMMING_VPOPCNT(GenHammingComputer16);
FAISS_INHERIT_HAMMING_VPOPCNT(GenHammingComputer32);
FAISS_INHERIT_HAMMING_VPOPCNT(GenHammingComputerM8);

#undef FAISS_INHERIT_HAMMING_VPOPCNT

/***************************************************************************
 * Custom AVX512_VPOPCNT specializations using VPOPCNTDQ.
 ***************************************************************************/

template <>
struct HammingComputer20_tpl<SIMDLevel::AVX512_VPOPCNT>
        : HammingComputer20_tpl<SIMDLevel::AVX512> {
    using HammingComputer20_tpl<SIMDLevel::AVX512>::HammingComputer20_tpl;

    static constexpr size_t batch_size = 8;
    static constexpr size_t kStride = get_code_size();
    // 160 bytes is what the three loads in hamming_batch() cover, and the
    // 16+4 or 4+16 split it applies per lane is written out for 20 bytes:
    // the literal offsets and the two group indices are not derived from
    // kStride, so another width needs the body reworked, not just retuned.
    static_assert(batch_size * kStride == 160);
    static_assert(kStride == 20, "hamming_batch() hardcodes the 16+4 split");
    static constexpr __mmask64 kTailMask = 0xFFFFFFFFull;

    /// Writes the query repeated batch_size times. The caller owns the buffer,
    /// so a computer used only through hamming() carries no batch state.
    static void build_batch_query(const uint8_t* a8, uint8_t* tile) {
        for (size_t k = 0; k < batch_size; k++) {
            memcpy(tile + k * kStride, a8, kStride);
        }
    }

    static void hamming_batch(
            const uint8_t* tile,
            const uint8_t* codes,
            int32_t* dis) {
        const __m512i zero = _mm512_setzero_si512();
        const __m512i p0 = _mm512_popcnt_epi8(_mm512_xor_si512(
                _mm512_loadu_si512(codes), _mm512_loadu_si512(tile)));
        const __m512i p1 = _mm512_popcnt_epi8(_mm512_xor_si512(
                _mm512_loadu_si512(codes + 64), _mm512_loadu_si512(tile + 64)));
        const __m512i p2 = _mm512_popcnt_epi8(_mm512_xor_si512(
                _mm512_maskz_loadu_epi8(kTailMask, codes + 128),
                _mm512_maskz_loadu_epi8(kTailMask, tile + 128)));

        alignas(64) uint64_t grp[24];
        _mm512_store_si512(grp, _mm512_sad_epu8(p0, zero));
        _mm512_store_si512(grp + 8, _mm512_sad_epu8(p1, zero));
        _mm512_store_si512(grp + 16, _mm512_sad_epu8(p2, zero));

        for (size_t k = 0; k < batch_size; k++) {
            const size_t s = k * kStride;
            const size_t g = s / 8;
            uint32_t xh, qh;
            if (s % 8 == 0) {
                memcpy(&xh, codes + s + 16, 4);
                memcpy(&qh, tile + s + 16, 4);
                dis[k] = static_cast<int32_t>(
                        grp[g] + grp[g + 1] + popcount32(xh ^ qh));
            } else {
                memcpy(&xh, codes + s, 4);
                memcpy(&qh, tile + s, 4);
                dis[k] = static_cast<int32_t>(
                        popcount32(xh ^ qh) + grp[g + 1] + grp[g + 2]);
            }
        }
    }
};

template <>
struct HammingComputer32_tpl<SIMDLevel::AVX512_VPOPCNT> {
    const uint8_t* a8;

    HammingComputer32_tpl() {}

    HammingComputer32_tpl(const uint8_t* a8_in, int code_size) {
        set(a8_in, code_size);
    }

    void set(const uint8_t* a8_in, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        a8 = a8_in;
    }

    inline int hamming(const uint8_t* b8) const {
        __m256i va = _mm256_loadu_si256((const __m256i*)a8);
        __m256i vb = _mm256_loadu_si256((const __m256i*)b8);
        __m256i vxor = _mm256_xor_si256(va, vb);
        __m256i vpcnt = _mm256_popcnt_epi64(vxor);
        __m128i lo = _mm256_castsi256_si128(vpcnt);
        __m128i hi = _mm256_extracti128_si256(vpcnt, 1);
        __m128i sum = _mm_add_epi64(lo, hi);
        return static_cast<int>(
                _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1));
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct HammingComputer64_tpl<SIMDLevel::AVX512_VPOPCNT> {
    const uint8_t* a8;

    HammingComputer64_tpl() {}

    HammingComputer64_tpl(const uint8_t* a8_in, int code_size) {
        set(a8_in, code_size);
    }

    void set(const uint8_t* a8_in, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 64);
        a8 = a8_in;
    }

    inline int hamming(const uint8_t* b8) const {
        __m512i vxor = _mm512_xor_si512(
                _mm512_loadu_si512(a8), _mm512_loadu_si512(b8));
        __m512i vpcnt = _mm512_popcnt_epi64(vxor);
        return _mm512_reduce_add_epi32(vpcnt);
    }

    inline static constexpr int get_code_size() {
        return 64;
    }
};

template <>
struct HammingComputerDefault_tpl<SIMDLevel::AVX512_VPOPCNT> {
    const uint8_t* a8;
    int quotient8;
    int remainder8;

    HammingComputerDefault_tpl() {}

    HammingComputerDefault_tpl(const uint8_t* a8_in, int code_size) {
        set(a8_in, code_size);
    }

    void set(const uint8_t* a8_2, int code_size) {
        this->a8 = a8_2;
        quotient8 = code_size / 8;
        remainder8 = code_size % 8;
    }

    int hamming(const uint8_t* b8) const {
        int accu = 0;

        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);

        int i = 0;
        int quotient64 = quotient8 / 8;
        for (; i < quotient64; ++i) {
            __m512i vxor = _mm512_xor_si512(
                    _mm512_loadu_si512(&a64[i * 8]),
                    _mm512_loadu_si512(&b64[i * 8]));
            __m512i vpcnt = _mm512_popcnt_epi64(vxor);
            accu += _mm512_reduce_add_epi32(vpcnt);
        }
        i *= 8;

        // Handle 4-word (256-bit) remainder with VPOPCNTDQ VL
        if (i + 4 <= quotient8) {
            __m256i vxor = _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i*)&a64[i]),
                    _mm256_loadu_si256((const __m256i*)&b64[i]));
            __m256i vpcnt = _mm256_popcnt_epi64(vxor);
            __m128i lo = _mm256_castsi256_si128(vpcnt);
            __m128i hi = _mm256_extracti128_si256(vpcnt, 1);
            __m128i sum = _mm_add_epi64(lo, hi);
            accu += static_cast<int>(
                    _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1));
            i += 4;
        }

        accu += hamming_popcount_tail(
                a64, b64, i, quotient8, a8, b8, remainder8);
        return accu;
    }

    inline int get_code_size() const {
        return quotient8 * 8 + remainder8;
    }
};

} // namespace faiss

#endif
