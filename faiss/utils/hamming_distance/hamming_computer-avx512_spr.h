/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_AVX512_SPR_H
#define HAMMING_COMPUTER_AVX512_SPR_H

// AVX512_SPR HammingComputer specializations using VPOPCNTDQ.
// On Sapphire Rapids+, _mm512_popcnt_epi64 (and _mm256_popcnt_epi64 with VL)
// are unconditionally available. This gives a faster path than the scalar
// popcount fallback used in the base AVX512 specializations when compiled
// without -mavx512vpopcntdq.

#include <cassert>
#include <cstdint>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/hamming_distance/hamming_computer-avx512.h>

#include <immintrin.h>

namespace faiss {

/***************************************************************************
 * AVX512_SPR inheriting specializations for types without custom SPR code.
 ***************************************************************************/

#define FAISS_INHERIT_HAMMING_SPR(Class)                                   \
    template <>                                                            \
    struct Class##                                                         \
            _tpl<SIMDLevel::AVX512_SPR> : Class##_tpl<SIMDLevel::AVX512> { \
        using Class##_tpl<SIMDLevel::AVX512>::Class##_tpl;                 \
    }

FAISS_INHERIT_HAMMING_SPR(HammingComputer16);
FAISS_INHERIT_HAMMING_SPR(HammingComputer20);
FAISS_INHERIT_HAMMING_SPR(GenHammingComputer8);
FAISS_INHERIT_HAMMING_SPR(GenHammingComputer16);
FAISS_INHERIT_HAMMING_SPR(GenHammingComputer32);
FAISS_INHERIT_HAMMING_SPR(GenHammingComputerM8);

#undef FAISS_INHERIT_HAMMING_SPR

/***************************************************************************
 * Custom AVX512_SPR specializations using VPOPCNTDQ.
 ***************************************************************************/

template <>
struct HammingComputer32_tpl<SIMDLevel::AVX512_SPR> {
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
struct HammingComputer64_tpl<SIMDLevel::AVX512_SPR> {
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
struct HammingComputerDefault_tpl<SIMDLevel::AVX512_SPR> {
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
