/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_AVX512_H
#define HAMMING_COMPUTER_AVX512_H

// AVX512 HammingComputer and GenHammingComputer specializations.
// Types without custom AVX512 code inherit from the NONE specializations
// in hamming_computer-generic.h. Custom specializations for
// HammingComputer64 and HammingComputerDefault use _mm512_popcnt_epi64
// when __AVX512VPOPCNTDQ__ is available. GenHammingComputer classes
// leverage SSE/AVX2 intrinsics.

#include <cassert>
#include <cstdint>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/hamming_distance/hamming_computer-generic.h>

#include <immintrin.h>

namespace faiss {

/***************************************************************************
 * AVX512 inheriting specializations for types without custom AVX512 code.
 * These explicitly inherit the scalar (NONE) implementation so that
 * every SIMDLevel has a concrete specialization.
 ***************************************************************************/

#define FAISS_INHERIT_HAMMING(Class)                                       \
    template <>                                                            \
    struct Class##_tpl<SIMDLevel::AVX512> : Class##_tpl<SIMDLevel::NONE> { \
        using Class##_tpl<SIMDLevel::NONE>::Class##_tpl;                   \
    }

FAISS_INHERIT_HAMMING(HammingComputer16);
FAISS_INHERIT_HAMMING(HammingComputer20);
FAISS_INHERIT_HAMMING(HammingComputer32);
FAISS_INHERIT_HAMMING(GenHammingComputer8);

#undef FAISS_INHERIT_HAMMING

/***************************************************************************
 * Custom AVX512 specializations.
 ***************************************************************************/

template <>
struct HammingComputer64_tpl<SIMDLevel::AVX512> {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;
    const uint64_t* a;

    HammingComputer64_tpl() {}

    HammingComputer64_tpl(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 64);
        a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
        a4 = a[4];
        a5 = a[5];
        a6 = a[6];
        a7 = a[7];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
#ifdef __AVX512VPOPCNTDQ__
        __m512i vxor =
                _mm512_xor_si512(_mm512_loadu_si512(a), _mm512_loadu_si512(b));
        __m512i vpcnt = _mm512_popcnt_epi64(vxor);
        // reduce performs better than adding the lower and higher parts
        return _mm512_reduce_add_epi32(vpcnt);
#else
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
                popcount64(b[2] ^ a2) + popcount64(b[3] ^ a3) +
                popcount64(b[4] ^ a4) + popcount64(b[5] ^ a5) +
                popcount64(b[6] ^ a6) + popcount64(b[7] ^ a7);
#endif
    }

    inline static constexpr int get_code_size() {
        return 64;
    }
};

template <>
struct HammingComputerDefault_tpl<SIMDLevel::AVX512> {
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
#ifdef __AVX512VPOPCNTDQ__
        int quotient64 = quotient8 / 8;
        for (; i < quotient64; ++i) {
            __m512i vxor = _mm512_xor_si512(
                    _mm512_loadu_si512(&a64[i * 8]),
                    _mm512_loadu_si512(&b64[i * 8]));
            __m512i vpcnt = _mm512_popcnt_epi64(vxor);
            // reduce performs better than adding the lower and higher parts
            accu += _mm512_reduce_add_epi32(vpcnt);
        }
        i *= 8;
#endif
        accu += hamming_popcount_tail(
                a64, b64, i, quotient8, a8, b8, remainder8);
        return accu;
    }

    inline int get_code_size() const {
        return quotient8 * 8 + remainder8;
    }
};

// I'm not sure whether this version is faster of slower, tbh
// todo: test on different CPUs
template <>
struct GenHammingComputer16_tpl<SIMDLevel::AVX512> {
    __m128i a;

    GenHammingComputer16_tpl(
            const uint8_t* a8,
            FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        a = _mm_loadu_si128((const __m128i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m128i b = _mm_loadu_si128((const __m128i_u*)b8);
        const __m128i cmp = _mm_cmpeq_epi8(a, b);
        const auto movemask = _mm_movemask_epi8(cmp);
        return 16 - popcount32(movemask);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct GenHammingComputer32_tpl<SIMDLevel::AVX512> {
    __m256i a;

    GenHammingComputer32_tpl(
            const uint8_t* a8,
            FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        a = _mm256_loadu_si256((const __m256i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m256i b = _mm256_loadu_si256((const __m256i_u*)b8);
        const __m256i cmp = _mm256_cmpeq_epi8(a, b);
        const uint32_t movemask = _mm256_movemask_epi8(cmp);
        return 32 - popcount32(movemask);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

// A specialized version might be needed for the very long
// GenHamming code_size. In such a case, one may accumulate
// counts using _mm256_sub_epi8 and then compute a horizontal
// sum (using _mm256_sad_epu8, maybe, in blocks of no larger
// than 256 * 32 bytes).

template <>
struct GenHammingComputerM8_tpl<SIMDLevel::AVX512> {
    const uint64_t* a;
    int n;

    GenHammingComputerM8_tpl(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = (uint64_t*)a8;
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        int accu = 0;

        int i = 0;
        int n4 = (n / 4) * 4;
        for (; i < n4; i += 4) {
            const __m256i av = _mm256_loadu_si256((const __m256i_u*)(a + i));
            const __m256i bv = _mm256_loadu_si256((const __m256i_u*)(b + i));
            const __m256i cmp = _mm256_cmpeq_epi8(av, bv);
            const uint32_t movemask = _mm256_movemask_epi8(cmp);
            accu += 32 - popcount32(movemask);
        }

        for (; i < n; i++)
            accu += generalized_hamming_64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 8;
    }
};

} // namespace faiss

#endif
