/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_AVX2_INL_H
#define HAMMING_AVX2_INL_H

// AVX2 version

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

#include <immintrin.h>

namespace faiss {

/* Elementary Hamming distance computation: unoptimized  */
template <size_t nbits, typename T>
inline T hamming(const uint8_t* bs1, const uint8_t* bs2) {
    const size_t nbytes = nbits / 8;
    size_t i;
    T h = 0;
    for (i = 0; i < nbytes; i++) {
        h += (T)hamdis_tab_ham_bytes[bs1[i] ^ bs2[i]];
    }
    return h;
}

/* Hamming distances for multiples of 64 bits */
template <size_t nbits>
inline hamdis_t hamming(const uint64_t* bs1, const uint64_t* bs2) {
    const size_t nwords = nbits / 64;
    size_t i;
    hamdis_t h = 0;
    for (i = 0; i < nwords; i++) {
        h += popcount64(bs1[i] ^ bs2[i]);
    }
    return h;
}

/* specialized (optimized) functions */
template <>
inline hamdis_t hamming<64>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]);
}

template <>
inline hamdis_t hamming<128>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]);
}

template <>
inline hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]) +
            popcount64(pa[2] ^ pb[2]) + popcount64(pa[3] ^ pb[3]);
}

/* Hamming distances for multiple of 64 bits */
inline hamdis_t hamming(
        const uint64_t* bs1,
        const uint64_t* bs2,
        size_t nwords) {
    hamdis_t h = 0;
    for (size_t i = 0; i < nwords; i++) {
        h += popcount64(bs1[i] ^ bs2[i]);
    }
    return h;
}

/******************************************************************
 * The HammingComputer series of classes compares a single code of
 * size 4 to 32 to incoming codes. They are intended for use as a
 * template class where it would be inefficient to switch on the code
 * size in the inner loop. Hopefully the compiler will inline the
 * hamming() functions and put the a0, a1, ... in registers.
 ******************************************************************/

struct HammingComputer4 {
    uint32_t a0;

    HammingComputer4() {}

    HammingComputer4(const uint8_t* a, int code_size) {
        set(a, code_size);
    }

    void set(const uint8_t* a, int code_size) {
        assert(code_size == 4);
        a0 = *(uint32_t*)a;
    }

    inline int hamming(const uint8_t* b) const {
        return popcount64(*(uint32_t*)b ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 4;
    }
};

struct HammingComputer8 {
    uint64_t a0;

    HammingComputer8() {}

    HammingComputer8(const uint8_t* a, int code_size) {
        set(a, code_size);
    }

    void set(const uint8_t* a, int code_size) {
        assert(code_size == 8);
        a0 = *(uint64_t*)a;
    }

    inline int hamming(const uint8_t* b) const {
        return popcount64(*(uint64_t*)b ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};

struct HammingComputer16 {
    uint64_t a0, a1;

    HammingComputer16() {}

    HammingComputer16(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 16);
        const uint64_t* a = (uint64_t*)a8;
        a0 = a[0];
        a1 = a[1];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

// when applied to an array, 1/2 of the 64-bit accesses are unaligned.
// This incurs a penalty of ~10% wrt. fully aligned accesses.
struct HammingComputer20 {
    uint64_t a0, a1;
    uint32_t a2;

    HammingComputer20() {}

    HammingComputer20(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 20);
        const uint64_t* a = (uint64_t*)a8;
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
                popcount64(*(uint32_t*)(b + 2) ^ a2);
    }

    inline static constexpr int get_code_size() {
        return 20;
    }
};

struct HammingComputer32 {
    uint64_t a0, a1, a2, a3;

    HammingComputer32() {}

    HammingComputer32(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 32);
        const uint64_t* a = (uint64_t*)a8;
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
                popcount64(b[2] ^ a2) + popcount64(b[3] ^ a3);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

struct HammingComputer64 {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;

    HammingComputer64() {}

    HammingComputer64(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 64);
        const uint64_t* a = (uint64_t*)a8;
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
        const uint64_t* b = (uint64_t*)b8;
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
                popcount64(b[2] ^ a2) + popcount64(b[3] ^ a3) +
                popcount64(b[4] ^ a4) + popcount64(b[5] ^ a5) +
                popcount64(b[6] ^ a6) + popcount64(b[7] ^ a7);
    }

    inline static constexpr int get_code_size() {
        return 64;
    }
};

struct HammingComputerDefault {
    const uint8_t* a8;
    int quotient8;
    int remainder8;

    HammingComputerDefault() {}

    HammingComputerDefault(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        this->a8 = a8;
        quotient8 = code_size / 8;
        remainder8 = code_size % 8;
    }

    int hamming(const uint8_t* b8) const {
        int accu = 0;

        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);
        int i = 0, len = quotient8;
        switch (len & 7) {
            default:
                while (len > 7) {
                    len -= 8;
                    accu += popcount64(a64[i] ^ b64[i]);
                    i++;
                    case 7:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                    case 6:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                    case 5:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                    case 4:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                    case 3:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                    case 2:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                    case 1:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                }
        }
        if (remainder8) {
            const uint8_t* a = a8 + 8 * quotient8;
            const uint8_t* b = b8 + 8 * quotient8;
            switch (remainder8) {
                case 7:
                    accu += hamdis_tab_ham_bytes[a[6] ^ b[6]];
                case 6:
                    accu += hamdis_tab_ham_bytes[a[5] ^ b[5]];
                case 5:
                    accu += hamdis_tab_ham_bytes[a[4] ^ b[4]];
                case 4:
                    accu += hamdis_tab_ham_bytes[a[3] ^ b[3]];
                case 3:
                    accu += hamdis_tab_ham_bytes[a[2] ^ b[2]];
                case 2:
                    accu += hamdis_tab_ham_bytes[a[1] ^ b[1]];
                case 1:
                    accu += hamdis_tab_ham_bytes[a[0] ^ b[0]];
                default:
                    break;
            }
        }

        return accu;
    }

    inline int get_code_size() const {
        return quotient8 * 8 + remainder8;
    }
};

// more inefficient than HammingComputerDefault (obsolete)
struct HammingComputerM8 {
    const uint64_t* a;
    int n;

    HammingComputerM8() {}

    HammingComputerM8(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = (uint64_t*)a8;
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        int accu = 0;
        for (int i = 0; i < n; i++)
            accu += popcount64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 8;
    }
};

// more inefficient than HammingComputerDefault (obsolete)
struct HammingComputerM4 {
    const uint32_t* a;
    int n;

    HammingComputerM4() {}

    HammingComputerM4(const uint8_t* a4, int code_size) {
        set(a4, code_size);
    }

    void set(const uint8_t* a4, int code_size) {
        assert(code_size % 4 == 0);
        a = (uint32_t*)a4;
        n = code_size / 4;
    }

    int hamming(const uint8_t* b8) const {
        const uint32_t* b = (uint32_t*)b8;
        int accu = 0;
        for (int i = 0; i < n; i++)
            accu += popcount64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 4;
    }
};

/***************************************************************************
 * Equivalence with a template class when code size is known at compile time
 **************************************************************************/

// default template
template <int CODE_SIZE>
struct HammingComputer : HammingComputerDefault {
    HammingComputer(const uint8_t* a, int code_size)
            : HammingComputerDefault(a, code_size) {}
};

#define SPECIALIZED_HC(CODE_SIZE)                                    \
    template <>                                                      \
    struct HammingComputer<CODE_SIZE> : HammingComputer##CODE_SIZE { \
        HammingComputer(const uint8_t* a)                            \
                : HammingComputer##CODE_SIZE(a, CODE_SIZE) {}        \
    }

SPECIALIZED_HC(4);
SPECIALIZED_HC(8);
SPECIALIZED_HC(16);
SPECIALIZED_HC(20);
SPECIALIZED_HC(32);
SPECIALIZED_HC(64);

#undef SPECIALIZED_HC

/***************************************************************************
 * generalized Hamming = number of bytes that are different between
 * two codes.
 ***************************************************************************/

inline int generalized_hamming_64(uint64_t a) {
    a |= a >> 1;
    a |= a >> 2;
    a |= a >> 4;
    a &= 0x0101010101010101UL;
    return popcount64(a);
}

struct GenHammingComputer8 {
    uint64_t a0;

    GenHammingComputer8(const uint8_t* a, int code_size) {
        assert(code_size == 8);
        a0 = *(uint64_t*)a;
    }

    inline int hamming(const uint8_t* b) const {
        return generalized_hamming_64(*(uint64_t*)b ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};

// I'm not sure whether this version is faster of slower, tbh
// todo: test on different CPUs
struct GenHammingComputer16 {
    __m128i a;

    GenHammingComputer16(const uint8_t* a8, int code_size) {
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

struct GenHammingComputer32 {
    __m256i a;

    GenHammingComputer32(const uint8_t* a8, int code_size) {
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

struct GenHammingComputerM8 {
    const uint64_t* a;
    int n;

    GenHammingComputerM8(const uint8_t* a8, int code_size) {
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
