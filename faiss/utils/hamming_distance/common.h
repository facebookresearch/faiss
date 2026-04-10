/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_hamming_common_h
#define FAISS_hamming_common_h

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

/* The Hamming distance type */
using hamdis_t = int32_t;

namespace faiss {

// trust the compiler to provide efficient popcount implementations
inline int popcount32(uint32_t x) {
    return __builtin_popcount(x);
}

// popcount
inline int popcount64(uint64_t x) {
    return __builtin_popcountl(x);
}

// This table was moved from .cpp to .h file, because
// otherwise it was causing compilation errors while trying to
// compile swig modules on Windows.
inline constexpr uint8_t hamdis_tab_ham_bytes[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
        2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4,
        2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5,
        3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

/***************************************************************************
 * Universal code: identical across all architectures
 ***************************************************************************/

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

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 4);
        const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
        a0 = *a32;
    }

    inline int hamming(const uint8_t* b) const {
        const uint32_t* b32 = reinterpret_cast<const uint32_t*>(b);
        return popcount64(*b32 ^ a0);
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

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 8);
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a);
        a0 = *a64;
    }

    inline int hamming(const uint8_t* b) const {
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b);
        return popcount64(*b64 ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};

/* Duff's device + byte remainder tail for HammingComputerDefault.
 * Processes uint64 words starting at index i_start using popcount,
 * then handles any remaining bytes via lookup table. */
inline int hamming_popcount_tail(
        const uint64_t* a64,
        const uint64_t* b64,
        int i_start,
        int quotient8,
        const uint8_t* a8,
        const uint8_t* b8,
        int remainder8) {
    int accu = 0;
    int i = i_start;
    int len = quotient8 - i_start;
    switch (len & 7) {
        default:
            while (len > 7) {
                len -= 8;
                accu += popcount64(a64[i] ^ b64[i]);
                i++;
                [[fallthrough]];
                case 7:
                    accu += popcount64(a64[i] ^ b64[i]);
                    i++;
                    [[fallthrough]];
                case 6:
                    accu += popcount64(a64[i] ^ b64[i]);
                    i++;
                    [[fallthrough]];
                case 5:
                    accu += popcount64(a64[i] ^ b64[i]);
                    i++;
                    [[fallthrough]];
                case 4:
                    accu += popcount64(a64[i] ^ b64[i]);
                    i++;
                    [[fallthrough]];
                case 3:
                    accu += popcount64(a64[i] ^ b64[i]);
                    i++;
                    [[fallthrough]];
                case 2:
                    accu += popcount64(a64[i] ^ b64[i]);
                    i++;
                    [[fallthrough]];
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
                [[fallthrough]];
            case 6:
                accu += hamdis_tab_ham_bytes[a[5] ^ b[5]];
                [[fallthrough]];
            case 5:
                accu += hamdis_tab_ham_bytes[a[4] ^ b[4]];
                [[fallthrough]];
            case 4:
                accu += hamdis_tab_ham_bytes[a[3] ^ b[3]];
                [[fallthrough]];
            case 3:
                accu += hamdis_tab_ham_bytes[a[2] ^ b[2]];
                [[fallthrough]];
            case 2:
                accu += hamdis_tab_ham_bytes[a[1] ^ b[1]];
                [[fallthrough]];
            case 1:
                accu += hamdis_tab_ham_bytes[a[0] ^ b[0]];
                break;
            default:
                break;
        }
    }
    return accu;
}

/***************************************************************************
 * Scalar code shared by generic, AVX2, and AVX512 backends.
 * NEON provides its own optimized versions of these.
 ***************************************************************************/

#ifndef __aarch64__

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

struct HammingComputer16 {
    uint64_t a0, a1;

    HammingComputer16() {}

    HammingComputer16(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
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

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 20);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a8);
        a0 = a[0];
        a1 = a[1];
        // can't read a[2] since it is uint64_t, not uint32_t
        // results in AddressSanitizer failure reading past end of array
        a2 = a32[4];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        const uint32_t* b32_tail = reinterpret_cast<const uint32_t*>(b + 2);
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
                popcount64(*b32_tail ^ a2);
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

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
                popcount64(b[2] ^ a2) + popcount64(b[3] ^ a3);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

struct GenHammingComputer8 {
    uint64_t a0;

    GenHammingComputer8(const uint8_t* a, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 8);
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a);
        a0 = *a64;
    }

    inline int hamming(const uint8_t* b) const {
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b);
        return generalized_hamming_64(*b64 ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};

#endif // !__aarch64__

/***************************************************************************
 * Scalar code shared by generic and AVX2 backends only.
 * AVX512 and NEON provide their own optimized versions.
 ***************************************************************************/

#if !defined(__aarch64__) && !defined(__AVX512F__)

struct HammingComputer64 {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;

    HammingComputer64() {}

    HammingComputer64(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 64);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
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

    HammingComputerDefault(const uint8_t* a8_in, int code_size) {
        set(a8_in, code_size);
    }

    void set(const uint8_t* a8_in, int code_size) {
        this->a8 = a8_in;
        quotient8 = code_size / 8;
        remainder8 = code_size % 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);
        return hamming_popcount_tail(
                a64, b64, 0, quotient8, a8, b8, remainder8);
    }

    inline int get_code_size() const {
        return quotient8 * 8 + remainder8;
    }
};

#endif // !__aarch64__ && !__AVX512F__

} // namespace faiss

#endif
