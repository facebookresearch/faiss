/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_NEON_INL_H
#define HAMMING_NEON_INL_H

// a specialized version of hamming is needed here, because both
// gcc, clang and msvc seem to generate suboptimal code sometimes.

#ifdef __aarch64__

#include <arm_neon.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

#include <faiss/utils/hamming_distance/common.h>

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
inline hamdis_t hamming(const uint64_t* pa, const uint64_t* pb) {
    constexpr size_t nwords256 = nbits / 256;
    constexpr size_t nwords128 = (nbits - nwords256 * 256) / 128;
    constexpr size_t nwords64 =
            (nbits - nwords256 * 256 - nwords128 * 128) / 64;

    hamdis_t h = 0;
    if (nwords256 > 0) {
        for (size_t i = 0; i < nwords256; i++) {
            h += hamming<256>(pa, pb);
            pa += 4;
            pb += 4;
        }
    }

    if (nwords128 > 0) {
        h += hamming<128>(pa, pb);
        pa += 2;
        pb += 2;
    }

    if (nwords64 > 0) {
        h += hamming<64>(pa, pb);
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
    const uint8_t* pa8 = reinterpret_cast<const uint8_t*>(pa);
    const uint8_t* pb8 = reinterpret_cast<const uint8_t*>(pb);
    uint8x16_t or0 = veorq_u8(vld1q_u8(pa8), vld1q_u8(pb8));
    uint8x16_t c0 = vcntq_u8(or0);
    auto dis = vaddvq_u8(c0);
    return dis;
}

template <>
inline hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    const uint8_t* pa8 = reinterpret_cast<const uint8_t*>(pa);
    const uint8_t* pb8 = reinterpret_cast<const uint8_t*>(pb);
    uint8x16_t or0 = veorq_u8(vld1q_u8(pa8), vld1q_u8(pb8));
    uint8x16_t or1 = veorq_u8(vld1q_u8(pa8 + 16), vld1q_u8(pb8 + 16));
    uint8x16_t c0 = vcntq_u8(or0);
    uint8x16_t c1 = vcntq_u8(or1);
    uint8x16_t ca = vpaddq_u8(c0, c1);
    auto dis = vaddvq_u8(ca);
    return dis;
}

/* Hamming distances for multiple of 64 bits */
inline hamdis_t hamming(const uint64_t* pa, const uint64_t* pb, size_t nwords) {
    const size_t nwords256 = nwords / 4;
    const size_t nwords128 = (nwords % 4) / 2;
    const size_t nwords64 = nwords % 2;

    hamdis_t h = 0;
    if (nwords256 > 0) {
        for (size_t i = 0; i < nwords256; i++) {
            h += hamming<256>(pa, pb);
            pa += 4;
            pb += 4;
        }
    }

    if (nwords128 > 0) {
        h += hamming<128>(pa, pb);
        pa += 2;
        pb += 2;
    }

    if (nwords64 > 0) {
        h += hamming<64>(pa, pb);
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
    uint8x16_t a0;

    HammingComputer16() {}

    HammingComputer16(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 16);
        a0 = vld1q_u8(a8);
    }

    inline int hamming(const uint8_t* b8) const {
        uint8x16_t b0 = vld1q_u8(b8);

        uint8x16_t or0 = veorq_u8(a0, b0);
        uint8x16_t c0 = vcntq_u8(or0);
        auto dis = vaddvq_u8(c0);
        return dis;
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

// when applied to an array, 1/2 of the 64-bit accesses are unaligned.
// This incurs a penalty of ~10% wrt. fully aligned accesses.
struct HammingComputer20 {
    uint8x16_t a0;
    uint32_t a2;

    HammingComputer20() {}

    HammingComputer20(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 20);

        a0 = vld1q_u8(a8);

        const uint32_t* a = (uint32_t*)a8;
        a2 = a[4];
    }

    inline int hamming(const uint8_t* b8) const {
        uint8x16_t b0 = vld1q_u8(b8);

        uint8x16_t or0 = veorq_u8(a0, b0);
        uint8x16_t c0 = vcntq_u8(or0);
        auto dis = vaddvq_u8(c0);

        const uint32_t* b = (uint32_t*)b8;
        return dis + popcount64(b[4] ^ a2);
    }

    inline static constexpr int get_code_size() {
        return 20;
    }
};

struct HammingComputer32 {
    uint8x16_t a0;
    uint8x16_t a1;

    HammingComputer32() {}

    HammingComputer32(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 32);
        a0 = vld1q_u8(a8);
        a1 = vld1q_u8(a8 + 16);
    }

    inline int hamming(const uint8_t* b8) const {
        uint8x16_t b0 = vld1q_u8(b8);
        uint8x16_t b1 = vld1q_u8(b8 + 16);

        uint8x16_t or0 = veorq_u8(a0, b0);
        uint8x16_t or1 = veorq_u8(a1, b1);
        uint8x16_t c0 = vcntq_u8(or0);
        uint8x16_t c1 = vcntq_u8(or1);
        uint8x16_t ca = vpaddq_u8(c0, c1);
        auto dis = vaddvq_u8(ca);
        return dis;
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

struct HammingComputer64 {
    HammingComputer32 hc0, hc1;

    HammingComputer64() {}

    HammingComputer64(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 64);
        hc0.set(a8, 32);
        hc1.set(a8 + 32, 32);
    }

    inline int hamming(const uint8_t* b8) const {
        return hc0.hamming(b8) + hc1.hamming(b8 + 32);
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

        int len256 = (quotient8 / 4) * 4;
        for (; i < len256; i += 4) {
            accu += ::faiss::hamming<256>(a64 + i, b64 + i);
            len -= 4;
        }

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
                    [[fallthrough]];
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
    uint8x8_t a0;

    GenHammingComputer8(const uint8_t* a8, int code_size) {
        assert(code_size == 8);
        a0 = vld1_u8(a8);
    }

    inline int hamming(const uint8_t* b8) const {
        uint8x8_t b0 = vld1_u8(b8);
        uint8x8_t reg = vceq_u8(a0, b0);
        uint8x8_t c0 = vcnt_u8(reg);
        return 8 - vaddv_u8(c0) / 8;
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};

struct GenHammingComputer16 {
    uint8x16_t a0;

    GenHammingComputer16(const uint8_t* a8, int code_size) {
        assert(code_size == 16);
        a0 = vld1q_u8(a8);
    }

    inline int hamming(const uint8_t* b8) const {
        uint8x16_t b0 = vld1q_u8(b8);
        uint8x16_t reg = vceqq_u8(a0, b0);
        uint8x16_t c0 = vcntq_u8(reg);
        return 16 - vaddvq_u8(c0) / 8;
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

struct GenHammingComputer32 {
    GenHammingComputer16 a0, a1;

    GenHammingComputer32(const uint8_t* a8, int code_size)
            : a0(a8, 16), a1(a8 + 16, 16) {
        assert(code_size == 32);
    }

    inline int hamming(const uint8_t* b8) const {
        return a0.hamming(b8) + a1.hamming(b8 + 16);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

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

        int n2 = (n / 2) * 2;
        int i = 0;
        for (; i < n2; i += 2) {
            uint8x16_t a0 = vld1q_u8((const uint8_t*)(a + i));
            uint8x16_t b0 = vld1q_u8((const uint8_t*)(b + i));
            uint8x16_t reg = vceqq_u8(a0, b0);
            uint8x16_t c0 = vcntq_u8(reg);
            auto dis = 16 - vaddvq_u8(c0) / 8;
            accu += dis;
        }

        for (; i < n; i++) {
            uint8x8_t a0 = vld1_u8((const uint8_t*)(a + i));
            uint8x8_t b0 = vld1_u8((const uint8_t*)(b + i));
            uint8x8_t reg = vceq_u8(a0, b0);
            uint8x8_t c0 = vcnt_u8(reg);
            auto dis = 8 - vaddv_u8(c0) / 8;
            accu += dis;
        }

        return accu;
    }

    inline int get_code_size() {
        return n * 8;
    }
};

} // namespace faiss

#endif

#endif
