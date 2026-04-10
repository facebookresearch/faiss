/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_NEON_INL_H
#define HAMMING_NEON_INL_H

// A specialized version of hamming is needed here, because both
// gcc, clang and msvc seem to generate suboptimal code sometimes.
//
// Universal code (HammingComputer4, HammingComputer8, generalized_hamming_64,
// byte-level hamming template, hamming_popcount_tail) comes from common.h.

#ifdef __aarch64__

#include <arm_neon.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

#include <faiss/utils/hamming_distance/common.h>

namespace faiss {

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
 * NEON-optimized HammingComputer classes for sizes 16-64.
 * Sizes 4 and 8 use the scalar versions from common.h.
 ******************************************************************/

struct HammingComputer16 {
    uint8x16_t a0;

    HammingComputer16() {}

    HammingComputer16(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
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

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
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

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
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

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
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

    HammingComputerDefault(const uint8_t* a8_in, int code_size) {
        set(a8_in, code_size);
    }

    void set(const uint8_t* a8_in, int code_size) {
        this->a8 = a8_in;
        quotient8 = code_size / 8;
        remainder8 = code_size % 8;
    }

    int hamming(const uint8_t* b8) const {
        int accu = 0;

        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);
        int i = 0;

        int len256 = (quotient8 / 4) * 4;
        for (; i < len256; i += 4) {
            accu += ::faiss::hamming<256>(a64 + i, b64 + i);
        }

        accu += hamming_popcount_tail(
                a64, b64, i, quotient8, a8, b8, remainder8);
        return accu;
    }

    inline int get_code_size() const {
        return quotient8 * 8 + remainder8;
    }
};

/***************************************************************************
 * NEON-optimized generalized Hamming computers.
 ***************************************************************************/

struct GenHammingComputer8 {
    uint8x8_t a0;

    GenHammingComputer8(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
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

    GenHammingComputer16(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
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

    GenHammingComputer32(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size)
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
