/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_GENERIC_H
#define HAMMING_COMPUTER_GENERIC_H

// Scalar (NONE) HammingComputer specializations and hamming_popcount_tail
// utility. No ISA-specific intrinsics. Per-ISA files (hamming_computer-avx2.h,
// etc.) include this file and inherit or override the NONE specializations.

#include <faiss/utils/hamming_distance/hamming_computer.h>

namespace faiss {

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
        if (remainder8 >= 4) {
            accu += popcount32(*(uint32_t*)a ^ *(uint32_t*)b);
            a += 4;
            b += 4;
            remainder8 -= 4;
        }
        if (remainder8 >= 2) {
            accu += popcount32(*(uint16_t*)a ^ *(uint16_t*)b);
            a += 2;
            b += 2;
            remainder8 -= 2;
        }
        if (remainder8 >= 1) {
            accu += popcount32(*a ^ *b);
            remainder8 -= 2;
        }
    }
    return accu;
}

/***************************************************************************
 * HammingComputer NONE specializations — scalar bodies.
 * Per-ISA backend files (hamming_computer-avx512.h, hamming_computer-neon.h,
 * etc.) provide their own specializations; those without custom code
 * inherit from NONE.
 ***************************************************************************/

template <>
struct HammingComputer16_tpl<SIMDLevel::NONE> {
    uint64_t a0, a1;

    HammingComputer16_tpl() {}

    HammingComputer16_tpl(const uint8_t* a8, int code_size) {
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
template <>
struct HammingComputer20_tpl<SIMDLevel::NONE> {
    uint64_t a0, a1;
    uint32_t a2;

    HammingComputer20_tpl() {}

    HammingComputer20_tpl(const uint8_t* a8, int code_size) {
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

template <>
struct HammingComputer32_tpl<SIMDLevel::NONE> {
    uint64_t a0, a1, a2, a3;

    HammingComputer32_tpl() {}

    HammingComputer32_tpl(const uint8_t* a8, int code_size) {
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

template <>
struct GenHammingComputer8_tpl<SIMDLevel::NONE> {
    uint64_t a0;

    GenHammingComputer8_tpl(
            const uint8_t* a,
            FAISS_MAYBE_UNUSED int code_size) {
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

/***************************************************************************
 * Scalar HammingComputer64 and HammingComputerDefault NONE specializations.
 * AVX512 and NEON override via per-ISA specializations.
 ***************************************************************************/

template <>
struct HammingComputer64_tpl<SIMDLevel::NONE> {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;

    HammingComputer64_tpl() {}

    HammingComputer64_tpl(const uint8_t* a8, int code_size) {
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

template <>
struct HammingComputerDefault_tpl<SIMDLevel::NONE> {
    const uint8_t* a8;
    int quotient8;
    int remainder8;

    HammingComputerDefault_tpl() {}

    HammingComputerDefault_tpl(const uint8_t* a8_in, int code_size) {
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

/***************************************************************************
 * Generalized HammingComputer NONE specializations (scalar bodies).
 * AVX2/AVX512/NEON override via per-ISA specializations.
 ***************************************************************************/

template <>
struct GenHammingComputer16_tpl<SIMDLevel::NONE> {
    uint64_t a0, a1;

    GenHammingComputer16_tpl(
            const uint8_t* a8,
            FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return generalized_hamming_64(b[0] ^ a0) +
                generalized_hamming_64(b[1] ^ a1);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct GenHammingComputer32_tpl<SIMDLevel::NONE> {
    uint64_t a0, a1, a2, a3;

    GenHammingComputer32_tpl(
            const uint8_t* a8,
            FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return generalized_hamming_64(b[0] ^ a0) +
                generalized_hamming_64(b[1] ^ a1) +
                generalized_hamming_64(b[2] ^ a2) +
                generalized_hamming_64(b[3] ^ a3);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct GenHammingComputerM8_tpl<SIMDLevel::NONE> {
    const uint64_t* a;
    int n;

    GenHammingComputerM8_tpl(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = reinterpret_cast<const uint64_t*>(a8);
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        int accu = 0;
        for (int i = 0; i < n; i++)
            accu += generalized_hamming_64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 8;
    }
};

} // namespace faiss

#endif
