/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_RVV_H
#define HAMMING_COMPUTER_RVV_H


#include <riscv_vector.h>

#include <faiss/utils/hamming_distance/hamming_computer-generic.h>
namespace faiss {

#define FAISS_INHERIT_HAMMING_RVV(Class)                                      \
    template <>                                                               \
    struct Class##_tpl<SIMDLevel::RISCV_RVV> : Class##_tpl<SIMDLevel::NONE> { \
        using Class##_tpl<SIMDLevel::NONE>::Class##_tpl;                      \
    }

FAISS_INHERIT_HAMMING_RVV(HammingComputer20);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputer8);

#undef FAISS_INHERIT_HAMMING_RVV

// Scalar popcount is faiss_rv_popcount64 (common.h): cpop when the TU is
// built with Zbb, inline SWAR otherwise.

template <>
struct HammingComputer32_tpl<SIMDLevel::RISCV_RVV> {
    static constexpr size_t kWords = 4; // 32 bytes / 8
    alignas(32) uint64_t a0;
    uint64_t a1, a2, a3;

    HammingComputer32_tpl() {}

    HammingComputer32_tpl(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        static_assert(sizeof(*this) == 32, "HC32 query block must be 32B");
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        a0 = a64[0];
        a1 = a64[1];
        a2 = a64[2];
        a3 = a64[3];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return (int)faiss_rv_popcount64(b[0] ^ a0) +
                (int)faiss_rv_popcount64(b[1] ^ a1) +
                (int)faiss_rv_popcount64(b[2] ^ a2) +
                (int)faiss_rv_popcount64(b[3] ^ a3);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct HammingComputer64_tpl<SIMDLevel::RISCV_RVV> {
    static constexpr size_t kWords = 8; // 64 bytes / 8
    alignas(16) uint64_t a[kWords];

    HammingComputer64_tpl() {}

    HammingComputer64_tpl(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 64);
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        for (size_t i = 0; i < kWords; i++) {
            a[i] = a64[i];
        }
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return (int)faiss_rv_popcount64(b[0] ^ a[0]) +
                (int)faiss_rv_popcount64(b[1] ^ a[1]) +
                (int)faiss_rv_popcount64(b[2] ^ a[2]) +
                (int)faiss_rv_popcount64(b[3] ^ a[3]) +
                (int)faiss_rv_popcount64(b[4] ^ a[4]) +
                (int)faiss_rv_popcount64(b[5] ^ a[5]) +
                (int)faiss_rv_popcount64(b[6] ^ a[6]) +
                (int)faiss_rv_popcount64(b[7] ^ a[7]);
    }

    inline static constexpr int get_code_size() {
        return 64;
    }
};

template <>
struct HammingComputer16_tpl<SIMDLevel::RISCV_RVV> {
    static constexpr size_t kWords = 2; // 16 bytes / 8
    alignas(16) uint64_t a0;
    uint64_t a1;

    HammingComputer16_tpl() {}

    HammingComputer16_tpl(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        static_assert(sizeof(*this) == 16, "HC16 query block must be 16B");
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        a0 = a64[0];
        a1 = a64[1];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return (int)faiss_rv_popcount64(b[0] ^ a0) +
                (int)faiss_rv_popcount64(b[1] ^ a1);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct HammingComputerDefault_tpl<SIMDLevel::RISCV_RVV> {
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
        int accu = 0;
        int i = 0;
        // 8-way unrolled word main loop (no Duff switch). faiss_rv_popcount64
        // is cpop when the TU is built with Zbb and inline SWAR otherwise.
        for (; i + 8 <= quotient8; i += 8) {
            accu += (int)faiss_rv_popcount64(a64[i + 0] ^ b64[i + 0]);
            accu += (int)faiss_rv_popcount64(a64[i + 1] ^ b64[i + 1]);
            accu += (int)faiss_rv_popcount64(a64[i + 2] ^ b64[i + 2]);
            accu += (int)faiss_rv_popcount64(a64[i + 3] ^ b64[i + 3]);
            accu += (int)faiss_rv_popcount64(a64[i + 4] ^ b64[i + 4]);
            accu += (int)faiss_rv_popcount64(a64[i + 5] ^ b64[i + 5]);
            accu += (int)faiss_rv_popcount64(a64[i + 6] ^ b64[i + 6]);
            accu += (int)faiss_rv_popcount64(a64[i + 7] ^ b64[i + 7]);
        }
        for (; i < quotient8; i++) {
            accu += (int)faiss_rv_popcount64(a64[i] ^ b64[i]);
        }
        int rem = remainder8;
        if (rem) {
            const uint8_t* a = a8 + 8 * quotient8;
            const uint8_t* b = b8 + 8 * quotient8;
            if (rem >= 4) {
                accu += popcount32(*(const uint32_t*)a ^ *(const uint32_t*)b);
                a += 4;
                b += 4;
                rem -= 4;
            }
            if (rem >= 2) {
                accu += popcount32(*(const uint16_t*)a ^ *(const uint16_t*)b);
                a += 2;
                b += 2;
                rem -= 2;
            }
            if (rem >= 1) {
                accu += popcount32(*a ^ *b);
            }
        }
        return accu;
    }

    inline int get_code_size() const {
        return quotient8 * 8 + remainder8;
    }
};

template <>
struct GenHammingComputer16_tpl<SIMDLevel::RISCV_RVV> {
    static constexpr size_t kBytes = 16;

    alignas(16) uint8_t a[kBytes];

    GenHammingComputer16_tpl() {}

    GenHammingComputer16_tpl(
            const uint8_t* a8,
            FAISS_MAYBE_UNUSED int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        uint64_t* dst = reinterpret_cast<uint64_t*>(a);
        for (size_t i = 0; i < kBytes / 8; i++) {
            dst[i] = a64[i];
        }
    }

    inline __attribute__((always_inline)) int hamming(
            const uint8_t* __restrict b8) const {
        const uint8_t* ap =
                static_cast<const uint8_t*>(__builtin_assume_aligned(a, 16));
        const uint8_t* bp = static_cast<const uint8_t*>(
                __builtin_assume_aligned(b8, 16));
        vuint8m1_t va = __riscv_vle8_v_u8m1(ap, kBytes);
        vuint8m1_t vb = __riscv_vle8_v_u8m1(bp, kBytes);
        vbool8_t ne = __riscv_vmsne(va, vb, kBytes);
        return static_cast<int>(__riscv_vcpop_m_b8(ne, kBytes));
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct GenHammingComputer32_tpl<SIMDLevel::RISCV_RVV> {
    static constexpr size_t kBytes = 32;
    alignas(32) uint8_t a[kBytes];

    GenHammingComputer32_tpl() {}

    GenHammingComputer32_tpl(
            const uint8_t* a8,
            FAISS_MAYBE_UNUSED int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        static_assert(sizeof(*this) == 32, "GenHC32 query block must be 32B");
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        uint64_t* dst = reinterpret_cast<uint64_t*>(a);
        for (size_t i = 0; i < kBytes / 8; i++) {
            dst[i] = a64[i];
        }
    }


    inline int hamming(const uint8_t* __restrict b8) const {
        const uint8_t* ap =
                static_cast<const uint8_t*>(__builtin_assume_aligned(a, 32));
        const uint8_t* bp = static_cast<const uint8_t*>(
                __builtin_assume_aligned(b8, 16));

        if (__riscv_vlenb() >= 32) {
            vuint8m1_t va = __riscv_vle8_v_u8m1(ap, kBytes);
            vuint8m1_t vb = __riscv_vle8_v_u8m1(bp, kBytes);
            vbool8_t ne = __riscv_vmsne(va, vb, kBytes);
            return static_cast<int>(__riscv_vcpop_m_b8(ne, kBytes));
        }
        vuint8m2_t va = __riscv_vle8_v_u8m2(ap, kBytes);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(bp, kBytes);
        vbool4_t ne = __riscv_vmsne(va, vb, kBytes);
        return static_cast<int>(__riscv_vcpop_m_b4(ne, kBytes));
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};


template <>
struct GenHammingComputerM8_tpl<SIMDLevel::RISCV_RVV> {
    static constexpr size_t kBlock = 128;
    const uint8_t* a8;
    int n;
    GenHammingComputerM8_tpl(const uint8_t* a8_in, int code_size) {
        assert(code_size % 8 == 0);
        this->a8 = a8_in;
        n = code_size;
    }
    int hamming(const uint8_t* b8) const {
        int accu = 0;
        size_t i = 0;
        for (; i + kBlock <= size_t(n); i += kBlock) {
            vuint8m8_t va = __riscv_vle8_v_u8m8(a8 + i, kBlock);
            vuint8m8_t vb = __riscv_vle8_v_u8m8(b8 + i, kBlock);
            vbool1_t ne = __riscv_vmsne(va, vb, kBlock);
            accu += static_cast<int>(__riscv_vcpop_m_b1(ne, kBlock));
        }
        size_t rem = size_t(n) - i;
        if (rem > 0) {
            vuint8m8_t va = __riscv_vle8_v_u8m8(a8 + i, rem);
            vuint8m8_t vb = __riscv_vle8_v_u8m8(b8 + i, rem);
            vbool1_t ne = __riscv_vmsne(va, vb, rem);
            accu += static_cast<int>(__riscv_vcpop_m_b1(ne, rem));
        }
        return accu;
    }

    inline int get_code_size() const {
        return n;
    }
};

} // namespace faiss

#endif
