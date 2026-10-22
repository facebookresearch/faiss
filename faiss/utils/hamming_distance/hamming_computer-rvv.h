/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_RVV_H
#define HAMMING_COMPUTER_RVV_H

// RVV HammingComputer specializations. Sizes 4/8 use the scalar structs from
// hamming_computer.h (register widths too small to justify SIMD). The
// remaining sizes either inherit the scalar NONE implementation or carry an
// RVV-native body where the vector unit wins:
//
//   HammingComputer32  — native RVV body:
//                        * hamming()       one code at a time. When Zvbb is
//                                          in -march it uses vcpop.v (the
//                                          NEON vcntq shape); on Zbb harts
//                                          it uses scalar cpop (runtime
//                                          probed); otherwise it falls back
//                                          to the scalar NONE popcount body.
//                        * hamming_batch() n consecutive codes in one pass:
//                                          base RVV 1.0 has no vector
//                                          popcount, so the 64-bit SWAR
//                                          recurrence runs on u64m8 lanes
//                                          and each code's 4 word counts are
//                                          folded with two slidedown adds.
//                                          This is the hot path for the
//                                          binary top-k bucket scan in
//                                          approx_topk_hamming-rvv.h (bench
//                                          P3 bpc32): without it the scan
//                                          pays a libgcc __popcountdi2 call
//                                          per 64-bit word on builds whose
//                                          -march omits Zbb/Zvbb.
//
// The other templated sizes (16/20/64/Default) inherit the NONE scalar
// bodies via FAISS_INHERIT_HAMMING_RVV.

#include <cstdio>
#include <cstring>

#include <riscv_vector.h>

#include <faiss/utils/hamming_distance/hamming_computer-generic.h>

namespace faiss {

namespace hamming_rvv_detail {

// Does this hart implement the scalar bit-manipulation extension Zbb?
// The faiss RVV TUs are built with -march=rv64gcv_zvfhmin (no Zbb), so
// __builtin_popcountll lowers to a libgcc __popcountdi2 call. Many RVV
// application cores do implement Zbb (one cpop instruction per word), but
// the compiler cannot emit it without the -march bit, so the encoding is
// used directly (funct3=001 unary op-imm, imm=0x602) and the feature is
// probed once at runtime from the kernel's ISA string. Cores without Zbb
// keep the vector SWAR path.
inline bool zbb_available() {
#if defined(__riscv_zbb)
    return true;
#else
    static const bool has = [] {
#if defined(__linux__)
        FILE* f = std::fopen("/proc/cpuinfo", "r");
        if (f == nullptr) {
            return false;
        }
        char line[1024];
        bool ok = false;
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            if (std::strstr(line, "isa") != nullptr &&
                std::strstr(line, "zbb") != nullptr) {
                ok = true;
                break;
            }
        }
        std::fclose(f);
        return ok;
#else
        return false;
#endif
    }();
    return has;
#endif
}

// cpop rd, rs1 (Zbb). Callers must check zbb_available() first.
inline uint64_t cpop64(uint64_t x) {
#if defined(__riscv_zbb)
    return uint64_t(__builtin_popcountll(x));
#else
    uint64_t r;
    asm(".insn i 0x13, 1, %0, %1, 0x602" : "=r"(r) : "r"(x));
    return r;
#endif
}

} // namespace hamming_rvv_detail

#define FAISS_INHERIT_HAMMING_RVV(Class)                                      \
    template <>                                                               \
    struct Class##_tpl<SIMDLevel::RISCV_RVV> : Class##_tpl<SIMDLevel::NONE> { \
        using Class##_tpl<SIMDLevel::NONE>::Class##_tpl;                      \
    }

FAISS_INHERIT_HAMMING_RVV(HammingComputer16);
FAISS_INHERIT_HAMMING_RVV(HammingComputer20);
FAISS_INHERIT_HAMMING_RVV(HammingComputer64);
FAISS_INHERIT_HAMMING_RVV(HammingComputerDefault);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputer8);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputer16);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputer32);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputerM8);

// HammingComputer32 has a native RVV body below (do not also inherit the
// scalar NONE body).

#undef FAISS_INHERIT_HAMMING_RVV

// Native RVV HammingComputer32 (32 bytes = 4 u64 words).
//
// hamming() keeps the per-code shape (query words in a plain array, one
// vector load is loop-invariant and hoisted by the compiler). On Zvbb
// builds it uses the vector popcount (vcpop.v); without Zvbb the intrinsic
// does not exist, so it falls back to the same scalar popcount body as the
// NONE specialization and all SIMD value comes from hamming_batch().
//
// hamming_batch() is the bucket-scan fast path. Base RVV 1.0 has no vector
// popcount, so the classic SWAR recurrence is evaluated on u64m8 lanes:
//
//   x = x - ((x >> 1) & 0x5555...)          // 2-bit partial counts
//   x = (x & 0x3333...) + ((x >> 2) & ...) // 4-bit partial counts
//   x = (x + (x >> 4)) & 0x0f0f...         // per-byte popcounts
//   x = (x * 0x0101010101010101) >> 56     // per-64-bit-word popcount
//
// and the four words of a code are folded with two slidedown adds, so lane
// 4*c holds the Hamming distance of code c. One u64m8 group covers 4 codes
// at VLEN=128 (16 lanes), 8 codes at VLEN=256.
//
// This is mathematically identical to the scalar popcount; the bucket
// reduction downstream is unchanged, so RVV/NONE results stay bit-exact.
template <>
struct HammingComputer32_tpl<SIMDLevel::RISCV_RVV> {
    static constexpr size_t kWords = 4; // 32 bytes / 8
    // Covers VLEN <= 512 (u64m8 VLMAX = VLEN / 8 = 64 lanes). Wider
    // hardware simply processes more than one group per call.
    static constexpr size_t kMaxPatLanes = 64;

    // The compiler forbids RVV intrinsic types as struct members, so the
    // query words are kept in a plain array.
    alignas(16) uint64_t a[kWords];

    // Query pattern [a0, a1, a2, a3] repeated, filled lazily on the first
    // hamming_batch() call: per-code hamming() users (faiss core) never pay
    // for it. mutable so hamming_batch() can stay const.
    alignas(16) mutable uint64_t qpat[kMaxPatLanes];
    mutable bool qpat_valid;

    HammingComputer32_tpl() : qpat_valid(false) {}

    HammingComputer32_tpl(const uint8_t* a8, int code_size)
            : qpat_valid(false) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        for (size_t i = 0; i < kWords; i++) {
            a[i] = a64[i];
        }
        qpat_valid = false;
    }

    inline int hamming(const uint8_t* b8) const {
#if defined(__riscv_zvbb)
        // v = b ^ a; d = sum_i popcount(v_i)
        vuint64m2_t va = __riscv_vle64_v_u64m2(a, kWords);
        vuint64m2_t vb = __riscv_vle64_v_u64m2(
                reinterpret_cast<const uint64_t*>(b8), kWords);
        vuint64m2_t vx = __riscv_vxor_vv_u64m2(vb, va, kWords);
        vuint64m1_t vs = __riscv_vredsum_vs_u64m2_u64m1(
                __riscv_vcpop_v_v_u64m2(vx, kWords),
                __riscv_vmv_v_x_u64m1(0, 1),
                kWords);
        return static_cast<int>(__riscv_vmv_x_s_u64m1_u64(vs));
#else
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        if (hamming_rvv_detail::zbb_available()) {
            using hamming_rvv_detail::cpop64;
            return static_cast<int>(
                    cpop64(b[0] ^ a[0]) + cpop64(b[1] ^ a[1]) +
                    cpop64(b[2] ^ a[2]) + cpop64(b[3] ^ a[3]));
        }
        return popcount64(b[0] ^ a[0]) + popcount64(b[1] ^ a[1]) +
                popcount64(b[2] ^ a[2]) + popcount64(b[3] ^ a[3]);
#endif
    }

    // Hamming distances of n consecutive codes at b8, written to out[0..n).
    // b8 need not be aligned; codes are read as 4 little-endian u64 words
    // exactly like set()/hamming().
    inline void hamming_batch(const uint8_t* b8, uint32_t n, uint32_t* out)
            const {
        if (n == 0) {
            return;
        }

        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);

        // Zbb fast path: four cpop instructions per code beat the vector
        // SWAR emulation on 128-bit vector units, because the SWAR chain is
        // long and only produces 4 code distances per u64m8 group.
        if (hamming_rvv_detail::zbb_available()) {
            using hamming_rvv_detail::cpop64;
            for (uint32_t c = 0; c < n; c++) {
                const uint64_t* b = b64 + size_t(c) * kWords;
                out[c] = static_cast<uint32_t>(
                        cpop64(b[0] ^ a[0]) + cpop64(b[1] ^ a[1]) +
                        cpop64(b[2] ^ a[2]) + cpop64(b[3] ^ a[3]));
            }
            return;
        }

        if (!qpat_valid) {
            for (size_t i = 0; i < kMaxPatLanes; i += kWords) {
                qpat[i + 0] = a[0];
                qpat[i + 1] = a[1];
                qpat[i + 2] = a[2];
                qpat[i + 3] = a[3];
            }
            qpat_valid = true;
        }

        const size_t total = size_t(n) * kWords;
        size_t done = 0;

        while (done < total) {
            const size_t rem = total - done;
            const size_t avl = rem < kMaxPatLanes ? rem : kMaxPatLanes;
            const size_t vl = __riscv_vsetvl_e64m8(avl);

            vuint64m8_t vx = __riscv_vle64_v_u64m8(b64 + done, vl);
            const vuint64m8_t vq = __riscv_vle64_v_u64m8(qpat, vl);
            vx = __riscv_vxor_vv_u64m8(vx, vq, vl);

            // SWAR popcount, see the header comment.
            vuint64m8_t t = __riscv_vsrl_vx_u64m8(vx, 1, vl);
            vx = __riscv_vsub_vv_u64m8(
                    vx,
                    __riscv_vand_vx_u64m8(t, 0x5555555555555555ULL, vl),
                    vl);
            t = __riscv_vsrl_vx_u64m8(vx, 2, vl);
            vx = __riscv_vadd_vv_u64m8(
                    __riscv_vand_vx_u64m8(vx, 0x3333333333333333ULL, vl),
                    __riscv_vand_vx_u64m8(t, 0x3333333333333333ULL, vl),
                    vl);
            t = __riscv_vsrl_vx_u64m8(vx, 4, vl);
            vx = __riscv_vand_vx_u64m8(
                    __riscv_vadd_vv_u64m8(vx, t, vl),
                    0x0F0F0F0F0F0F0F0FULL,
                    vl);
            vx = __riscv_vmul_vx_u64m8(vx, 0x0101010101010101ULL, vl);
            vx = __riscv_vsrl_vx_u64m8(vx, 56, vl);

            // Fold the 4 words of each code: lanes 4c .. 4c+3 -> lane 4c.
            vx = __riscv_vadd_vv_u64m8(
                    vx, __riscv_vslidedown_vx_u64m8(vx, 1, vl), vl);
            vx = __riscv_vadd_vv_u64m8(
                    vx, __riscv_vslidedown_vx_u64m8(vx, 2, vl), vl);

            alignas(64) uint64_t tmp[kMaxPatLanes];
            __riscv_vse64_v_u64m8(tmp, vx, vl);

            const size_t out_base = done / kWords;
            const size_t codes = vl / kWords;
            for (size_t c = 0; c < codes; c++) {
                out[out_base + c] = static_cast<uint32_t>(tmp[c * kWords]);
            }
            done += vl;
        }
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

} // namespace faiss

#endif
