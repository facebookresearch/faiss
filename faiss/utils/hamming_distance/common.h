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
#include <faiss/utils/popcount.h>
#include <faiss/utils/simd_levels.h>

#ifdef __aarch64__
// Include <arm_neon.h> at global scope so the NEON types it declares
// (uint8x16_t, uint16x8_t, ...) end up in `::` and not inside `faiss::`.
#include <arm_neon.h>
#endif

// RVV free-function hamming<128/256/512> kernels (Base RVV 1.0, no Zvbb).
// Guarded by the faiss-wide COMPILE_SIMD_RISCV_RVV macro so NONE builds
// (and SWIG) keep the scalar semantics below untouched.
#if !defined(SWIG) && defined(__riscv) && defined(__riscv_vector) && \
        defined(COMPILE_SIMD_RISCV_RVV)
#define FAISS_HAMMING_FREEFN_RVV 1
#include <riscv_vector.h>
#endif

/* The Hamming distance type */
using hamdis_t = int32_t;

namespace faiss {

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

/***************************************************************************
 * Free-function hamming<nbits>() templates.
 * hamming<64> is architecture-independent.
 * hamming<128>, hamming<256>, the primary template, and the runtime-nwords
 * overload have NEON-optimized versions on aarch64.
 ***************************************************************************/

#ifndef SWIG

/* hamming<64> — identical on all architectures */
template <size_t nbits>
inline hamdis_t hamming(const uint64_t* bs1, const uint64_t* bs2);

template <>
inline hamdis_t hamming<64>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]);
}

#ifdef __aarch64__

/* Hamming distances for multiples of 64 bits — NEON version */
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

#else // !__aarch64__

/* Hamming distances for multiples of 64 bits — scalar version */
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

#if defined(__riscv)

// Scalar 64-bit popcount for the RVV hamming kernels. Zbb is not part of the
// RISC-V SIMD baseline, and a `.option arch, +zbb` scope-escape would only
// teach the assembler the cpop mnemonic — on hardware without Zbb the
// instruction traps at runtime. So branch at compile time instead: cpop via
// the builtin when the TU's -march enables Zbb (the FAISS_SIMD_RVV_SRC TUs
// are built with -march=rv64gcv_zvfhmin_zbb), an inline SWAR popcount
// otherwise. Some TUs that instantiate the scalar kernels (e.g. the NONE
// fallback TU in hamming.cpp) are not covered by that per-file march, hence
// the fallback. SWAR is ~10 instructions per word vs 1 for cpop, so Zbb
// where the hardware has it is the fast path.
inline uint64_t faiss_rv_popcount64(uint64_t x) {
#if defined(__riscv_zbb)
    return static_cast<uint64_t>(__builtin_popcountll(x));
#else
    x -= (x >> 1) & 0x5555555555555555ULL;
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (x * 0x0101010101010101ULL) >> 56;
#endif
}

#endif // __riscv

#ifdef FAISS_HAMMING_FREEFN_RVV

// RVV specializations for the fixed 128/256/512-bit code sizes (the sizes
// the hammings_impl<nbits> dispatch in hamming_impl.h maps cs16/cs32/cs64
// onto). Byte-domain SWAR popcount (Base RVV 1.0 — no Zvbb needed), the
// exact 3-step byte-SWAR chain (verified for all 256 byte values):
//   t1 = x - ((x >> 1) & 0x55)          — pairs of bits (2-bit fields)
//   t2 = (t1 & 0x33) + ((t1 >> 2) & 0x33) — nibble counts (<= 4)
//   t3 = (t2 + (t2 >> 4)) & 0x0f        — byte popcount (<= 8, no u8
//                                          overflow: t2 <= 68)
// Then one widening u16 vwredsumu per group (each lane <= 8 — safe).
// The whole 16-byte code fits one e8m1 group (vl=16) at any VLEN >= 128.
// Fixed template sizes only (nbits is a compile-time constant of the
// specialization) — no benchmark-size hardcoding involved.
//
// always_inline: the 4x-unrolled hammings_impl loop must see these bodies
// inline. GCC decides that per TU, and its budget shifts with unrelated
// code in the TU — with the ragged helpers in hamming_impl.h, hamming<256>
// was left out of line and the hammings_matrix/cs32 row measured 2.7x
// slower on SG2044. Pin the three kernels explicitly.
template <>
__attribute__((always_inline)) inline hamdis_t hamming<128>(
        const uint64_t* pa,
        const uint64_t* pb) {
    const uint8_t* pa8 = reinterpret_cast<const uint8_t*>(pa);
    const uint8_t* pb8 = reinterpret_cast<const uint8_t*>(pb);
    constexpr size_t kBytes = 16; // one e8m1 group
    vuint8m1_t va = __riscv_vle8_v_u8m1(pa8, kBytes);
    vuint8m1_t vb = __riscv_vle8_v_u8m1(pb8, kBytes);
    vuint8m1_t x = __riscv_vxor_vv_u8m1(va, vb, kBytes);
    vuint8m1_t xr1 = __riscv_vsrl_vx_u8m1(x, 1, kBytes);
    vuint8m1_t t1 = __riscv_vsub_vv_u8m1(
            x, __riscv_vand_vx_u8m1(xr1, 0x55, kBytes), kBytes);
    vuint8m1_t t1r2 = __riscv_vsrl_vx_u8m1(t1, 2, kBytes);
    vuint8m1_t t2 = __riscv_vadd_vv_u8m1(
            __riscv_vand_vx_u8m1(t1, 0x33, kBytes),
            __riscv_vand_vx_u8m1(t1r2, 0x33, kBytes),
            kBytes);
    vuint8m1_t t2r4 = __riscv_vsrl_vx_u8m1(t2, 4, kBytes);
    vuint8m1_t t3 = __riscv_vand_vx_u8m1(
            __riscv_vadd_vv_u8m1(t2, t2r4, kBytes), 0x0f, kBytes);
    // 16 lanes * 8 bits = 128 <= 255, so a NON-widening u8 vredsum is
    // exact and avoids the u8->u16 widening reduction path.
    vuint8m1_t s = __riscv_vredsum_vs_u8m1_u8m1(
            t3, __riscv_vmv_v_x_u8m1(0, 1), kBytes);
    return static_cast<hamdis_t>(__riscv_vmv_x_s_u8m1_u8(s));
}

// hamming<256>: two e8m1 half-groups (measured best data layout — fusing
// both halves into one e8m2/e8m4 group was slower on SG2044, see
// experiment hamming_freefn_rvv rounds 1/4/5). The two reductions are
// folded into a single extract:
//   s = vwredsumu(t3_half0, zero)
//   s = vwredsumu(t3_half1, s)      // vs1 carries the first half's sum
// The SWAR chains stay independent; only the reduction trees' final adds
// are chained. Removes one vmv.x.s, one zero-extend pair and one scalar
// add per call versus two independent extracts (-14.5% RVV/NONE ratio).
template <>
__attribute__((always_inline)) inline hamdis_t hamming<256>(
        const uint64_t* pa,
        const uint64_t* pb) {
    const uint8_t* pa8 = reinterpret_cast<const uint8_t*>(pa);
    const uint8_t* pb8 = reinterpret_cast<const uint8_t*>(pb);
    constexpr size_t kHalf = 16; // one e8m1 group
    vuint8m1_t va0 = __riscv_vle8_v_u8m1(pa8, kHalf);
    vuint8m1_t vb0 = __riscv_vle8_v_u8m1(pb8, kHalf);
    vuint8m1_t x0 = __riscv_vxor_vv_u8m1(va0, vb0, kHalf);
    vuint8m1_t x0r1 = __riscv_vsrl_vx_u8m1(x0, 1, kHalf);
    vuint8m1_t t1_0 = __riscv_vsub_vv_u8m1(
            x0, __riscv_vand_vx_u8m1(x0r1, 0x55, kHalf), kHalf);
    vuint8m1_t t1_0r2 = __riscv_vsrl_vx_u8m1(t1_0, 2, kHalf);
    vuint8m1_t t2_0 = __riscv_vadd_vv_u8m1(
            __riscv_vand_vx_u8m1(t1_0, 0x33, kHalf),
            __riscv_vand_vx_u8m1(t1_0r2, 0x33, kHalf),
            kHalf);
    vuint8m1_t t2_0r4 = __riscv_vsrl_vx_u8m1(t2_0, 4, kHalf);
    vuint8m1_t t3_0 = __riscv_vand_vx_u8m1(
            __riscv_vadd_vv_u8m1(t2_0, t2_0r4, kHalf), 0x0f, kHalf);
    vuint16m1_t s = __riscv_vwredsumu_vs_u8m1_u16m1(
            t3_0, __riscv_vmv_v_x_u16m1(0, 1), kHalf);

    vuint8m1_t va1 = __riscv_vle8_v_u8m1(pa8 + kHalf, kHalf);
    vuint8m1_t vb1 = __riscv_vle8_v_u8m1(pb8 + kHalf, kHalf);
    vuint8m1_t x1 = __riscv_vxor_vv_u8m1(va1, vb1, kHalf);
    vuint8m1_t x1r1 = __riscv_vsrl_vx_u8m1(x1, 1, kHalf);
    vuint8m1_t t1_1 = __riscv_vsub_vv_u8m1(
            x1, __riscv_vand_vx_u8m1(x1r1, 0x55, kHalf), kHalf);
    vuint8m1_t t1_1r2 = __riscv_vsrl_vx_u8m1(t1_1, 2, kHalf);
    vuint8m1_t t2_1 = __riscv_vadd_vv_u8m1(
            __riscv_vand_vx_u8m1(t1_1, 0x33, kHalf),
            __riscv_vand_vx_u8m1(t1_1r2, 0x33, kHalf),
            kHalf);
    vuint8m1_t t2_1r4 = __riscv_vsrl_vx_u8m1(t2_1, 4, kHalf);
    vuint8m1_t t3_1 = __riscv_vand_vx_u8m1(
            __riscv_vadd_vv_u8m1(t2_1, t2_1r4, kHalf), 0x0f, kHalf);
    s = __riscv_vwredsumu_vs_u8m1_u16m1(t3_1, s, kHalf);
    return static_cast<hamdis_t>(__riscv_vmv_x_s_u16m1_u16(s));
}

// hamming<512>: 64 bytes are exactly one e8m4 group at VLEN=128
// (VLMAX = 4 * 128 / 8 = 64). Without this specialization the generic
// scalar template is used for cs64 and measured ~3x slower (2159 us vs
// 700 us once the kernel existed). The 3-step byte-SWAR chain plus a
// single widening reduce; 64 lanes * 8 bits = 512 fits the u16
// accumulator. The 2x hamming<256> reuse form measured slower (715 us),
// so the dedicated e8m4 group is kept.
template <>
__attribute__((always_inline)) inline hamdis_t hamming<512>(
        const uint64_t* pa,
        const uint64_t* pb) {
    const uint8_t* pa8 = reinterpret_cast<const uint8_t*>(pa);
    const uint8_t* pb8 = reinterpret_cast<const uint8_t*>(pb);
    constexpr size_t kBytes = 64; // one e8m4 group
    vuint8m4_t va = __riscv_vle8_v_u8m4(pa8, kBytes);
    vuint8m4_t vb = __riscv_vle8_v_u8m4(pb8, kBytes);
    vuint8m4_t x = __riscv_vxor_vv_u8m4(va, vb, kBytes);
    vuint8m4_t xr1 = __riscv_vsrl_vx_u8m4(x, 1, kBytes);
    vuint8m4_t t1 = __riscv_vsub_vv_u8m4(
            x, __riscv_vand_vx_u8m4(xr1, 0x55, kBytes), kBytes);
    vuint8m4_t t1r2 = __riscv_vsrl_vx_u8m4(t1, 2, kBytes);
    vuint8m4_t t2 = __riscv_vadd_vv_u8m4(
            __riscv_vand_vx_u8m4(t1, 0x33, kBytes),
            __riscv_vand_vx_u8m4(t1r2, 0x33, kBytes),
            kBytes);
    vuint8m4_t t2r4 = __riscv_vsrl_vx_u8m4(t2, 4, kBytes);
    vuint8m4_t t3 = __riscv_vand_vx_u8m4(
            __riscv_vadd_vv_u8m4(t2, t2r4, kBytes), 0x0f, kBytes);
    vuint16m1_t s = __riscv_vwredsumu_vs_u8m4_u16m1(
            t3, __riscv_vmv_v_x_u16m1(0, 1), kBytes);
    return static_cast<hamdis_t>(__riscv_vmv_x_s_u16m1_u16(s));
}

#else

/* specialized (optimized) functions */
template <>
inline hamdis_t hamming<128>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]);
}

template <>
inline hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]) +
            popcount64(pa[2] ^ pb[2]) + popcount64(pa[3] ^ pb[3]);
}

#endif // FAISS_HAMMING_FREEFN_RVV

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

#endif // __aarch64__

/***************************************************************************
 * Bit-level Hamming distance implementation functions.
 * These depend only on the hamming<nbits>() free functions above.
 ***************************************************************************/

template <size_t nbits>
inline void hammings_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        hamdis_t* __restrict dis) {
    size_t i, j;
    constexpr size_t nwords = nbits / 64;
    for (i = 0; i < n1; i++) {
        const uint64_t* __restrict bs1_ = bs1 + i * nwords;
        hamdis_t* __restrict dis_ = dis + i * n2;
#ifdef FAISS_HAMMING_FREEFN_RVV
        // 4x unrolled inner loop: four adjacent codes interleaved on the
        // same query so later codes' loads/xors overlap earlier codes'
        // reduction-tree latency. Generic for any nbits (the compiler
        // instantiates the fixed-size hamming<nbits> body four times);
        // the tail handles n2 % 4.
        j = 0;
        for (; j + 4 <= n2; j += 4) {
            hamdis_t h0 = hamming<nbits>(bs1_, bs2 + j * nwords);
            hamdis_t h1 = hamming<nbits>(bs1_, bs2 + (j + 1) * nwords);
            hamdis_t h2 = hamming<nbits>(bs1_, bs2 + (j + 2) * nwords);
            hamdis_t h3 = hamming<nbits>(bs1_, bs2 + (j + 3) * nwords);
            dis_[j] = h0;
            dis_[j + 1] = h1;
            dis_[j + 2] = h2;
            dis_[j + 3] = h3;
        }
        for (; j < n2; j++) {
            dis_[j] = hamming<nbits>(bs1_, bs2 + j * nwords);
        }
#else
        if constexpr (nwords >= 4) {
            // Wide codes are quicker one candidate at a time; otherwise the
            // compiler batches candidates and must gather scattered bytes.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
            for (j = 0; j < n2; j++) {
                dis_[j] = hamming<nbits>(bs1_, bs2 + j * nwords);
            }
        } else {
            for (j = 0; j < n2; j++) {
                dis_[j] = hamming<nbits>(bs1_, bs2 + j * nwords);
            }
        }
#endif
    }
}

inline void hammings_impl_runtime(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        size_t nbits,
        hamdis_t* __restrict dis) {
    size_t i, j;
    const size_t nwords = nbits / 64;
    for (i = 0; i < n1; i++) {
        const uint64_t* __restrict bs1_ = bs1 + i * nwords;
        hamdis_t* __restrict dis_ = dis + i * n2;
        for (j = 0; j < n2; j++) {
            dis_[j] = hamming(bs1_, bs2 + j * nwords, nwords);
        }
    }
}

template <size_t nbits>
inline void hamming_count_thres_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        hamdis_t ht,
        size_t* __restrict nptr) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    const uint64_t* bs2_ = bs2;

    for (i = 0; i < n1; i++) {
        bs2 = bs2_;
        for (j = 0; j < n2; j++) {
            if (hamming<nbits>(bs1, bs2) <= ht) {
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    *nptr = posm;
}

template <size_t nbits>
inline void crosshamming_count_thres_impl(
        const uint64_t* __restrict dbs,
        size_t n,
        int ht,
        size_t* __restrict nptr) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    const uint64_t* bs1 = dbs;
    for (i = 0; i < n; i++) {
        const uint64_t* bs2 = bs1 + nwords;
        for (j = i + 1; j < n; j++) {
            if (hamming<nbits>(bs1, bs2) <= ht) {
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    *nptr = posm;
}

template <size_t nbits>
inline size_t match_hamming_thres_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        int ht,
        int64_t* __restrict idx,
        hamdis_t* __restrict hams) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    hamdis_t h;
    const uint64_t* bs2_ = bs2;
    for (i = 0; i < n1; i++) {
        bs2 = bs2_;
        for (j = 0; j < n2; j++) {
            h = hamming<nbits>(bs1, bs2);
            if (h <= ht) {
                *idx = i;
                idx++;
                *idx = j;
                idx++;
                *hams = h;
                hams++;
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    return posm;
}

#endif // SWIG

} // namespace faiss

#endif
