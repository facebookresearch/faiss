/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/* QT_8bit distance kernel for RISC-V Vector.
 *
 * Free-standing (no exceptions, heap or threads), so a freestanding RISC-V
 * target can compile it. sq-rvv.cpp calls `l2_distance`; the block, strided
 * and 4-bit forms serve callers that store codes transposed, such as the
 * MTIA kernels in faiss/fb/mtia. No Buck target here builds the RISC-V CPU
 * path. See sq8_coefficients.h for the meaning of a and e.
 */

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace sq8_rvv {

/// Candidates per transposed block. One block fills an f32 LMUL 4 group at
/// VLEN=512, so every lane holds one candidate. On a smaller VLEN the block
/// functions loop over the block in pieces of one vector length.
constexpr size_t kBlockVectors = 64;

/// Squared L2 for a whole block, from codes stored transposed: the byte for
/// dimension i of lane c sits at `block[i * kBlockVectors + c]`. Each lane
/// accumulates one candidate, so no reduction is needed.
inline void l2_distance_block(
        const uint8_t* block,
        const float* a,
        const float* e,
        size_t d,
        float* out) {
    for (size_t c = 0; c < kBlockVectors;) {
        const size_t vl = __riscv_vsetvl_e32m4(kBlockVectors - c);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t i = 0; i < d; i++) {
            vuint8m1_t c8 =
                    __riscv_vle8_v_u8m1(block + c + i * kBlockVectors, vl);
            vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(c8, vl), vl);
            // a[i] and e[i] are the same for every candidate in the block.
            vfloat32m4_t t = __riscv_vfmv_v_f_f32m4(e[i], vl);
            t = __riscv_vfnmsac_vf_f32m4(t, a[i], cf, vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
        }
        __riscv_vse32_v_f32m4(out + c, acc, vl);
        c += vl;
    }
}

/// Squared L2 for two transposed blocks at once.
///
/// Issuing the loads of both blocks first keeps two loads in flight. Four
/// blocks would need 16 of the 32 vector registers for accumulators and
/// spill.
inline void l2_distance_block_x2(
        const uint8_t* block0,
        const uint8_t* block1,
        const float* a,
        const float* e,
        size_t d,
        float* out0,
        float* out1) {
    for (size_t c = 0; c < kBlockVectors;) {
        const size_t vl = __riscv_vsetvl_e32m4(kBlockVectors - c);
        vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t i = 0; i < d; i++) {
            vuint8m1_t v0 =
                    __riscv_vle8_v_u8m1(block0 + c + i * kBlockVectors, vl);
            vuint8m1_t v1 =
                    __riscv_vle8_v_u8m1(block1 + c + i * kBlockVectors, vl);
            const float ei = e[i];
            const float ai = a[i];
            vfloat32m4_t c0 = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(v0, vl), vl);
            vfloat32m4_t t0 = __riscv_vfmv_v_f_f32m4(ei, vl);
            t0 = __riscv_vfnmsac_vf_f32m4(t0, ai, c0, vl);
            acc0 = __riscv_vfmacc_vv_f32m4(acc0, t0, t0, vl);
            vfloat32m4_t c1 = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(v1, vl), vl);
            vfloat32m4_t t1 = __riscv_vfmv_v_f_f32m4(ei, vl);
            t1 = __riscv_vfnmsac_vf_f32m4(t1, ai, c1, vl);
            acc1 = __riscv_vfmacc_vv_f32m4(acc1, t1, t1, vl);
        }
        __riscv_vse32_v_f32m4(out0 + c, acc0, vl);
        __riscv_vse32_v_f32m4(out1 + c, acc1, vl);
        c += vl;
    }
}

/// The code-dependent inner product term for two transposed blocks at once.
inline void ip_partial_block_x2(
        const uint8_t* block0,
        const uint8_t* block1,
        const float* eq,
        size_t d,
        float* out0,
        float* out1) {
    for (size_t c = 0; c < kBlockVectors;) {
        const size_t vl = __riscv_vsetvl_e32m4(kBlockVectors - c);
        vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t i = 0; i < d; i++) {
            vuint8m1_t v0 =
                    __riscv_vle8_v_u8m1(block0 + c + i * kBlockVectors, vl);
            vuint8m1_t v1 =
                    __riscv_vle8_v_u8m1(block1 + c + i * kBlockVectors, vl);
            const float q = eq[i];
            acc0 = __riscv_vfmacc_vf_f32m4(
                    acc0,
                    q,
                    __riscv_vfcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf4_u32m4(v0, vl), vl),
                    vl);
            acc1 = __riscv_vfmacc_vf_f32m4(
                    acc1,
                    q,
                    __riscv_vfcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf4_u32m4(v1, vl), vl),
                    vl);
        }
        __riscv_vse32_v_f32m4(out0 + c, acc0, vl);
        __riscv_vse32_v_f32m4(out1 + c, acc1, vl);
        c += vl;
    }
}

/// The low nibbles of a row of packed QT_4bit codes, as floats.
inline vfloat32m4_t low_nibbles(vuint8m1_t packed, size_t vl) {
    return __riscv_vfcvt_f_xu_v_f32m4(
            __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(packed, 0x0F, vl), vl),
            vl);
}

/// Squared L2 for a transposed block of QT_4bit codes. Faiss packs
/// dimension i into `code[i / 2]`, even dimensions in the low nibble and odd
/// in the high one, so d dimensions occupy (d + 1) / 2 bytes. For an odd d
/// the last byte holds one dimension, in its low nibble.
inline void l2_distance_4bit_block(
        const uint8_t* block,
        const float* a,
        const float* e,
        size_t d,
        float* out) {
    for (size_t c = 0; c < kBlockVectors;) {
        const size_t vl = __riscv_vsetvl_e32m4(kBlockVectors - c);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t i = 0; i < d / 2; i++) {
            vuint8m1_t packed =
                    __riscv_vle8_v_u8m1(block + c + i * kBlockVectors, vl);
            vfloat32m4_t lo = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(
                            __riscv_vand_vx_u8m1(packed, 0x0F, vl), vl),
                    vl);
            vfloat32m4_t hi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(
                            __riscv_vsrl_vx_u8m1(packed, 4, vl), vl),
                    vl);
            vfloat32m4_t t = __riscv_vfmv_v_f_f32m4(e[2 * i], vl);
            t = __riscv_vfnmsac_vf_f32m4(t, a[2 * i], lo, vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
            t = __riscv_vfmv_v_f_f32m4(e[2 * i + 1], vl);
            t = __riscv_vfnmsac_vf_f32m4(t, a[2 * i + 1], hi, vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
        }
        if (d % 2 != 0) {
            vfloat32m4_t lo = low_nibbles(
                    __riscv_vle8_v_u8m1(
                            block + c + (d / 2) * kBlockVectors, vl),
                    vl);
            vfloat32m4_t t = __riscv_vfmv_v_f_f32m4(e[d - 1], vl);
            t = __riscv_vfnmsac_vf_f32m4(t, a[d - 1], lo, vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
        }
        __riscv_vse32_v_f32m4(out + c, acc, vl);
        c += vl;
    }
}

/// The code-dependent inner product term for a transposed QT_4bit block.
inline void ip_partial_4bit_block(
        const uint8_t* block,
        const float* eq,
        size_t d,
        float* out) {
    for (size_t c = 0; c < kBlockVectors;) {
        const size_t vl = __riscv_vsetvl_e32m4(kBlockVectors - c);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t i = 0; i < d / 2; i++) {
            vuint8m1_t packed =
                    __riscv_vle8_v_u8m1(block + c + i * kBlockVectors, vl);
            vfloat32m4_t lo = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(
                            __riscv_vand_vx_u8m1(packed, 0x0F, vl), vl),
                    vl);
            vfloat32m4_t hi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(
                            __riscv_vsrl_vx_u8m1(packed, 4, vl), vl),
                    vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, eq[2 * i], lo, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, eq[2 * i + 1], hi, vl);
        }
        if (d % 2 != 0) {
            vfloat32m4_t lo = low_nibbles(
                    __riscv_vle8_v_u8m1(
                            block + c + (d / 2) * kBlockVectors, vl),
                    vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, eq[d - 1], lo, vl);
        }
        __riscv_vse32_v_f32m4(out + c, acc, vl);
        c += vl;
    }
}

/// Squared L2 for one QT_4bit candidate in the transposed layout, for the
/// runs an `IDSelector` produces, which start anywhere.
inline float l2_distance_4bit_strided(
        const uint8_t* p,
        const float* a,
        const float* e,
        size_t d) {
    float acc = 0.0f;
    for (size_t i = 0; i < d / 2; i++) {
        const uint8_t packed = p[i * kBlockVectors];
        const float t0 =
                e[2 * i] - a[2 * i] * static_cast<float>(packed & 0x0F);
        const float t1 =
                e[2 * i + 1] - a[2 * i + 1] * static_cast<float>(packed >> 4);
        acc += t0 * t0 + t1 * t1;
    }
    if (d % 2 != 0) {
        const uint8_t packed = p[(d / 2) * kBlockVectors];
        const float t = e[d - 1] - a[d - 1] * static_cast<float>(packed & 0x0F);
        acc += t * t;
    }
    return acc;
}

/// The code-dependent inner product term for one strided QT_4bit candidate.
inline float ip_partial_4bit_strided(
        const uint8_t* p,
        const float* eq,
        size_t d) {
    float acc = 0.0f;
    for (size_t i = 0; i < d / 2; i++) {
        const uint8_t packed = p[i * kBlockVectors];
        acc += eq[2 * i] * static_cast<float>(packed & 0x0F);
        acc += eq[2 * i + 1] * static_cast<float>(packed >> 4);
    }
    if (d % 2 != 0) {
        const uint8_t packed = p[(d / 2) * kBlockVectors];
        acc += eq[d - 1] * static_cast<float>(packed & 0x0F);
    }
    return acc;
}

/// Squared L2 for two transposed QT_4bit blocks at once: half the bytes of
/// QT_8bit and two loads in flight.
inline void l2_distance_4bit_block_x2(
        const uint8_t* block0,
        const uint8_t* block1,
        const float* a,
        const float* e,
        size_t d,
        float* out0,
        float* out1) {
    for (size_t c = 0; c < kBlockVectors;) {
        const size_t vl = __riscv_vsetvl_e32m4(kBlockVectors - c);
        vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t i = 0; i < d / 2; i++) {
            vuint8m1_t p0 =
                    __riscv_vle8_v_u8m1(block0 + c + i * kBlockVectors, vl);
            vuint8m1_t p1 =
                    __riscv_vle8_v_u8m1(block1 + c + i * kBlockVectors, vl);
            const float e0 = e[2 * i];
            const float e1 = e[2 * i + 1];
            const float a0 = a[2 * i];
            const float a1 = a[2 * i + 1];
            auto step = [&](vuint8m1_t p, vfloat32m4_t acc) {
                vfloat32m4_t lo = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(
                                __riscv_vand_vx_u8m1(p, 0x0F, vl), vl),
                        vl);
                vfloat32m4_t hi = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(
                                __riscv_vsrl_vx_u8m1(p, 4, vl), vl),
                        vl);
                vfloat32m4_t t = __riscv_vfmv_v_f_f32m4(e0, vl);
                t = __riscv_vfnmsac_vf_f32m4(t, a0, lo, vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
                t = __riscv_vfmv_v_f_f32m4(e1, vl);
                t = __riscv_vfnmsac_vf_f32m4(t, a1, hi, vl);
                return __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
            };
            acc0 = step(p0, acc0);
            acc1 = step(p1, acc1);
        }
        if (d % 2 != 0) {
            const size_t last = (d / 2) * kBlockVectors;
            auto tail = [&](vuint8m1_t p, vfloat32m4_t acc) {
                vfloat32m4_t t = __riscv_vfmv_v_f_f32m4(e[d - 1], vl);
                t = __riscv_vfnmsac_vf_f32m4(
                        t, a[d - 1], low_nibbles(p, vl), vl);
                return __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
            };
            acc0 = tail(__riscv_vle8_v_u8m1(block0 + c + last, vl), acc0);
            acc1 = tail(__riscv_vle8_v_u8m1(block1 + c + last, vl), acc1);
        }
        __riscv_vse32_v_f32m4(out0 + c, acc0, vl);
        __riscv_vse32_v_f32m4(out1 + c, acc1, vl);
        c += vl;
    }
}

/// Squared L2 for one candidate whose code is stored transposed: its d bytes
/// sit `kBlockVectors` apart. For runs that do not start on a block
/// boundary, such as the runs an `IDSelector` produces.
inline float l2_distance_strided(
        const uint8_t* p,
        const float* a,
        const float* e,
        size_t d) {
    const size_t vl = __riscv_vsetvl_e8m1(d > 0 ? d : 1);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    size_t i = 0;
    while (i < d) {
        const size_t v = __riscv_vsetvl_e8m1(d - i);
        vuint8m1_t c8 = __riscv_vlse8_v_u8m1(
                p + i * kBlockVectors,
                static_cast<ptrdiff_t>(kBlockVectors),
                v);
        vfloat32m4_t cf =
                __riscv_vfcvt_f_xu_v_f32m4(__riscv_vzext_vf4_u32m4(c8, v), v);
        vfloat32m4_t t = __riscv_vle32_v_f32m4(e + i, v);
        t = __riscv_vfnmsac_vv_f32m4(t, __riscv_vle32_v_f32m4(a + i, v), cf, v);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, t, t, v);
        i += v;
    }
    vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
            acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

/// The code-dependent inner product term for one transposed candidate.
inline float ip_partial_strided(const uint8_t* p, const float* eq, size_t d) {
    const size_t vl = __riscv_vsetvl_e8m1(d > 0 ? d : 1);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    size_t i = 0;
    while (i < d) {
        const size_t v = __riscv_vsetvl_e8m1(d - i);
        vuint8m1_t c8 = __riscv_vlse8_v_u8m1(
                p + i * kBlockVectors,
                static_cast<ptrdiff_t>(kBlockVectors),
                v);
        vfloat32m4_t cf =
                __riscv_vfcvt_f_xu_v_f32m4(__riscv_vzext_vf4_u32m4(c8, v), v);
        acc = __riscv_vfmacc_vv_f32m4_tu(
                acc, __riscv_vle32_v_f32m4(eq + i, v), cf, v);
        i += v;
    }
    vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
            acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

/// Squared L2 on the 8-bit grid for a transposed block, with integer
/// weights. The host scales the result back.
inline void l2_distance_int_block(
        const uint8_t* block,
        const uint8_t* qcode,
        const int32_t* w,
        size_t d,
        float* out) {
    for (size_t c = 0; c < kBlockVectors;) {
        const size_t vl = __riscv_vsetvl_e32m4(kBlockVectors - c);
        vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
        for (size_t i = 0; i < d; i++) {
            vuint8m1_t c8 =
                    __riscv_vle8_v_u8m1(block + c + i * kBlockVectors, vl);
            vint32m4_t c = __riscv_vreinterpret_v_u32m4_i32m4(
                    __riscv_vzext_vf4_u32m4(c8, vl));
            vint32m4_t diff = __riscv_vsub_vx_i32m4(
                    c, static_cast<int32_t>(qcode[i]), vl);
            vint32m4_t sq = __riscv_vmul_vv_i32m4(diff, diff, vl);
            acc = __riscv_vmacc_vx_i32m4(acc, w[i], sq, vl);
        }
        __riscv_vse32_v_f32m4(out + c, __riscv_vfcvt_f_x_v_f32m4(acc, vl), vl);
        c += vl;
    }
}

/// The code-dependent term of the inner product, for a transposed block.
inline void ip_partial_block(
        const uint8_t* block,
        const float* eq,
        size_t d,
        float* out) {
    for (size_t c = 0; c < kBlockVectors;) {
        const size_t vl = __riscv_vsetvl_e32m4(kBlockVectors - c);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t i = 0; i < d; i++) {
            vuint8m1_t c8 =
                    __riscv_vle8_v_u8m1(block + c + i * kBlockVectors, vl);
            vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(c8, vl), vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, eq[i], cf, vl);
        }
        __riscv_vse32_v_f32m4(out + c, acc, vl);
        c += vl;
    }
}

/// Squared L2 distance between a query and one 8-bit code.
inline float l2_distance(
        const uint8_t* code,
        const float* a,
        const float* e,
        size_t d) {
    // VLMAX for e8m1 equals the f32 lane count of an m4 group. Guard d == 0,
    // because vsetvl then returns 0 and the chunk loop stalls.
    const size_t vl = __riscv_vsetvl_e8m1(d > 0 ? d : 1);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    size_t i = 0;
    if (i + vl <= d) {
        // Software pipeline: load the next chunk's code bytes at the top.
        vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vl);
        i += vl;

        for (; i + vl <= d; i += vl) {
            vuint8m1_t c8_next = __riscv_vle8_v_u8m1(code + i, vl);
            vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(c8, vl), vl);
            vfloat32m4_t t = __riscv_vle32_v_f32m4(e + i - vl, vl);
            t = __riscv_vfnmsac_vv_f32m4(
                    t, __riscv_vle32_v_f32m4(a + i - vl, vl), cf, vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
            c8 = c8_next;
        }

        vfloat32m4_t cf =
                __riscv_vfcvt_f_xu_v_f32m4(__riscv_vzext_vf4_u32m4(c8, vl), vl);
        vfloat32m4_t t = __riscv_vle32_v_f32m4(e + i - vl, vl);
        t = __riscv_vfnmsac_vv_f32m4(
                t, __riscv_vle32_v_f32m4(a + i - vl, vl), cf, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
    }

    // Tail: fewer than vl dimensions remain. The reduction covers every lane.
    if (i < d) {
        const size_t vt = __riscv_vsetvl_e8m1(d - i);
        vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vt);
        vfloat32m4_t cf =
                __riscv_vfcvt_f_xu_v_f32m4(__riscv_vzext_vf4_u32m4(c8, vt), vt);
        vfloat32m4_t t = __riscv_vle32_v_f32m4(e + i, vt);
        t = __riscv_vfnmsac_vv_f32m4(
                t, __riscv_vle32_v_f32m4(a + i, vt), cf, vt);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, t, t, vt);
    }

    vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
            acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

/// Inner product between a query and one 8-bit code, without the constant.
///
/// Expanding the reconstruction,
///
///     sum q[i] * v[i] = sum q[i]*rmin[i]  +  sum (q[i]*a[i]) * code[i]
///
/// the first term does not depend on the code, so the caller adds it. This
/// computes the second, with u[i] = q[i] * a[i].
inline float ip_partial(const uint8_t* code, const float* u, size_t d) {
    const size_t vl = __riscv_vsetvl_e8m1(d > 0 ? d : 1);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    size_t i = 0;
    for (; i + vl <= d; i += vl) {
        vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vl);
        vfloat32m4_t cf =
                __riscv_vfcvt_f_xu_v_f32m4(__riscv_vzext_vf4_u32m4(c8, vl), vl);
        acc = __riscv_vfmacc_vv_f32m4(
                acc, cf, __riscv_vle32_v_f32m4(u + i, vl), vl);
    }
    if (i < d) {
        const size_t vt = __riscv_vsetvl_e8m1(d - i);
        vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vt);
        vfloat32m4_t cf =
                __riscv_vfcvt_f_xu_v_f32m4(__riscv_vzext_vf4_u32m4(c8, vt), vt);
        acc = __riscv_vfmacc_vv_f32m4_tu(
                acc, cf, __riscv_vle32_v_f32m4(u + i, vt), vt);
    }

    vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
            acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

/// Squared L2 distance with the query quantized onto the same 8-bit grid.
///
///     sum (q[i] - v[i])^2  =  sum a[i]^2 * (qcode[i] - code[i])^2
///
/// The weights hold a[i]^2 as integers, so the loop is integer only. The
/// caller scales the result back and must keep max(w) * 65025 * d inside
/// int32.
inline float l2_distance_int(
        const uint8_t* code,
        const uint8_t* qcode,
        const int32_t* w,
        size_t d) {
    // Accumulate at LMUL 4, not 8. A group of eight uses eight of the 32
    // vector registers and leaves only four groups, which is slower.
    vint32m4_t sum = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvlmax_e32m4());
    size_t i = 0;
    while (i < d) {
        const size_t vl = __riscv_vsetvl_e16m2(d - i);
        vint16m2_t c = __riscv_vreinterpret_v_u16m2_i16m2(
                __riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(code + i, vl), vl));
        vint16m2_t q =
                __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(
                        __riscv_vle8_v_u8m1(qcode + i, vl), vl));
        vint16m2_t diff = __riscv_vsub_vv_i16m2(q, c, vl);
        // A difference of two bytes fits in int16 and its square in int32.
        vint32m4_t sq = __riscv_vwmul_vv_i32m4(diff, diff, vl);
        sum = __riscv_vmacc_vv_i32m4_tu(
                sum, sq, __riscv_vle32_v_i32m4(w + i, vl), vl);
        i += vl;
    }
    vint32m1_t r = __riscv_vredsum_vs_i32m4_i32m1(
            sum, __riscv_vmv_v_x_i32m1(0, 1), __riscv_vsetvlmax_e32m4());
    return static_cast<float>(__riscv_vmv_x_s_i32m1_i32(r));
}

} // namespace sq8_rvv
} // namespace faiss
