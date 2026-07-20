/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/rabitq_simd.h>

#ifdef COMPILE_SIMD_RISCV_RVV

#include <riscv_vector.h>

namespace faiss::rabitq {

// SWAR per-byte popcount over a u8m4 group: each output byte holds the
// population count (0..8) of the corresponding input byte.
static inline vuint8m4_t popcount_u8m4(vuint8m4_t v, size_t vl) {
    vuint8m4_t t = __riscv_vsrl_vx_u8m4(v, 1, vl);
    t = __riscv_vand_vx_u8m4(t, 0x55, vl);
    v = __riscv_vsub_vv_u8m4(v, t, vl);
    t = __riscv_vsrl_vx_u8m4(v, 2, vl);
    t = __riscv_vand_vx_u8m4(t, 0x33, vl);
    v = __riscv_vand_vx_u8m4(v, 0x33, vl);
    v = __riscv_vadd_vv_u8m4(v, t, vl);
    t = __riscv_vsrl_vx_u8m4(v, 4, vl);
    v = __riscv_vadd_vv_u8m4(v, t, vl);
    return __riscv_vand_vx_u8m4(v, 0x0F, vl);
}

// Shared body for bitwise_{and,xor}_dot_product. @p combine applies the
// per-element bit op (AND or XOR) between the data and query bit-planes; the
// popcount of the result for query bit-plane j is weighted by 2^j.
template <typename Op>
static inline uint64_t bitwise_dot_product_rvv(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb,
        Op combine) {
    size_t vlmax = __riscv_vsetvlmax_e16m8();
    vuint16m8_t acc = __riscv_vmv_v_x_u16m8(0, vlmax);
    size_t i = 0;
    while (i < size) {
        size_t vl = __riscv_vsetvl_e8m4(size - i);
        vuint8m4_t vx = __riscv_vle8_v_u8m4(data + i, vl);
        for (size_t j = 0; j < qb; j++) {
            vuint8m4_t vq = __riscv_vle8_v_u8m4(query + j * size + i, vl);
            vuint8m4_t vp = popcount_u8m4(combine(vx, vq, vl), vl);
            vuint16m8_t vw = __riscv_vzext_vf2_u16m8(vp, vl);
            vw = __riscv_vsll_vx_u16m8(vw, j, vl);
            acc = __riscv_vadd_vv_u16m8_tu(acc, acc, vw, vl);
        }
        i += vl;
    }
    vuint32m1_t red = __riscv_vmv_v_x_u32m1(0, 1);
    red = __riscv_vwredsumu_vs_u16m8_u32m1(acc, red, vlmax);
    return __riscv_vmv_x_s_u32m1_u32(red);
}

template <>
uint64_t bitwise_and_dot_product<SIMDLevel::RISCV_RVV>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return bitwise_dot_product_rvv(
            query, data, size, qb, [](vuint8m4_t a, vuint8m4_t b, size_t vl) {
                return __riscv_vand_vv_u8m4(a, b, vl);
            });
}

template <>
BitwiseAndDotProductResult bitwise_and_dot_product_with_popcount<
        SIMDLevel::RISCV_RVV>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return bitwise_and_dot_product_with_popcount<SIMDLevel::NONE>(
            query, data, size, qb);
}

template <>
uint64_t bitwise_xor_dot_product<SIMDLevel::RISCV_RVV>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return bitwise_dot_product_rvv(
            query, data, size, qb, [](vuint8m4_t a, vuint8m4_t b, size_t vl) {
                return __riscv_vxor_vv_u8m4(a, b, vl);
            });
}

template <>
uint64_t popcount<SIMDLevel::RISCV_RVV>(const uint8_t* data, size_t size) {
    size_t vlmax = __riscv_vsetvlmax_e16m8();
    vuint16m8_t acc = __riscv_vmv_v_x_u16m8(0, vlmax);
    size_t i = 0;
    while (i < size) {
        size_t vl = __riscv_vsetvl_e8m4(size - i);
        vuint8m4_t v = popcount_u8m4(__riscv_vle8_v_u8m4(data + i, vl), vl);
        vuint16m8_t vw = __riscv_vzext_vf2_u16m8(v, vl);
        acc = __riscv_vadd_vv_u16m8_tu(acc, acc, vw, vl);
        i += vl;
    }
    vuint32m1_t red = __riscv_vmv_v_x_u32m1(0, 1);
    red = __riscv_vwredsumu_vs_u16m8_u32m1(acc, red, vlmax);
    return __riscv_vmv_x_s_u32m1_u32(red);
}

} // namespace faiss::rabitq

namespace faiss::rabitq::multibit {

static float ip_1exbit_rvv(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        float cb) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vbool4_t sb = __riscv_vlm_v_b4(sign_bits + i / 8, vl);
        vbool4_t eb = __riscv_vlm_v_b4(ex_code + i / 8, vl);
        vfloat32m8_t recon = __riscv_vfmv_v_f_f32m8(cb, vl);
        recon = __riscv_vfadd_vf_f32m8_mu(sb, recon, recon, 2.0f, vl);
        recon = __riscv_vfadd_vf_f32m8_mu(eb, recon, recon, 1.0f, vl);
        vfloat32m8_t rq = __riscv_vle32_v_f32m8(rotated_q + i, vl);
        acc = __riscv_vfmacc_vv_f32m8_tu(acc, rq, recon, vl);
        i += vl;
    }
    vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(acc, sum, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

// no-tree-vectorize keeps the ex_bits >= 2 tail on the existing scalar
// ip_scalar without GCC turning its 64-bit memcpy window into a vluxei64 gather
// at unaligned addresses, which SIGBUSes on strict-align RVV cores (SpacemiT
// X60).
template <>
__attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize"))) float
compute_inner_product<SIMDLevel::RISCV_RVV>(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    if (ex_bits == 1) {
        return ip_1exbit_rvv(sign_bits, ex_code, rotated_q, d, cb);
    }
    // ex_bits >= 2 needs strided bit-plane extraction (PEXT on x86); RVV has no
    // cheap equivalent without Zvbb, so reuse the scalar kernel.
    return ip_scalar(sign_bits, ex_code, rotated_q, 0, d, ex_bits, cb);
}

} // namespace faiss::rabitq::multibit

#endif // COMPILE_SIMD_RISCV_RVV
