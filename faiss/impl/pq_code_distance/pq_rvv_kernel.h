/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/* 8-bit product-quantizer code distance for RISC-V Vector.
 *
 * The header is free-standing: it includes only <cstddef>, <cstdint> and
 * <riscv_vector.h>. It uses no exceptions, no heap and no threads, so a
 * freestanding RISC-V target can compile it. pq_code_distance/rvv.cpp calls
 * this, so the CPU tests cover it.
 *
 * PQDecoder8 fixes nbits = 8, so ksub = 256 and the scalar loop is
 *
 *     for (m = 0; m < M; m++) result += sim_table[m * 256 + code[m]];
 *
 * That is a gather, not a contiguous read. RVV issues one vluxei32 per strip:
 * build the element offset m * 256 + code[m] across the lanes, scale it to
 * bytes, then index sim_table.
 *
 * The offset stays inside uint32 while M * 256 * 4 bytes does. At M = 1024
 * that is 1 MiB, so no PQ shape in Faiss overflows it.
 *
 * The accumulator is LMUL 4, not 8, for the same reason as
 * utils/simd_impl/distances_rvv_kernel.h.
 */

#include <cstddef>
#include <cstdint>

#include <riscv_vector.h>

namespace faiss {
namespace pq_rvv_kernel {

/// Element offsets into sim_table for subquantizers [m, m + vl).
inline vuint32m4_t row_offsets(size_t m, size_t vl) {
    // (m + lane) * 256
    vuint32m4_t row = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl), m, vl);
    return __riscv_vsll_vx_u32m4(row, 8, vl);
}

/// Byte offsets for one code, given the precomputed row offsets.
inline vuint32m4_t byte_offsets(
        vuint32m4_t row_off,
        const uint8_t* code,
        size_t m,
        size_t vl) {
    vuint8m1_t cb = __riscv_vle8_v_u8m1(code + m, vl);
    vuint32m4_t idx = __riscv_vzext_vf4_u32m4(cb, vl);
    // (row + code) * sizeof(float)
    return __riscv_vsll_vx_u32m4(
            __riscv_vadd_vv_u32m4(row_off, idx, vl), 2, vl);
}

inline float reduce(vfloat32m4_t acc, size_t vlmax) {
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m4_f32m1(acc, zero, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

/// Distance between a query and one 8-bit PQ code.
/// @param sim_table  M * 256 floats, laid out (M, ksub)
inline float distance_8bit(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    size_t m = 0;
    while (m < M) {
        const size_t vl = __riscv_vsetvl_e32m4(M - m);
        vuint32m4_t off = byte_offsets(row_offsets(m, vl), code, m, vl);
        vfloat32m4_t v = __riscv_vluxei32_v_f32m4(sim_table, off, vl);
        // Tail-undisturbed: lanes past vl keep earlier partial sums, and the
        // final reduction covers vlmax.
        acc = __riscv_vfadd_vv_f32m4_tu(acc, acc, v, vl);
        m += vl;
    }
    return reduce(acc, vlmax);
}

} // namespace pq_rvv_kernel
} // namespace faiss
