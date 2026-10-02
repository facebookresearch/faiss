/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/* Vector-to-vector distance kernels for RISC-V Vector.
 *
 * The header is free-standing: it includes only <cstddef> and
 * <riscv_vector.h>. It uses no exceptions, no heap and no threads, so a
 * freestanding RISC-V target can compile it. distances_rvv.cpp calls these,
 * so the CPU tests cover them.
 *
 * These inline the horizontal reduction rather than passing the intrinsic to
 * a helper. Clang rejects a builtin used as a function argument.
 *
 * They accumulate at LMUL 4, not 8. A group of eight uses eight of the 32
 * vector registers and leaves only four groups. Measured on MTIA Artemis at
 * d=128, called repeatedly the way a list scan calls it, the same arithmetic
 * costs 67 nanoseconds per call at LMUL 8 and 51 at LMUL 4.

 */

#include <cstddef>

#include <riscv_vector.h>

namespace faiss {
namespace rvv_kernel {

/// Squared L2 distance between two fp32 vectors.
inline float l2_sqr(const float* x, const float* y, size_t d) {
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        const size_t vl = __riscv_vsetvl_e32m4(d - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t vy = __riscv_vle32_v_f32m4(y + i, vl);
        vx = __riscv_vfsub_vv_f32m4(vx, vy, vl);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, vx, vx, vl);
        i += vl;
    }
    vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    vfloat32m1_t r = __riscv_vfredusum_vs_f32m4_f32m1(acc, init, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(r);
}

/// Inner product of two fp32 vectors.
inline float inner_product(const float* x, const float* y, size_t d) {
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        const size_t vl = __riscv_vsetvl_e32m4(d - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t vy = __riscv_vle32_v_f32m4(y + i, vl);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, vx, vy, vl);
        i += vl;
    }
    vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    vfloat32m1_t r = __riscv_vfredusum_vs_f32m4_f32m1(acc, init, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(r);
}

} // namespace rvv_kernel
} // namespace faiss
