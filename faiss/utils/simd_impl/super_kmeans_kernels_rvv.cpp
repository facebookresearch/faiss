/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_RISCV_RVV

#include <faiss/utils/simd_impl/super_kmeans_kernels.h>

#include <riscv_vector.h>

namespace faiss {
namespace detail {

// RVV block_l2: two accumulators + masked tail, LMUL chosen at run time.
namespace {

__attribute__((always_inline)) inline float block_l2_m1(
        const float* x,
        const float* y,
        int n) {
    const size_t vl = __riscv_vsetvlmax_e32m1();
    const int ivl = static_cast<int>(vl);
    vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

    int m = 0;
    for (; m + 2 * ivl <= n; m += 2 * ivl) {
        vfloat32m1_t x0 = __riscv_vle32_v_f32m1(x + m, vl);
        vfloat32m1_t y0 = __riscv_vle32_v_f32m1(y + m, vl);
        vfloat32m1_t d0 = __riscv_vfsub_vv_f32m1(x0, y0, vl);
        acc0 = __riscv_vfmacc_vv_f32m1(acc0, d0, d0, vl);

        vfloat32m1_t x1 = __riscv_vle32_v_f32m1(x + m + ivl, vl);
        vfloat32m1_t y1 = __riscv_vle32_v_f32m1(y + m + ivl, vl);
        vfloat32m1_t d1 = __riscv_vfsub_vv_f32m1(x1, y1, vl);
        acc1 = __riscv_vfmacc_vv_f32m1(acc1, d1, d1, vl);
    }
    for (; m + ivl <= n; m += ivl) {
        vfloat32m1_t x0 = __riscv_vle32_v_f32m1(x + m, vl);
        vfloat32m1_t y0 = __riscv_vle32_v_f32m1(y + m, vl);
        vfloat32m1_t d0 = __riscv_vfsub_vv_f32m1(x0, y0, vl);
        acc0 = __riscv_vfmacc_vv_f32m1(acc0, d0, d0, vl);
    }
    if (m < n) {
        const size_t vlt = __riscv_vsetvl_e32m1(static_cast<size_t>(n - m));
        vfloat32m1_t x0 = __riscv_vle32_v_f32m1(x + m, vlt);
        vfloat32m1_t y0 = __riscv_vle32_v_f32m1(y + m, vlt);
        vfloat32m1_t d0 = __riscv_vfsub_vv_f32m1(x0, y0, vlt);
        acc0 = __riscv_vfmacc_vv_f32m1(acc0, d0, d0, vlt);
    }

    vfloat32m1_t acc = __riscv_vfadd_vv_f32m1(acc0, acc1, vl);
    vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m1_f32m1(acc, seed, vl));
}

__attribute__((always_inline)) inline float block_l2_m2(
        const float* x,
        const float* y,
        int n) {
    const size_t vl = __riscv_vsetvlmax_e32m2();
    const int ivl = static_cast<int>(vl);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    int m = 0;
    for (; m + 2 * ivl <= n; m += 2 * ivl) {
        vfloat32m2_t x0 = __riscv_vle32_v_f32m2(x + m, vl);
        vfloat32m2_t y0 = __riscv_vle32_v_f32m2(y + m, vl);
        vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(x0, y0, vl);
        acc0 = __riscv_vfmacc_vv_f32m2(acc0, d0, d0, vl);

        vfloat32m2_t x1 = __riscv_vle32_v_f32m2(x + m + ivl, vl);
        vfloat32m2_t y1 = __riscv_vle32_v_f32m2(y + m + ivl, vl);
        vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(x1, y1, vl);
        acc1 = __riscv_vfmacc_vv_f32m2(acc1, d1, d1, vl);
    }
    for (; m + ivl <= n; m += ivl) {
        vfloat32m2_t x0 = __riscv_vle32_v_f32m2(x + m, vl);
        vfloat32m2_t y0 = __riscv_vle32_v_f32m2(y + m, vl);
        vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(x0, y0, vl);
        acc0 = __riscv_vfmacc_vv_f32m2(acc0, d0, d0, vl);
    }
    if (m < n) {
        const size_t vlt = __riscv_vsetvl_e32m2(static_cast<size_t>(n - m));
        vfloat32m2_t x0 = __riscv_vle32_v_f32m2(x + m, vlt);
        vfloat32m2_t y0 = __riscv_vle32_v_f32m2(y + m, vlt);
        vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(x0, y0, vlt);
        acc0 = __riscv_vfmacc_vv_f32m2(acc0, d0, d0, vlt);
    }

    vfloat32m2_t acc = __riscv_vfadd_vv_f32m2(acc0, acc1, vl);
    vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(acc, seed, vl));
}

} // namespace

template <>
float block_l2<SIMDLevel::RISCV_RVV>(const float* x, const float* y, int n) {
    // e32m2 VLMAX = 2 * VLEN / 32. VLEN=128 -> 8; VLEN>=256 -> >= 16.
    // A plain vsetvlmax + branch costs ~1 cycle and is negligible next to the
    // 16+ loads the kernel does.
    if (__riscv_vsetvlmax_e32m2() >= 16) {
        return block_l2_m2(x, y, n);
    }
    return block_l2_m1(x, y, n);
}

// One x block against four consecutive y blocks; x is loaded once and reused
// across the four y streams (four independent m2 accumulators — m2 keeps the
// wider loads without the register pressure that four m4 accumulators would
// cause, and its reduction is done once per (block, quad)).
template <>
void block_l2_4j<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        int ystride,
        int n,
        int nj,
        float* out) {
    if (nj != 4) {
        for (int t = 0; t < nj; ++t) {
            out[t] = block_l2<SIMDLevel::RISCV_RVV>(
                    x, y + static_cast<long>(t) * ystride, n);
        }
        return;
    }

    const size_t vl = __riscv_vsetvlmax_e32m2();
    const int ivl = static_cast<int>(vl);
    const float* y0 = y;
    const float* y1 = y + static_cast<long>(ystride);
    const float* y2 = y + 2L * ystride;
    const float* y3 = y + 3L * ystride;

    vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t a1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t a2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t a3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    int m = 0;
    for (; m + ivl <= n; m += ivl) {
        vfloat32m2_t xv = __riscv_vle32_v_f32m2(x + m, vl);
        vfloat32m2_t d0 =
                __riscv_vfsub_vv_f32m2(xv, __riscv_vle32_v_f32m2(y0 + m, vl), vl);
        a0 = __riscv_vfmacc_vv_f32m2(a0, d0, d0, vl);
        vfloat32m2_t d1 =
                __riscv_vfsub_vv_f32m2(xv, __riscv_vle32_v_f32m2(y1 + m, vl), vl);
        a1 = __riscv_vfmacc_vv_f32m2(a1, d1, d1, vl);
        vfloat32m2_t d2 =
                __riscv_vfsub_vv_f32m2(xv, __riscv_vle32_v_f32m2(y2 + m, vl), vl);
        a2 = __riscv_vfmacc_vv_f32m2(a2, d2, d2, vl);
        vfloat32m2_t d3 =
                __riscv_vfsub_vv_f32m2(xv, __riscv_vle32_v_f32m2(y3 + m, vl), vl);
        a3 = __riscv_vfmacc_vv_f32m2(a3, d3, d3, vl);
    }
    if (m < n) {
        const size_t vlt = __riscv_vsetvl_e32m2(static_cast<size_t>(n - m));
        vfloat32m2_t xv = __riscv_vle32_v_f32m2(x + m, vlt);
        vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(
                xv, __riscv_vle32_v_f32m2(y0 + m, vlt), vlt);
        a0 = __riscv_vfmacc_vv_f32m2(a0, d0, d0, vlt);
        vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(
                xv, __riscv_vle32_v_f32m2(y1 + m, vlt), vlt);
        a1 = __riscv_vfmacc_vv_f32m2(a1, d1, d1, vlt);
        vfloat32m2_t d2 = __riscv_vfsub_vv_f32m2(
                xv, __riscv_vle32_v_f32m2(y2 + m, vlt), vlt);
        a2 = __riscv_vfmacc_vv_f32m2(a2, d2, d2, vlt);
        vfloat32m2_t d3 = __riscv_vfsub_vv_f32m2(
                xv, __riscv_vle32_v_f32m2(y3 + m, vlt), vlt);
        a3 = __riscv_vfmacc_vv_f32m2(a3, d3, d3, vlt);
    }
    vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(a0, seed, vl));
    out[1] = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(a1, seed, vl));
    out[2] = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(a2, seed, vl));
    out[3] = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(a3, seed, vl));
}

} // namespace detail
} // namespace faiss

#endif // COMPILE_SIMD_RISCV_RVV
