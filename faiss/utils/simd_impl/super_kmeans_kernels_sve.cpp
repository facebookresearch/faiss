/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_SVE

#include <faiss/utils/simd_impl/super_kmeans_kernels.h>

#include <arm_sve.h>

namespace faiss {
namespace detail {

template <>
float block_l2<SIMDLevel::ARM_SVE>(const float* x, const float* y, int n) {
    svfloat32_t acc = svdup_n_f32(0.0f);
    const int lanes = static_cast<int>(svcntw());
    for (int m = 0; m < n; m += lanes) {
        const svbool_t pg = svwhilelt_b32(m, n);
        const svfloat32_t xv = svld1_f32(pg, x + m);
        const svfloat32_t yv = svld1_f32(pg, y + m);
        const svfloat32_t diff = svsub_f32_x(pg, xv, yv);
        acc = svmla_f32_m(pg, acc, diff, diff);
    }
    return svaddv_f32(svptrue_b32(), acc);
}

} // namespace detail
} // namespace faiss

#endif // COMPILE_SIMD_ARM_SVE
