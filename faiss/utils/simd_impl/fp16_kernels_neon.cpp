/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_NEON

#include <faiss/utils/simd_impl/fp16_kernels.h>

#include <arm_neon.h>

namespace faiss {
namespace detail {

template <>
void fp16_madd<SIMDLevel::ARM_NEON>(
        size_t d,
        const uint16_t* x,
        float w,
        float* c) {
    const float32x4_t wv = vdupq_n_f32(w);
    size_t j = 0;
    for (; j + 4 <= d; j += 4) {
        const float32x4_t xv =
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x + j)));
        vst1q_f32(c + j, vaddq_f32(vld1q_f32(c + j), vmulq_f32(xv, wv)));
    }
    for (; j < d; ++j) {
        c[j] += decode_fp16(x[j]) * w;
    }
}

template <>
void fp16_to_fp32_kernel<SIMDLevel::ARM_NEON>(
        size_t n,
        const uint16_t* x,
        float* out) {
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        vst1q_f32(out + j, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x + j))));
    }
    for (; j < n; ++j) {
        out[j] = decode_fp16(x[j]);
    }
}

} // namespace detail
} // namespace faiss

#endif // COMPILE_SIMD_ARM_NEON
