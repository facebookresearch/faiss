/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#define AUTOVEC_LEVEL SIMDLevel::ARM_NEON
#include <faiss/utils/simd_impl/distances_autovec-inl.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::ARM_NEON>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    const size_t n_simd = n - (n & 3);
    const float32x4_t bfv = vdupq_n_f32(bf);
    size_t i;
    for (i = 0; i < n_simd; i += 4) {
        const float32x4_t ai = vld1q_f32(a + i);
        const float32x4_t bi = vld1q_f32(b + i);
        const float32x4_t ci = vfmaq_f32(ai, bfv, bi);
        vst1q_f32(c + i, ci);
    }
    for (; i < n; ++i)
        c[i] = a[i] + bf * b[i];
}

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny);

template <>
void fvec_inner_products_ny<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_inner_products_ny<SIMDLevel::NONE>(dis, x, y, d, ny);
}

template <>
void fvec_L2sqr_ny<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny<SIMDLevel::NONE>(dis, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::ARM_NEON>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::ARM_NEON>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed_ref<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

template <>
int fvec_madd_and_argmin<SIMDLevel::ARM_NEON>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    float32x4_t vminv = vdupq_n_f32(1e20);
    uint32x4_t iminv = vdupq_n_u32(static_cast<uint32_t>(-1));
    size_t i;
    {
        const size_t n_simd = n - (n & 3);
        const uint32_t iota[] = {0, 1, 2, 3};
        uint32x4_t iv = vld1q_u32(iota);
        const uint32x4_t incv = vdupq_n_u32(4);
        const float32x4_t bfv = vdupq_n_f32(bf);
        for (i = 0; i < n_simd; i += 4) {
            const float32x4_t ai = vld1q_f32(a + i);
            const float32x4_t bi = vld1q_f32(b + i);
            const float32x4_t ci = vfmaq_f32(ai, bfv, bi);
            vst1q_f32(c + i, ci);
            const uint32x4_t less_than = vcltq_f32(ci, vminv);
            vminv = vminq_f32(ci, vminv);
            iminv = vorrq_u32(
                    vandq_u32(less_than, iv),
                    vandq_u32(vmvnq_u32(less_than), iminv));
            iv = vaddq_u32(iv, incv);
        }
    }
    float vmin = vminvq_f32(vminv);
    uint32_t imin;
    {
        const float32x4_t vminy = vdupq_n_f32(vmin);
        const uint32x4_t equals = vceqq_f32(vminv, vminy);
        imin = vminvq_u32(vorrq_u32(
                vandq_u32(equals, iminv),
                vandq_u32(
                        vmvnq_u32(equals),
                        vdupq_n_u32(std::numeric_limits<uint32_t>::max()))));
    }
    for (; i < n; ++i) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = static_cast<uint32_t>(i);
        }
    }
    return static_cast<int>(imin);
}

} // namespace faiss
