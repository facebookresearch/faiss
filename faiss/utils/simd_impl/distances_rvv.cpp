/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#ifdef COMPILE_SIMD_RISCV_RVV

#include <faiss/utils/extra_distances.h>
#include <riscv_vector.h>

namespace faiss {

template <>
float fvec_norm_L2sqr<SIMDLevel::RISCV_RVV>(const float* x, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        acc = __riscv_vfmacc_vv_f32m8_tu(acc, vx, vx, vl);
        i += vl;
    }
    vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(acc, sum, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

template <>
float fvec_L2sqr<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
        vx = __riscv_vfsub_vv_f32m8(vx, vy, vl);
        acc = __riscv_vfmacc_vv_f32m8_tu(acc, vx, vx, vl);
        i += vl;
    }
    vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(acc, sum, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

template <>
float fvec_inner_product<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
        acc = __riscv_vfmacc_vv_f32m8_tu(acc, vx, vy, vl);
        i += vl;
    }
    vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(acc, sum, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

template <>
float fvec_L1<SIMDLevel::RISCV_RVV>(const float* x, const float* y, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
        vx = __riscv_vfsub_vv_f32m8(vx, vy, vl);
        vx = __riscv_vfsgnjx_vv_f32m8(vx, vx, vl);
        acc = __riscv_vfadd_vv_f32m8_tu(acc, acc, vx, vl);
        i += vl;
    }
    vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(acc, sum, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

template <>
float fvec_Linf<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vmax = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
        vx = __riscv_vfsub_vv_f32m8(vx, vy, vl);
        vx = __riscv_vfsgnjx_vv_f32m8(vx, vx, vl);
        vmax = __riscv_vfmax_vv_f32m8_tu(vmax, vmax, vx, vl);
        i += vl;
    }
    vfloat32m1_t max = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    max = __riscv_vfredmax_vs_f32m8_f32m1(vmax, max, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(max);
}

template <>
void fvec_inner_product_batch_4<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    dis0 = fvec_inner_product<SIMDLevel::RISCV_RVV>(x, y0, d);
    dis1 = fvec_inner_product<SIMDLevel::RISCV_RVV>(x, y1, d);
    dis2 = fvec_inner_product<SIMDLevel::RISCV_RVV>(x, y2, d);
    dis3 = fvec_inner_product<SIMDLevel::RISCV_RVV>(x, y3, d);
}

template <>
void fvec_L2sqr_batch_4<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    dis0 = fvec_L2sqr<SIMDLevel::RISCV_RVV>(x, y0, d);
    dis1 = fvec_L2sqr<SIMDLevel::RISCV_RVV>(x, y1, d);
    dis2 = fvec_L2sqr<SIMDLevel::RISCV_RVV>(x, y2, d);
    dis3 = fvec_L2sqr<SIMDLevel::RISCV_RVV>(x, y3, d);
}

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::RISCV_RVV>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        acc = __riscv_vfmacc_vv_f32m8_tu(acc, vx, vx, vl);
        i += vl;
    }
    vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(acc, sum, vlmax);
    float x_sqlen = __riscv_vfmv_f_s_f32m1_f32(sum);
    i = 0;
    while (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);
        acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + j * d_offset + i, vl);
            acc = __riscv_vfmacc_vf_f32m8(acc, x[j], vy, vl);
        }
        vfloat32m8_t vres = __riscv_vle32_v_f32m8(y_sqlen + i, vl);
        vres = __riscv_vfadd_vf_f32m8(vres, x_sqlen, vl);
        acc = __riscv_vfmul_vf_f32m8(acc, 2.0f, vl);
        vres = __riscv_vfsub_vv_f32m8(vres, acc, vl);
        __riscv_vse32_v_f32m8(dis + i, vres, vl);
        i += vl;
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::RISCV_RVV>(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product<SIMDLevel::RISCV_RVV>(x, y, d);
        y += d;
    }
}

template <>
void fvec_L2sqr_ny<SIMDLevel::RISCV_RVV>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr<SIMDLevel::RISCV_RVV>(x, y, d);
        y += d;
    }
}

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::RISCV_RVV>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny<SIMDLevel::RISCV_RVV>(distances_tmp_buffer, x, y, d, ny);
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vmin = __riscv_vfmv_v_f_f32m8(__builtin_inff(), vlmax);
    size_t i = 0;
    while (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);
        vfloat32m8_t vd = __riscv_vle32_v_f32m8(distances_tmp_buffer + i, vl);
        vmin = __riscv_vfmin_vv_f32m8_tu(vmin, vmin, vd, vl);
        i += vl;
    }
    vfloat32m1_t rmin = __riscv_vfmv_s_f_f32m1(__builtin_inff(), 1);
    rmin = __riscv_vfredmin_vs_f32m8_f32m1(vmin, rmin, vlmax);
    float min_dis = __riscv_vfmv_f_s_f32m1_f32(rmin);
    i = 0;
    while (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);
        vfloat32m8_t vd = __riscv_vle32_v_f32m8(distances_tmp_buffer + i, vl);
        long j = __riscv_vfirst_m_b4(
                __riscv_vmfeq_vf_f32m8_b4(vd, min_dis, vl), vl);
        if (j >= 0)
            return i + static_cast<size_t>(j);
        i += vl;
    }
    return 0;
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::RISCV_RVV>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    fvec_L2sqr_ny_transposed<SIMDLevel::RISCV_RVV>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vmin = __riscv_vfmv_v_f_f32m8(__builtin_inff(), vlmax);
    size_t i = 0;
    while (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);
        vfloat32m8_t vd = __riscv_vle32_v_f32m8(distances_tmp_buffer + i, vl);
        vmin = __riscv_vfmin_vv_f32m8_tu(vmin, vmin, vd, vl);
        i += vl;
    }
    vfloat32m1_t rmin = __riscv_vfmv_s_f_f32m1(__builtin_inff(), 1);
    rmin = __riscv_vfredmin_vs_f32m8_f32m1(vmin, rmin, vlmax);
    float min_dis = __riscv_vfmv_f_s_f32m1_f32(rmin);
    i = 0;
    while (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);
        vfloat32m8_t vd = __riscv_vle32_v_f32m8(distances_tmp_buffer + i, vl);
        long j = __riscv_vfirst_m_b4(
                __riscv_vmfeq_vf_f32m8_b4(vd, min_dis, vl), vl);
        if (j >= 0)
            return i + static_cast<size_t>(j);
        i += vl;
    }
    return 0;
}

template <>
void fvec_madd<SIMDLevel::RISCV_RVV>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        va = __riscv_vfmacc_vf_f32m8(va, bf, vb, vl);
        __riscv_vse32_v_f32m8(c + i, va, vl);
        i += vl;
    }
}

template <>
int fvec_madd_and_argmin<SIMDLevel::RISCV_RVV>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    fvec_madd<SIMDLevel::RISCV_RVV>(n, a, bf, b, c);
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vmin = __riscv_vfmv_v_f_f32m8(1e20f, vlmax);
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vc = __riscv_vle32_v_f32m8(c + i, vl);
        vmin = __riscv_vfmin_vv_f32m8_tu(vmin, vmin, vc, vl);
        i += vl;
    }
    vfloat32m1_t rmin = __riscv_vfmv_s_f_f32m1(1e20f, 1);
    rmin = __riscv_vfredmin_vs_f32m8_f32m1(vmin, rmin, vlmax);
    float min_val = __riscv_vfmv_f_s_f32m1_f32(rmin);
    i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vc = __riscv_vle32_v_f32m8(c + i, vl);
        long j = __riscv_vfirst_m_b4(
                __riscv_vmfeq_vf_f32m8_b4(vc, min_val, vl), vl);
        if (j >= 0)
            return static_cast<int>(i + static_cast<size_t>(j));
        i += vl;
    }
    return -1;
}

#define DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(metric)                 \
    template <>                                                     \
    float VectorDistance<metric, SIMDLevel::RISCV_RVV>::operator()( \
            const float* x, const float* y) const {                 \
        return VectorDistance<metric, SIMDLevel::NONE>(             \
                this->d, this->metric_arg)(x, y);                   \
    }

DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_L2)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_INNER_PRODUCT)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_L1)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Linf)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Lp)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Canberra)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_BrayCurtis)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_JensenShannon)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Jaccard)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_NaNEuclidean)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_GOWER)

#undef DEFINE_VECTOR_DISTANCE_RVV_FALLBACK

} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::RISCV_RVV
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

#endif // COMPILE_SIMD_RISCV_RVV
