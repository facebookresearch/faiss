/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#ifdef COMPILE_SIMD_RISCV_RVV

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/extra_distances.h>
#include <riscv_vector.h>
#include <vector>

namespace faiss {

template <typename Vec, typename Reduce>
static inline float rvv_reduce(
        Vec value,
        size_t vl,
        float identity,
        Reduce reduce) {
    vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(identity, 1);
    vfloat32m1_t result = reduce(value, init, vl);
    return __riscv_vfmv_f_s_f32m1_f32(result);
}

static inline size_t rvv_argmin(const float* values, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vmin = __riscv_vfmv_v_f_f32m8(__builtin_inff(), vlmax);
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vd = __riscv_vle32_v_f32m8(values + i, vl);
        vmin = __riscv_vfmin_vv_f32m8_tu(vmin, vmin, vd, vl);
        i += vl;
    }
    float min_val = rvv_reduce(
            vmin, vlmax, __builtin_inff(), __riscv_vfredmin_vs_f32m8_f32m1);
    i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vd = __riscv_vle32_v_f32m8(values + i, vl);
        long j = __riscv_vfirst_m_b4(
                __riscv_vmfeq_vf_f32m8_b4(vd, min_val, vl), vl);
        if (j >= 0)
            return i + static_cast<size_t>(j);
        i += vl;
    }
    return n;
}

// Write the ksub table entries of one subquantizer that has dsub = 2.
// The function computes one of these 2 results:
//   inner product: out[k] = x0 * c0[k] + x1 * c1[k]
//   L2:            out[k] = (c0[k] - x0)^2 + (c1[k] - x1)^2
template <bool is_inner_product>
static inline void rvv_pq2_row(
        const float* c0,
        const float* c1,
        float x0,
        float x1,
        size_t ksub,
        size_t vl,
        float* out) {
    for (size_t k = 0; k < ksub; k += vl) {
        vfloat32m1_t vc0 = __riscv_vle32_v_f32m1(c0 + k, vl);
        vfloat32m1_t vc1 = __riscv_vle32_v_f32m1(c1 + k, vl);
        if constexpr (is_inner_product) {
            vfloat32m1_t r = __riscv_vfmacc_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vc0, x0, vl), x1, vc1, vl);
            __riscv_vse32_v_f32m1(out + k, r, vl);
        } else {
            vfloat32m1_t d0 = __riscv_vfsub_vf_f32m1(vc0, x0, vl);
            vfloat32m1_t d1 = __riscv_vfsub_vf_f32m1(vc1, x1, vl);
            d0 = __riscv_vfmul_vv_f32m1(d0, d0, vl);
            d1 = __riscv_vfmul_vv_f32m1(d1, d1, vl);
            vfloat32m1_t r = __riscv_vfadd_vv_f32m1(d0, d1, vl);
            __riscv_vse32_v_f32m1(out + k, r, vl);
        }
    }
}

template <bool is_inner_product>
static void rvv_pq2_tables(
        size_t M,
        size_t ksub,
        const float* c0_all,
        const float* c1_all,
        size_t nx,
        const float* x,
        size_t vl,
        float* dis_tables) {
    const size_t d = 2 * M;
    for (size_t i = 0; i < nx; i++) {
        const float* xi = x + i * d;
        float* oi = dis_tables + i * M * ksub;
        size_t m = 0;
        // The loop computes 2 subquantizers per iteration. The 2 computations
        // are independent, so the processor can run them at the same time.
        for (; m + 1 < M; m += 2) {
            rvv_pq2_row<is_inner_product>(
                    c0_all + m * ksub,
                    c1_all + m * ksub,
                    xi[2 * m],
                    xi[2 * m + 1],
                    ksub,
                    vl,
                    oi + m * ksub);
            rvv_pq2_row<is_inner_product>(
                    c0_all + (m + 1) * ksub,
                    c1_all + (m + 1) * ksub,
                    xi[2 * (m + 1)],
                    xi[2 * (m + 1) + 1],
                    ksub,
                    vl,
                    oi + (m + 1) * ksub);
        }
        // If M is odd, the loop above skips the last subquantizer. This loop
        // computes it.
        for (; m < M; m++) {
            rvv_pq2_row<is_inner_product>(
                    c0_all + m * ksub,
                    c1_all + m * ksub,
                    xi[2 * m],
                    xi[2 * m + 1],
                    ksub,
                    vl,
                    oi + m * ksub);
        }
    }
}

template <>
void compute_PQ_dis_tables_dsub2<SIMDLevel::RISCV_RVV>(
        size_t d,
        size_t ksub,
        const float* all_centroids,
        size_t nx,
        const float* x,
        bool is_inner_product,
        float* dis_tables) {
    const size_t M = d / 2;
    FAISS_THROW_IF_NOT(ksub % 8 == 0);

    // Separate the centroids into 2 planes. Each plane holds one component of
    // every centroid, so a unit-stride load can read the plane.
    std::vector<float> c0_all(M * ksub);
    std::vector<float> c1_all(M * ksub);
    for (size_t m = 0; m < M; m++) {
        const float* cm = all_centroids + m * ksub * 2;
        float* c0m = c0_all.data() + m * ksub;
        float* c1m = c1_all.data() + m * ksub;
        for (size_t k = 0; k < ksub; k++) {
            c0m[k] = cm[2 * k];
            c1m[k] = cm[2 * k + 1];
        }
    }

    // ksub is a power of 2, and ksub is a multiple of 8. Therefore vl divides
    // ksub exactly.
    const size_t vl = __riscv_vsetvl_e32m1(ksub);

    if (is_inner_product) {
        rvv_pq2_tables<true>(
                M, ksub, c0_all.data(), c1_all.data(), nx, x, vl, dis_tables);
    } else {
        rvv_pq2_tables<false>(
                M, ksub, c0_all.data(), c1_all.data(), nx, x, vl, dis_tables);
    }
}

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
    return rvv_reduce(acc, vlmax, 0.0f, __riscv_vfredusum_vs_f32m8_f32m1);
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
    return rvv_reduce(acc, vlmax, 0.0f, __riscv_vfredusum_vs_f32m8_f32m1);
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
    return rvv_reduce(acc, vlmax, 0.0f, __riscv_vfredusum_vs_f32m8_f32m1);
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
    return rvv_reduce(acc, vlmax, 0.0f, __riscv_vfredusum_vs_f32m8_f32m1);
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
    return rvv_reduce(vmax, vlmax, 0.0f, __riscv_vfredmax_vs_f32m8_f32m1);
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
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vacc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t vacc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t vacc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t vacc3 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m4(d - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t vy = __riscv_vle32_v_f32m4(y0 + i, vl);
        vacc0 = __riscv_vfmacc_vv_f32m4_tu(vacc0, vx, vy, vl);
        vy = __riscv_vle32_v_f32m4(y1 + i, vl);
        vacc1 = __riscv_vfmacc_vv_f32m4_tu(vacc1, vx, vy, vl);
        vy = __riscv_vle32_v_f32m4(y2 + i, vl);
        vacc2 = __riscv_vfmacc_vv_f32m4_tu(vacc2, vx, vy, vl);
        vy = __riscv_vle32_v_f32m4(y3 + i, vl);
        vacc3 = __riscv_vfmacc_vv_f32m4_tu(vacc3, vx, vy, vl);
        i += vl;
    }
    dis0 = rvv_reduce(vacc0, vlmax, 0.0f, __riscv_vfredusum_vs_f32m4_f32m1);
    dis1 = rvv_reduce(vacc1, vlmax, 0.0f, __riscv_vfredusum_vs_f32m4_f32m1);
    dis2 = rvv_reduce(vacc2, vlmax, 0.0f, __riscv_vfredusum_vs_f32m4_f32m1);
    dis3 = rvv_reduce(vacc3, vlmax, 0.0f, __riscv_vfredusum_vs_f32m4_f32m1);
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
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vacc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t vacc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t vacc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t vacc3 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m4(d - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t vy = __riscv_vle32_v_f32m4(y0 + i, vl);
        vy = __riscv_vfsub_vv_f32m4(vx, vy, vl);
        vacc0 = __riscv_vfmacc_vv_f32m4_tu(vacc0, vy, vy, vl);
        vy = __riscv_vle32_v_f32m4(y1 + i, vl);
        vy = __riscv_vfsub_vv_f32m4(vx, vy, vl);
        vacc1 = __riscv_vfmacc_vv_f32m4_tu(vacc1, vy, vy, vl);
        vy = __riscv_vle32_v_f32m4(y2 + i, vl);
        vy = __riscv_vfsub_vv_f32m4(vx, vy, vl);
        vacc2 = __riscv_vfmacc_vv_f32m4_tu(vacc2, vy, vy, vl);
        vy = __riscv_vle32_v_f32m4(y3 + i, vl);
        vy = __riscv_vfsub_vv_f32m4(vx, vy, vl);
        vacc3 = __riscv_vfmacc_vv_f32m4_tu(vacc3, vy, vy, vl);
        i += vl;
    }
    dis0 = rvv_reduce(vacc0, vlmax, 0.0f, __riscv_vfredusum_vs_f32m4_f32m1);
    dis1 = rvv_reduce(vacc1, vlmax, 0.0f, __riscv_vfredusum_vs_f32m4_f32m1);
    dis2 = rvv_reduce(vacc2, vlmax, 0.0f, __riscv_vfredusum_vs_f32m4_f32m1);
    dis3 = rvv_reduce(vacc3, vlmax, 0.0f, __riscv_vfredusum_vs_f32m4_f32m1);
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
    // Compute the squared length of the query subvector.
    float x_sqlen = 0;
    for (size_t j = 0; j < d; j++) {
        x_sqlen += x[j] * x[j];
    }

    // The loop below divides the ny dimension into chunks. Each chunk uses
    // e32m8. The chunk size is VLMAX, so the kernel fills the whole vector
    // register at every VLEN.
    const size_t chunk = __riscv_vsetvlmax_e32m8();
    size_t i = 0;

    // dis[i] = x_sqlen + y_sqlen[i] - 2 * sum_j x[j] * y[j * d_offset + i]
    //
    // The code accumulates the dot product first, and it applies the factor
    // of 2 afterwards. Matches SIMDLevel::NONE implementation which scales the
    // accumulated dot product in the same way.
    for (; i + chunk <= ny; i += chunk) {
        vfloat32m8_t dp = __riscv_vfmv_v_f_f32m8(0.0f, chunk);
        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t y_vec =
                    __riscv_vle32_v_f32m8(y + j * d_offset + i, chunk);
            dp = __riscv_vfmacc_vf_f32m8(dp, x[j], y_vec, chunk);
        }

        vfloat32m8_t acc = __riscv_vle32_v_f32m8(y_sqlen + i, chunk);
        acc = __riscv_vfadd_vf_f32m8(acc, x_sqlen, chunk);
        dp = __riscv_vfmul_vf_f32m8(dp, 2.0f, chunk);
        acc = __riscv_vfsub_vv_f32m8(acc, dp, chunk);

        __riscv_vse32_v_f32m8(dis + i, acc, chunk);
    }

    // This block computes the last ny % chunk elements.
    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);

        vfloat32m8_t dp = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t y_vec =
                    __riscv_vle32_v_f32m8(y + j * d_offset + i, vl);
            dp = __riscv_vfmacc_vf_f32m8(dp, x[j], y_vec, vl);
        }

        vfloat32m8_t acc = __riscv_vle32_v_f32m8(y_sqlen + i, vl);
        acc = __riscv_vfadd_vf_f32m8(acc, x_sqlen, vl);
        dp = __riscv_vfmul_vf_f32m8(dp, 2.0f, vl);
        acc = __riscv_vfsub_vv_f32m8(acc, dp, vl);

        __riscv_vse32_v_f32m8(dis + i, acc, vl);
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::RISCV_RVV>(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    // The loop below divides the ny dimension into chunks. Each chunk uses
    // e32m4.
    // ip[i] = sum_{j=0}^{d-1} x[j] * y[i * d + j]
    const size_t chunk = __riscv_vsetvlmax_e32m4();
    const ptrdiff_t stride_bytes = (ptrdiff_t)d * (ptrdiff_t)sizeof(float);
    size_t i = 0;

    for (; i + chunk <= ny; i += chunk) {
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, chunk);
        const float* yb = y + i * d; // the first vector of this chunk

        for (size_t j = 0; j < d; j++) {
            vfloat32m4_t y_vec =
                    __riscv_vlse32_v_f32m4(yb + j, stride_bytes, chunk);
            acc = __riscv_vfmacc_vf_f32m4(acc, x[j], y_vec, chunk);
        }

        __riscv_vse32_v_f32m4(ip + i, acc, chunk);
    }

    // This block computes the last ny % chunk elements.
    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m4(ny - i);

        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        const float* yb = y + i * d;

        for (size_t j = 0; j < d; j++) {
            vfloat32m4_t y_vec =
                    __riscv_vlse32_v_f32m4(yb + j, stride_bytes, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, x[j], y_vec, vl);
        }

        __riscv_vse32_v_f32m4(ip + i, acc, vl);
    }
}

template <>
void fvec_L2sqr_ny<SIMDLevel::RISCV_RVV>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    // The loop below divides the ny dimension into chunks. Each chunk uses
    // e32m8.
    const size_t chunk = __riscv_vsetvlmax_e32m8();
    const ptrdiff_t stride_bytes = (ptrdiff_t)d * (ptrdiff_t)sizeof(float);
    size_t i = 0;

    for (; i + chunk <= ny; i += chunk) {
        vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, chunk);
        const float* yb = y + i * d; // the first vector of this chunk
        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t y_vec =
                    __riscv_vlse32_v_f32m8(yb + j, stride_bytes, chunk);
            vfloat32m8_t diff = __riscv_vfsub_vf_f32m8(y_vec, x[j], chunk);
            acc = __riscv_vfmacc_vv_f32m8(acc, diff, diff, chunk);
        }
        __riscv_vse32_v_f32m8(dis + i, acc, chunk);
    }

    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);
        vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        const float* yb = y + i * d;
        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t y_vec =
                    __riscv_vlse32_v_f32m8(yb + j, stride_bytes, vl);
            vfloat32m8_t diff = __riscv_vfsub_vf_f32m8(y_vec, x[j], vl);
            acc = __riscv_vfmacc_vv_f32m8(acc, diff, diff, vl);
        }
        __riscv_vse32_v_f32m8(dis + i, acc, vl);
    }
}

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::RISCV_RVV>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, d, ny);
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
    return fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
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
    const size_t j = rvv_argmin(c, n);
    return j < n ? static_cast<int>(j) : -1;
}

template <>
void fvec_add<SIMDLevel::RISCV_RVV>(
        size_t d,
        const float* a,
        const float* b,
        float* c) {
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        va = __riscv_vfadd_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(c + i, va, vl);
        i += vl;
    }
}

template <>
void fvec_add<SIMDLevel::RISCV_RVV>(
        size_t d,
        const float* a,
        float b,
        float* c) {
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        va = __riscv_vfadd_vf_f32m8(va, b, vl);
        __riscv_vse32_v_f32m8(c + i, va, vl);
        i += vl;
    }
}

template <>
void fvec_sub<SIMDLevel::RISCV_RVV>(
        size_t d,
        const float* a,
        const float* b,
        float* c) {
    size_t i = 0;
    while (i < d) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        va = __riscv_vfsub_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(c + i, va, vl);
        i += vl;
    }
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
