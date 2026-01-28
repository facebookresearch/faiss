/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/simdlib.h>

#define AUTOVEC_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/distances_autovec-inl.h>

namespace faiss {

/*******
Functions with SIMDLevel::NONE
*/

template <>
void fvec_madd<SIMDLevel::NONE>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + bf * b[i];
}

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::NONE>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    float x_sqlen = 0;
    for (size_t j = 0; j < d; j++) {
        x_sqlen += x[j] * x[j];
    }

    for (size_t i = 0; i < ny; i++) {
        float dp = 0;
        for (size_t j = 0; j < d; j++) {
            dp += x[j] * y[i + j * d_offset];
        }

        dis[i] = x_sqlen + y_sqlen[i] - 2 * dp;
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::NONE>(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
// BLAS slower for the use cases here
#if 0
{
    FINTEGER di = d;
    FINTEGER nyi = ny;
    float one = 1.0, zero = 0.0;
    FINTEGER onei = 1;
    sgemv_ ("T", &di, &nyi, &one, y, &di, x, &onei, &zero, ip, &onei);
}
#endif
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product(x, y, d);
        y += d;
    }
}

template <>
void fvec_L2sqr_ny<SIMDLevel::NONE>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr(x, y, d);
        y += d;
    }
}

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::NONE>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny<SIMDLevel::NONE>(distances_tmp_buffer, x, y, d, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::NONE>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    fvec_L2sqr_ny_transposed<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

template <>
int fvec_madd_and_argmin<SIMDLevel::NONE>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    float vmin = 1e20;
    int imin = -1;

    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = i;
        }
    }
    return imin;
}

/*********************************************************
 * dispatching functions
 */

float fvec_L1(const float* x, const float* y, size_t d) {
    DISPATCH_SIMDLevel(fvec_L1, x, y, d);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    DISPATCH_SIMDLevel(fvec_Linf, x, y, d);
}

// dispatching functions

float fvec_norm_L2sqr(const float* x, size_t d) {
    DISPATCH_SIMDLevel(fvec_norm_L2sqr, x, d);
}

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    DISPATCH_SIMDLevel(fvec_L2sqr, x, y, d);
}

float fvec_inner_product(const float* x, const float* y, size_t d) {
    DISPATCH_SIMDLevel(fvec_inner_product, x, y, d);
}

/// Special version of inner product that computes 4 distances
/// between x and yi
void fvec_inner_product_batch_4(
        const float* __restrict x,
        const float* __restrict y0,
        const float* __restrict y1,
        const float* __restrict y2,
        const float* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    DISPATCH_SIMDLevel(
            fvec_inner_product_batch_4,
            x,
            y0,
            y1,
            y2,
            y3,
            d,
            dis0,
            dis1,
            dis2,
            dis3);
}

/// Special version of L2sqr that computes 4 distances
/// between x and yi, which is performance oriented.
void fvec_L2sqr_batch_4(
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
    DISPATCH_SIMDLevel(
            fvec_L2sqr_batch_4, x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    DISPATCH_SIMDLevel(
            fvec_L2sqr_ny_transposed, dis, x, y, y_sqlen, d, d_offset, ny);
}

void fvec_inner_products_ny(
        float* ip, /* output inner product */
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    DISPATCH_SIMDLevel(fvec_inner_products_ny, ip, x, y, d, ny);
}

void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    DISPATCH_SIMDLevel(fvec_L2sqr_ny, dis, x, y, d, ny);
}

size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    DISPATCH_SIMDLevel(
            fvec_L2sqr_ny_nearest, distances_tmp_buffer, x, y, d, ny);
}

size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    DISPATCH_SIMDLevel(
            fvec_L2sqr_ny_nearest_y_transposed,
            distances_tmp_buffer,
            x,
            y,
            y_sqlen,
            d,
            d_offset,
            ny);
}

void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c) {
    DISPATCH_SIMDLevel(fvec_madd, n, a, bf, b, c);
}

int fvec_madd_and_argmin(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    DISPATCH_SIMDLevel(fvec_madd_and_argmin, n, a, bf, b, c);
}

/*********************************************************
 * Vector to vector functions
 *********************************************************/

// TODO: Move to dynamic dispatch
// dynamic dispatch is blocked due to the following error:
// error: 'simd8float32<faiss::SIMDLevel::AVX2>::loadu' is not a member
void fvec_sub(size_t d, const float* a, const float* b, float* c) {
    size_t i;
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32<SIMDLevel::NONE> ci, ai, bi;
        ai.loadu(a + i);
        bi.loadu(b + i);
        ci = ai - bi;
        ci.storeu(c + i);
    }
    // finish non-multiple of 8 remainder
    for (; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

void fvec_add(size_t d, const float* a, const float* b, float* c) {
    size_t i;
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32<SIMDLevel::NONE> ci, ai, bi;
        ai.loadu(a + i);
        bi.loadu(b + i);
        ci = ai + bi;
        ci.storeu(c + i);
    }
    // finish non-multiple of 8 remainder
    for (; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

void fvec_add(size_t d, const float* a, float b, float* c) {
    size_t i;
    simd8float32<SIMDLevel::NONE> bv(b);
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32<SIMDLevel::NONE> ci, ai;
        ai.loadu(a + i);
        ci = ai + bv;
        ci.storeu(c + i);
    }
    // finish non-multiple of 8 remainder
    for (; i < d; i++) {
        c[i] = a[i] + b;
    }
}

/***************************************************************************
 * PQ tables computations
 ***************************************************************************/

namespace {

// TODO dispatch to optimized code

/// compute the IP for dsub = 2 for 8 centroids and 4 sub-vectors at a time
template <bool is_inner_product, SIMDLevel SL>
void pq2_8cents_table(
        const simd8float32<SL> centroids[8],
        const simd8float32<SL> x,
        float* out,
        size_t ldo,
        size_t nout = 4) {
    using simd8float32 = simd8float32<SL>;

    simd8float32 ips[4];

    for (int i = 0; i < 4; i++) {
        simd8float32 p1, p2;
        if (is_inner_product) {
            p1 = x * centroids[2 * i];
            p2 = x * centroids[2 * i + 1];
        } else {
            p1 = (x - centroids[2 * i]);
            p1 = p1 * p1;
            p2 = (x - centroids[2 * i + 1]);
            p2 = p2 * p2;
        }
        ips[i] = hadd(p1, p2);
    }

    simd8float32 ip02a = geteven(ips[0], ips[1]);
    simd8float32 ip02b = geteven(ips[2], ips[3]);
    simd8float32 ip0 = getlow128(ip02a, ip02b);
    simd8float32 ip2 = gethigh128(ip02a, ip02b);

    simd8float32 ip13a = getodd(ips[0], ips[1]);
    simd8float32 ip13b = getodd(ips[2], ips[3]);
    simd8float32 ip1 = getlow128(ip13a, ip13b);
    simd8float32 ip3 = gethigh128(ip13a, ip13b);

    switch (nout) {
        case 4:
            ip3.storeu(out + 3 * ldo);
            [[fallthrough]];
        case 3:
            ip2.storeu(out + 2 * ldo);
            [[fallthrough]];
        case 2:
            ip1.storeu(out + 1 * ldo);
            [[fallthrough]];
        case 1:
            ip0.storeu(out);
    }
}

template <SIMDLevel SL>
simd8float32<SL> load_simd8float32_partial(const float* x, int n) {
    ALIGNED(32) float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float* wp = tmp;
    for (int i = 0; i < n; i++) {
        *wp++ = *x++;
    }
    return simd8float32<SL>(tmp);
}

} // anonymous namespace

// TODO dispatch to optimized code

void compute_PQ_dis_tables_dsub2(
        size_t d,
        size_t ksub,
        const float* all_centroids,
        size_t nx,
        const float* x,
        bool is_inner_product,
        float* dis_tables) {
    size_t M = d / 2;
    FAISS_THROW_IF_NOT(ksub % 8 == 0);
    using simd8float32 = simd8float32<SIMDLevel::NONE>;

    for (size_t m0 = 0; m0 < M; m0 += 4) {
        int m1 = std::min(M, m0 + 4);
        for (int k0 = 0; k0 < ksub; k0 += 8) {
            simd8float32 centroids[8];
            for (int k = 0; k < 8; k++) {
                ALIGNED(32) float centroid[8];
                size_t wp = 0;
                size_t rp = (m0 * ksub + k + k0) * 2;
                for (int m = m0; m < m1; m++) {
                    centroid[wp++] = all_centroids[rp];
                    centroid[wp++] = all_centroids[rp + 1];
                    rp += 2 * ksub;
                }
                centroids[k] = simd8float32(centroid);
            }
            for (size_t i = 0; i < nx; i++) {
                simd8float32 xi;
                if (m1 == m0 + 4) {
                    xi.loadu(x + i * d + m0 * 2);
                } else {
                    xi = load_simd8float32_partial<SIMDLevel::NONE>(
                            x + i * d + m0 * 2, 2 * (m1 - m0));
                }

                if (is_inner_product) {
                    pq2_8cents_table<true>(
                            centroids,
                            xi,
                            dis_tables + (i * M + m0) * ksub + k0,
                            ksub,
                            m1 - m0);
                } else {
                    pq2_8cents_table<false>(
                            centroids,
                            xi,
                            dis_tables + (i * M + m0) * ksub + k0,
                            ksub,
                            m1 - m0);
                }
            }
        }
    }
}

} // namespace faiss
