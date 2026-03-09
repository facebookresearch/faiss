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
#include <faiss/impl/simdlib/simdlib_dispatch.h>

#define AUTOVEC_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/distances_autovec-inl.h>

#define THE_SIMDLEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/distances_simdlib256.h>

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
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
    }
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

} // namespace faiss
