/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/distances.h>

namespace faiss {

template <>
void fvec_sub<THE_SIMDLEVEL>(
        size_t d,
        const float* a,
        const float* b,
        float* c) {
    size_t i;
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32_tpl<THE_SIMDLEVEL> ci, ai, bi;
        ai.loadu(a + i);
        bi.loadu(b + i);
        ci = ai - bi;
        ci.storeu(c + i);
    }
    for (; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

template <>
void fvec_add<THE_SIMDLEVEL>(
        size_t d,
        const float* a,
        const float* b,
        float* c) {
    size_t i;
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32_tpl<THE_SIMDLEVEL> ci, ai, bi;
        ai.loadu(a + i);
        bi.loadu(b + i);
        ci = ai + bi;
        ci.storeu(c + i);
    }
    for (; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

template <>
void fvec_add<THE_SIMDLEVEL>(size_t d, const float* a, float b, float* c) {
    size_t i;
    simd8float32_tpl<THE_SIMDLEVEL> bv(b);
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32_tpl<THE_SIMDLEVEL> ci, ai;
        ai.loadu(a + i);
        ci = ai + bv;
        ci.storeu(c + i);
    }
    for (; i < d; i++) {
        c[i] = a[i] + b;
    }
}

/***************************************************************************
 * PQ tables computations
 ***************************************************************************/

namespace {

/// compute the IP for dsub = 2 for 8 centroids and 4 sub-vectors at a time
template <SIMDLevel SL, bool is_inner_product>
void pq2_8cents_table(
        const simd8float32_tpl<SL> centroids[8],
        const simd8float32_tpl<SL> x,
        float* out,
        size_t ldo,
        size_t nout = 4) {
    simd8float32_tpl<SL> ips[4];

    for (int i = 0; i < 4; i++) {
        simd8float32_tpl<SL> p1, p2;
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

    simd8float32_tpl<SL> ip02a = geteven(ips[0], ips[1]);
    simd8float32_tpl<SL> ip02b = geteven(ips[2], ips[3]);
    simd8float32_tpl<SL> ip0 = getlow128(ip02a, ip02b);
    simd8float32_tpl<SL> ip2 = gethigh128(ip02a, ip02b);

    simd8float32_tpl<SL> ip13a = getodd(ips[0], ips[1]);
    simd8float32_tpl<SL> ip13b = getodd(ips[2], ips[3]);
    simd8float32_tpl<SL> ip1 = getlow128(ip13a, ip13b);
    simd8float32_tpl<SL> ip3 = gethigh128(ip13a, ip13b);

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
simd8float32_tpl<SL> load_simd8float32_partial(const float* x, int n) {
    ALIGNED(32) float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float* wp = tmp;
    for (int i = 0; i < n; i++) {
        *wp++ = *x++;
    }
    return simd8float32_tpl<SL>(tmp);
}

} // anonymous namespace

template <>
void compute_PQ_dis_tables_dsub2<THE_SIMDLEVEL>(
        size_t d,
        size_t ksub,
        const float* all_centroids,
        size_t nx,
        const float* x,
        bool is_inner_product,
        float* dis_tables) {
    size_t M = d / 2;
    FAISS_THROW_IF_NOT(ksub % 8 == 0);

    for (size_t m0 = 0; m0 < M; m0 += 4) {
        int m1 = std::min(M, m0 + 4);
        for (int k0 = 0; k0 < ksub; k0 += 8) {
            simd8float32_tpl<THE_SIMDLEVEL> centroids[8];
            for (int k = 0; k < 8; k++) {
                ALIGNED(32) float centroid[8];
                size_t wp = 0;
                size_t rp = (m0 * ksub + k + k0) * 2;
                for (int m = m0; m < m1; m++) {
                    centroid[wp++] = all_centroids[rp];
                    centroid[wp++] = all_centroids[rp + 1];
                    rp += 2 * ksub;
                }
                centroids[k] = simd8float32_tpl<THE_SIMDLEVEL>(centroid);
            }
            for (size_t i = 0; i < nx; i++) {
                simd8float32_tpl<THE_SIMDLEVEL> xi;
                if (m1 == m0 + 4) {
                    xi.loadu(x + i * d + m0 * 2);
                } else {
                    xi = load_simd8float32_partial<THE_SIMDLEVEL>(
                            x + i * d + m0 * 2, 2 * (m1 - m0));
                }

                if (is_inner_product) {
                    pq2_8cents_table<THE_SIMDLEVEL, true>(
                            centroids,
                            xi,
                            dis_tables + (i * M + m0) * ksub + k0,
                            ksub,
                            m1 - m0);
                } else {
                    pq2_8cents_table<THE_SIMDLEVEL, false>(
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
