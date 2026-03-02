/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/distances.h>

namespace faiss {

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_norm_L2sqr<AUTOVEC_LEVEL>(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_L2sqr<AUTOVEC_LEVEL>(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_inner_product<AUTOVEC_LEVEL>(
        const float* x,
        const float* y,
        size_t d) {
    float res = 0.F;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * y[i];
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_L1<AUTOVEC_LEVEL>(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += fabs(tmp);
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_Linf<AUTOVEC_LEVEL>(const float* x, const float* y, size_t d) {
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        res = fmax(res, fabs(x[i] - y[i]));
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
void fvec_inner_product_batch_4<AUTOVEC_LEVEL>(
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
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        d0 += x[i] * y0[i];
        d1 += x[i] * y1[i];
        d2 += x[i] * y2[i];
        d3 += x[i] * y3[i];
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
void fvec_L2sqr_batch_4<AUTOVEC_LEVEL>(
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
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

} // namespace faiss
