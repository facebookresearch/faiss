/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL not defined"
#endif

namespace faiss {

constexpr faiss::SIMDLevel SL = THE_SIMD_LEVEL;
/******************************************************************
 * These functions are simple enough that the compile will do a good job
 * vectorizing them given the appropriate flags.
 ******************************************************************/

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_norm_L2sqr<SL>(const float* x, size_t d) {
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
float fvec_L2sqr<SL>(const float* x, const float* y, size_t d) {
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
float fvec_inner_product<SL>(const float* x, const float* y, size_t d) {
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
float fvec_L1<SL>(const float* x, const float* y, size_t d) {
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
float fvec_Linf<SL>(const float* x, const float* y, size_t d) {
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
void fvec_inner_product_batch_4<SL>(
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
void fvec_L2sqr_batch_4<SL>(
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

/******************************************************************
 * VectorDistance::operator() specializations — defined out-of-class
 * so that SIMD compilation units produce externally-linkable symbols.
 ******************************************************************/

template <>
float VectorDistance<METRIC_L2, SL>::operator()(const float* x, const float* y)
        const {
    return fvec_L2sqr<SL>(x, y, this->d);
}

template <>
float VectorDistance<METRIC_INNER_PRODUCT, SL>::operator()(
        const float* x,
        const float* y) const {
    return fvec_inner_product<SL>(x, y, this->d);
}

template <>
float VectorDistance<METRIC_L1, SL>::operator()(const float* x, const float* y)
        const {
    return fvec_L1<SL>(x, y, this->d);
}

template <>
float VectorDistance<METRIC_Linf, SL>::operator()(
        const float* x,
        const float* y) const {
    return fvec_Linf<SL>(x, y, this->d);
}

template <>
float VectorDistance<METRIC_Lp, SL>::operator()(const float* x, const float* y)
        const {
    float accu = 0;
    for (size_t i = 0; i < this->d; i++) {
        float diff = fabs(x[i] - y[i]);
        accu += powf(diff, this->metric_arg);
    }
    return accu;
}

template <>
float VectorDistance<METRIC_Canberra, SL>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < this->d; i++) {
        float xi = x[i], yi = y[i];
        accu += fabs(xi - yi) / (fabs(xi) + fabs(yi));
    }
    return accu;
}

template <>
float VectorDistance<METRIC_BrayCurtis, SL>::operator()(
        const float* x,
        const float* y) const {
    float accu_num = 0, accu_den = 0;
    for (size_t i = 0; i < this->d; i++) {
        float xi = x[i], yi = y[i];
        accu_num += fabs(xi - yi);
        accu_den += fabs(xi + yi);
    }
    return accu_num / accu_den;
}

template <>
float VectorDistance<METRIC_JensenShannon, SL>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < this->d; i++) {
        float xi = x[i], yi = y[i];
        float mi = 0.5 * (xi + yi);
        float kl1 = -xi * log(mi / xi);
        float kl2 = -yi * log(mi / yi);
        accu += kl1 + kl2;
    }
    return 0.5 * accu;
}

template <>
float VectorDistance<METRIC_Jaccard, SL>::operator()(
        const float* x,
        const float* y) const {
    float accu_num = 0, accu_den = 0;
    for (size_t i = 0; i < this->d; i++) {
        accu_num += fmin(x[i], y[i]);
        accu_den += fmax(x[i], y[i]);
    }
    return accu_num / accu_den;
}

template <>
float VectorDistance<METRIC_NaNEuclidean, SL>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    size_t present = 0;
    for (size_t i = 0; i < this->d; i++) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            float diff = x[i] - y[i];
            accu += diff * diff;
            present++;
        }
    }
    if (present == 0) {
        return NAN;
    }
    return float(this->d) / float(present) * accu;
}

template <>
float VectorDistance<METRIC_GOWER, SL>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    size_t valid_dims = 0;

    for (size_t i = 0; i < this->d; i++) {
        if (std::isnan(x[i]) || std::isnan(y[i])) {
            continue;
        }

        if (x[i] >= 0 && y[i] >= 0) {
            if (x[i] > 1 || y[i] > 1) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            accu += fabs(x[i] - y[i]);
        } else if (x[i] < 0 && y[i] < 0) {
            accu += float(int(x[i] != y[i]));
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
        valid_dims++;
    }

    if (valid_dims == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return accu / valid_dims;
}

} // namespace faiss
