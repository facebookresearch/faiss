/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/** In this file are the implementations of extra metrics beyond L2
 *  and inner product */

#include <faiss/utils/distances.h>
#include <type_traits>

namespace faiss {

template <MetricType mt>
struct VectorDistance {
    size_t d;
    float metric_arg;

    inline float operator()(const float* x, const float* y) const;

    // heap template to use for this type of metric
    using C = typename std::conditional<
            mt == METRIC_INNER_PRODUCT,
            CMin<float, int64_t>,
            CMax<float, int64_t>>::type;
};

template <>
inline float VectorDistance<METRIC_L2>::operator()(
        const float* x,
        const float* y) const {
    return fvec_L2sqr(x, y, d);
}

template <>
inline float VectorDistance<METRIC_INNER_PRODUCT>::operator()(
        const float* x,
        const float* y) const {
    return fvec_inner_product(x, y, d);
}

template <>
inline float VectorDistance<METRIC_L1>::operator()(
        const float* x,
        const float* y) const {
    return fvec_L1(x, y, d);
}

template <>
inline float VectorDistance<METRIC_Linf>::operator()(
        const float* x,
        const float* y) const {
    return fvec_Linf(x, y, d);
    /*
        float vmax = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = fabs (x[i] - y[i]);
            if (diff > vmax) vmax = diff;
        }
     return vmax;*/
}

template <>
inline float VectorDistance<METRIC_Lp>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < d; i++) {
        float diff = fabs(x[i] - y[i]);
        accu += powf(diff, metric_arg);
    }
    return accu;
}

template <>
inline float VectorDistance<METRIC_Canberra>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < d; i++) {
        float xi = x[i], yi = y[i];
        accu += fabs(xi - yi) / (fabs(xi) + fabs(yi));
    }
    return accu;
}

template <>
inline float VectorDistance<METRIC_BrayCurtis>::operator()(
        const float* x,
        const float* y) const {
    float accu_num = 0, accu_den = 0;
    for (size_t i = 0; i < d; i++) {
        float xi = x[i], yi = y[i];
        accu_num += fabs(xi - yi);
        accu_den += fabs(xi + yi);
    }
    return accu_num / accu_den;
}

template <>
inline float VectorDistance<METRIC_JensenShannon>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < d; i++) {
        float xi = x[i], yi = y[i];
        float mi = 0.5 * (xi + yi);
        float kl1 = -xi * log(mi / xi);
        float kl2 = -yi * log(mi / yi);
        accu += kl1 + kl2;
    }
    return 0.5 * accu;
}

} // namespace faiss
