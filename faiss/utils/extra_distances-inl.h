/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** In this file are the implementations of extra metrics beyond L2
 *  and inner product */

#include <faiss/MetricType.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances.h>
#include <cmath>
#include <type_traits>

namespace faiss {

/***************************************************************************
 * VectorDistance base class - contains common data members and type defs
 **************************************************************************/

template <MetricType mt>
struct VectorDistanceBase {
    size_t d;
    float metric_arg;
    static constexpr MetricType metric = mt;
    static constexpr bool is_similarity = is_similarity_metric(mt);

    using C = typename std::conditional<
            is_similarity_metric(mt),
            CMin<float, int64_t>,
            CMax<float, int64_t>>::type;
};

/***************************************************************************
 * VectorDistance struct template - specializations for each metric type
 **************************************************************************/

template <MetricType mt, SIMDLevel level>
struct VectorDistance : VectorDistanceBase<mt> {
    inline float operator()(const float* x, const float* y) const;
};

template <SIMDLevel level>
struct VectorDistance<METRIC_L2, level> : VectorDistanceBase<METRIC_L2> {
    inline float operator()(const float* x, const float* y) const {
        return fvec_L2sqr<level>(x, y, this->d);
    }
};

template <SIMDLevel level>
struct VectorDistance<METRIC_INNER_PRODUCT, level>
        : VectorDistanceBase<METRIC_INNER_PRODUCT> {
    inline float operator()(const float* x, const float* y) const {
        return fvec_inner_product<level>(x, y, this->d);
    }
};

template <SIMDLevel level>
struct VectorDistance<METRIC_L1, level> : VectorDistanceBase<METRIC_L1> {
    inline float operator()(const float* x, const float* y) const {
        return fvec_L1<level>(x, y, this->d);
    }
};

template <SIMDLevel level>
struct VectorDistance<METRIC_Linf, level> : VectorDistanceBase<METRIC_Linf> {
    inline float operator()(const float* x, const float* y) const {
        return fvec_Linf<level>(x, y, this->d);
    }
};

template <>
struct VectorDistance<METRIC_Lp, SIMDLevel::NONE>
        : VectorDistanceBase<METRIC_Lp> {
    inline float operator()(const float* x, const float* y) const {
        float accu = 0;
        for (size_t i = 0; i < this->d; i++) {
            float diff = fabs(x[i] - y[i]);
            accu += powf(diff, this->metric_arg);
        }
        return accu;
    }
};

template <>
struct VectorDistance<METRIC_Canberra, SIMDLevel::NONE>
        : VectorDistanceBase<METRIC_Canberra> {
    inline float operator()(const float* x, const float* y) const {
        float accu = 0;
        for (size_t i = 0; i < this->d; i++) {
            float xi = x[i], yi = y[i];
            accu += fabs(xi - yi) / (fabs(xi) + fabs(yi));
        }
        return accu;
    }
};

template <>
struct VectorDistance<METRIC_BrayCurtis, SIMDLevel::NONE>
        : VectorDistanceBase<METRIC_BrayCurtis> {
    inline float operator()(const float* x, const float* y) const {
        float accu_num = 0, accu_den = 0;
        for (size_t i = 0; i < this->d; i++) {
            float xi = x[i], yi = y[i];
            accu_num += fabs(xi - yi);
            accu_den += fabs(xi + yi);
        }
        return accu_num / accu_den;
    }
};

template <>
struct VectorDistance<METRIC_JensenShannon, SIMDLevel::NONE>
        : VectorDistanceBase<METRIC_JensenShannon> {
    inline float operator()(const float* x, const float* y) const {
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
};

template <>
struct VectorDistance<METRIC_Jaccard, SIMDLevel::NONE>
        : VectorDistanceBase<METRIC_Jaccard> {
    inline float operator()(const float* x, const float* y) const {
        // WARNING: this distance is defined only for positive input vectors.
        // Providing vectors with negative values would lead to incorrect
        // results.
        float accu_num = 0, accu_den = 0;
        for (size_t i = 0; i < this->d; i++) {
            accu_num += fmin(x[i], y[i]);
            accu_den += fmax(x[i], y[i]);
        }
        return accu_num / accu_den;
    }
};

template <>
struct VectorDistance<METRIC_NaNEuclidean, SIMDLevel::NONE>
        : VectorDistanceBase<METRIC_NaNEuclidean> {
    inline float operator()(const float* x, const float* y) const {
        // https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html
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
};

template <>
struct VectorDistance<METRIC_GOWER, SIMDLevel::NONE>
        : VectorDistanceBase<METRIC_GOWER> {
    inline float operator()(const float* x, const float* y) const {
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
};

/***************************************************************************
 * Dispatching function that takes a metric type and a consumer object
 * the consumer object should contain a return type T and a operation template
 * function f() that is called to perform the operation.
 *
 * The first argument of the function is the VectorDistance object. The rest
 * are passed in as is. The object also dispatches to the current SIMD level.
 **************************************************************************/

template <class Consumer, class... Types>
typename Consumer::T dispatch_VectorDistance(
        size_t d,
        MetricType metric,
        float metric_arg,
        Consumer& consumer,
        Types... args) {
    auto dispatch_metric = [&]<MetricType mt>() {
        auto call = [&]<SIMDLevel level>() {
            VectorDistance<mt, level> vd = {d, metric_arg};
            return consumer.template f<VectorDistance<mt, level>>(vd, args...);
        };

        constexpr bool has_simd = mt == METRIC_INNER_PRODUCT ||
                mt == METRIC_L2 || mt == METRIC_L1 || mt == METRIC_Linf;
        if constexpr (!has_simd) {
            return call.template operator()<SIMDLevel::NONE>();
        } else {
            DISPATCH_SIMDLevel(call.template operator());
        }
    };
    return with_metric_type(metric, dispatch_metric);
}

} // namespace faiss
