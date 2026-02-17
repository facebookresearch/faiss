/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** In this file are the implementations of extra metrics beyond L2
 *  and inner product */

#include <stdint.h>

#include <faiss/Index.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/utils/Heap.h>

namespace faiss {

struct FlatCodesDistanceComputer;

void pairwise_extra_distances(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        MetricType mt,
        float metric_arg,
        float* dis,
        int64_t ldq = -1,
        int64_t ldb = -1,
        int64_t ldd = -1);

void knn_extra_metrics(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        MetricType mt,
        float metric_arg,
        size_t k,
        float* distances,
        int64_t* indexes,
        const IDSelector* sel = nullptr);

/** get a DistanceComputer that refers to this type of distance and
 *  indexes a flat array of size nb */
FlatCodesDistanceComputer* get_extra_distance_computer(
        size_t d,
        MetricType mt,
        float metric_arg,
        size_t nb,
        const float* xb);

/// Dispatch to a lambda with MetricType as a compile-time constant.
/// This allows writing generic code that works with different metrics
/// while maintaining compile-time optimization.
///
/// Example usage:
///   auto result = with_metric_type(runtime_metric, [&](auto metric_tag) {
///       constexpr MetricType M = decltype(metric_tag)::value;
///       return compute_distance<M>(x, y);
///   });
#ifndef SWIG

template <typename LambdaType>
inline auto with_metric_type(MetricType metric, LambdaType&& action) {
    switch (metric) {
        case METRIC_INNER_PRODUCT:
            return action.template operator()<METRIC_INNER_PRODUCT>();
        case METRIC_L2:
            return action.template operator()<METRIC_L2>();
        case METRIC_L1:
            return action.template operator()<METRIC_L1>();
        case METRIC_Linf:
            return action.template operator()<METRIC_Linf>();
        case METRIC_Lp:
            return action.template operator()<METRIC_Lp>();
        case METRIC_Canberra:
            return action.template operator()<METRIC_Canberra>();
        case METRIC_BrayCurtis:
            return action.template operator()<METRIC_BrayCurtis>();
        case METRIC_JensenShannon:
            return action.template operator()<METRIC_JensenShannon>();
        case METRIC_Jaccard:
            return action.template operator()<METRIC_Jaccard>();
        case METRIC_NaNEuclidean:
            return action.template operator()<METRIC_NaNEuclidean>();
        case METRIC_GOWER:
            return action.template operator()<METRIC_GOWER>();
        default:
            FAISS_THROW_FMT(
                    "with_metric_type called with unknown metric %d",
                    int(metric));
    }
}
#endif // SWIG

} // namespace faiss

#include <faiss/utils/extra_distances-inl.h>
