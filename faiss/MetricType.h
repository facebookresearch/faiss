/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_METRIC_TYPE_H
#define FAISS_METRIC_TYPE_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace faiss {

/// The metric space for vector comparison for Faiss indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
enum MetricType {
    METRIC_INNER_PRODUCT = 0, ///< maximum inner product search
    METRIC_L2 = 1,            ///< squared L2 search
    METRIC_L1,                ///< L1 (aka cityblock)
    METRIC_Linf,              ///< infinity distance
    METRIC_Lp,                ///< L_p distance, p is given by a faiss::Index
                              /// metric_arg

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,

    /// sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i)) where a_i, b_i > 0
    METRIC_Jaccard,
    /// Squared Euclidean distance, ignoring NaNs
    METRIC_NaNEuclidean,
    /// Gower's distance - numeric dimensions are in [0,1] and categorical
    /// dimensions are negative integers
    METRIC_GOWER,
};

/// all vector indices are this type
using idx_t = int64_t;

/// this function is used to distinguish between min and max indexes since
/// we need to support similarity and dis-similarity metrics in a flexible way
constexpr bool is_similarity_metric(MetricType metric_type) {
    return ((metric_type == METRIC_INNER_PRODUCT) ||
            (metric_type == METRIC_Jaccard));
}

/// Dispatch to a lambda with MetricType as a compile-time constant.
/// This allows writing generic code that works with different metrics
/// while maintaining compile-time optimization.
///
/// There are better ways to do this in C++20, but this is a simple way
/// to gets the job done for C++17.
///
/// Example usage:
///   auto result = with_metric_type(runtime_metric, [&](auto metric_tag) {
///       constexpr MetricType M = decltype(metric_tag)::value;
///       return compute_distance<M>(x, y);
///   });
#ifndef SWIG
template <MetricType M>
struct metric_type_constant {
    static constexpr MetricType value = M;
};

template <typename LambdaType>
inline auto with_metric_type(MetricType metric, LambdaType&& action) {
    switch (metric) {
        case METRIC_INNER_PRODUCT:
            return action(metric_type_constant<METRIC_INNER_PRODUCT>{});
        case METRIC_L2:
            return action(metric_type_constant<METRIC_L2>{});
        case METRIC_L1:
            return action(metric_type_constant<METRIC_L1>{});
        case METRIC_Linf:
            return action(metric_type_constant<METRIC_Linf>{});
        case METRIC_Lp:
            return action(metric_type_constant<METRIC_Lp>{});
        case METRIC_Canberra:
            return action(metric_type_constant<METRIC_Canberra>{});
        case METRIC_BrayCurtis:
            return action(metric_type_constant<METRIC_BrayCurtis>{});
        case METRIC_JensenShannon:
            return action(metric_type_constant<METRIC_JensenShannon>{});
        case METRIC_Jaccard:
            return action(metric_type_constant<METRIC_Jaccard>{});
        case METRIC_NaNEuclidean:
            return action(metric_type_constant<METRIC_NaNEuclidean>{});
        case METRIC_GOWER:
            return action(metric_type_constant<METRIC_GOWER>{});
        default: {
            fprintf(stderr,
                    "FATAL ERROR: with_metric_type called with unknown "
                    "metric %d\n",
                    static_cast<int>(metric));
            abort();
        }
    }
}
#endif // SWIG

} // namespace faiss

#endif
