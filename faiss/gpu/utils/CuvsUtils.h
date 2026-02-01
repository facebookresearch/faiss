// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/Tensor.cuh>

#include <cuvs/distance/distance.hpp>

#pragma GCC visibility push(default)
namespace faiss {
namespace gpu {

inline cuvs::distance::DistanceType metricFaissToCuvs(
        MetricType metric,
        bool exactDistance) {
    switch (metric) {
        case MetricType::METRIC_INNER_PRODUCT:
            return cuvs::distance::DistanceType::InnerProduct;
        case MetricType::METRIC_L2:
            return cuvs::distance::DistanceType::L2Expanded;
        case MetricType::METRIC_L1:
            return cuvs::distance::DistanceType::L1;
        case MetricType::METRIC_Linf:
            return cuvs::distance::DistanceType::Linf;
        case MetricType::METRIC_Lp:
            return cuvs::distance::DistanceType::LpUnexpanded;
        case MetricType::METRIC_Canberra:
            return cuvs::distance::DistanceType::Canberra;
        case MetricType::METRIC_BrayCurtis:
            return cuvs::distance::DistanceType::BrayCurtis;
        case MetricType::METRIC_JensenShannon:
            return cuvs::distance::DistanceType::JensenShannon;
        default:
            RAFT_FAIL("Distance type not supported");
    }
}

/// Identify matrix rows containing non NaN values. validRows[i] is false if row
/// i contains a NaN value and true otherwise.
void validRowIndices(
        GpuResources* res,
        Tensor<float, 2, true>& vecs,
        bool* validRows);

/// Filter out matrix rows containing NaN values. The vectors and indices are
/// updated in-place.
idx_t inplaceGatherFilteredRows(
        GpuResources* res,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices);
} // namespace gpu
} // namespace faiss
#pragma GCC visibility pop
