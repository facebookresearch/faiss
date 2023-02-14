/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <faiss/gpu/impl/RaftFlatIndex.cuh>
#include <vector>

#include <faiss/gpu/impl/RaftUtils.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>

namespace faiss {
namespace gpu {

using namespace raft::distance;
using namespace raft::neighbors;

RaftFlatIndex::RaftFlatIndex(
        GpuResources* res,
        int dim,
        bool useFloat16,
        MemorySpace space)
        : FlatIndex(res, dim, useFloat16, space) {}

void RaftFlatIndex::query(
        Tensor<float, 2, true>& input,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool exactDistance) {
    raft::handle_t& handle = resources_->getRaftHandleCurrentDevice();

    auto index = raft::make_device_matrix_view<const float, idx_t>(
            vectors_.data(), vectors_.getSize(0), vectors_.getSize(1));
    auto search = raft::make_device_matrix_view<const float, idx_t>(
            input.data(), input.getSize(0), input.getSize(1));
    auto inds = raft::make_device_matrix_view<idx_t, idx_t>(
            outIndices.data(), outIndices.getSize(0), outIndices.getSize(1));
    auto dists = raft::make_device_matrix_view<float, idx_t>(
            outDistances.data(),
            outDistances.getSize(0),
            outDistances.getSize(1));

    DistanceType distance = faiss_to_raft(metric, exactDistance);

    std::vector<raft::device_matrix_view<const float, idx_t>> index_vec = {index};

    // For now, use RAFT's fused KNN when k <= 64 and L2 metric is used
    if (k <= 64 && metric == MetricType::METRIC_L2 && vectors_.getSize(0) > 0) {
        brute_force::fused_l2_knn(handle, index, search, inds, dists, distance);
    } else {
        brute_force::knn(handle, index_vec, search, inds, dists, k, distance);
    }
}
} // namespace gpu
} // namespace faiss
