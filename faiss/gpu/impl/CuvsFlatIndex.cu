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

#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsFlatIndex.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>

#include <optional>
#include <vector>

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>

namespace faiss {
namespace gpu {

using namespace cuvs::distance;
using namespace cuvs::neighbors;

CuvsFlatIndex::CuvsFlatIndex(
        GpuResources* res,
        int dim,
        bool useFloat16,
        MemorySpace space)
        : FlatIndex(res, dim, useFloat16, space) {}

void CuvsFlatIndex::query(
        Tensor<float, 2, true>& input,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool exactDistance) {
    /**
     * cuVS doesn't yet support half-precision in bfknn.
     * Use FlatIndex for float16 for now
     */
    if (useFloat16_) {
        auto stream = resources_->getDefaultStreamCurrentDevice();

        // We need to convert the input to float16 for comparison to ourselves
        auto inputHalf = convertTensorTemporary<float, half, 2>(
                resources_, stream, input);

        FlatIndex::query(
                inputHalf,
                k,
                metric,
                metricArg,
                outDistances,
                outIndices,
                exactDistance);
    } else {
        raft::device_resources& handle =
                resources_->getRaftHandleCurrentDevice();

        auto index = raft::make_device_matrix_view<const float, int64_t>(
                vectors_.data(), vectors_.getSize(0), vectors_.getSize(1));
        auto search = raft::make_device_matrix_view<const float, int64_t>(
                input.data(), input.getSize(0), input.getSize(1));

        auto inds = raft::make_device_matrix_view<idx_t, int64_t>(
                outIndices.data(),
                outIndices.getSize(0),
                outIndices.getSize(1));
        auto dists = raft::make_device_matrix_view<float, int64_t>(
                outDistances.data(),
                outDistances.getSize(0),
                outDistances.getSize(1));

        cuvsDistanceType distance = metricFaissToCuvs(metric, exactDistance);

        std::optional<raft::device_vector_view<const float, int64_t>>
                norms_view = raft::make_device_vector_view(
                        norms_.data(), norms_.getSize(0));

        cuvs::neighbors::brute_force::index idx(
                handle, index, norms_view, distance, metricArg);
        cuvs::neighbors::brute_force::search(
                handle, idx, search, inds, dists, std::nullopt);

        if (metric == MetricType::METRIC_Lp) {
            raft::linalg::unary_op(
                    handle,
                    raft::make_const_mdspan(dists),
                    dists,
                    [metricArg] __device__(const float& a) {
                        return powf(a, metricArg);
                    });
        } else if (metric == MetricType::METRIC_JensenShannon) {
            raft::linalg::unary_op(
                    handle,
                    raft::make_const_mdspan(dists),
                    dists,
                    [] __device__(const float& a) { return powf(a, 2); });
        }
    }
}

void CuvsFlatIndex::query(
        Tensor<half, 2, true>& vecs,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool exactDistance) {
    FAISS_ASSERT(useFloat16_);

    // FIXME: ref https://github.com/rapidsai/raft/issues/1280
    FlatIndex::query(
            vecs,
            k,
            metric,
            metricArg,
            outDistances,
            outIndices,
            exactDistance);
}

} // namespace gpu
} // namespace faiss
