// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <faiss/gpu/impl/CuvsIVFRaBitQ.cuh>

#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>

namespace faiss {
namespace gpu {

namespace {

cuvs::neighbors::ivf_rabitq::search_mode toCuvsSearchMode(
        IVFRaBitQSearchMode mode) {
    using CuvsMode = cuvs::neighbors::ivf_rabitq::search_mode;
    switch (mode) {
        case IVFRaBitQSearchMode::LUT16:
            return CuvsMode::LUT16;
        case IVFRaBitQSearchMode::LUT32:
            return CuvsMode::LUT32;
        case IVFRaBitQSearchMode::QUANT4:
            return CuvsMode::QUANT4;
        case IVFRaBitQSearchMode::QUANT8:
            return CuvsMode::QUANT8;
    }
    FAISS_THROW_MSG("invalid IVF-RaBitQ search mode");
}

} // namespace

CuvsIVFRaBitQ::CuvsIVFRaBitQ(
        GpuResources* resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        const GpuIndexIVFRaBitQConfig& config)
        : resources_(resources),
          dim_(dim),
          nlist_(nlist),
          metric_(metric),
          config_(config) {}

void CuvsIVFRaBitQ::train(idx_t n, const float* x) {
    const auto& raft_handle = resources_->getRaftHandleCurrentDevice();

    cuvs::neighbors::ivf_rabitq::index_params params;
    params.metric = metricFaissToCuvs(metric_, false);
    params.n_lists = static_cast<uint32_t>(nlist_);
    params.bits_per_dim = config_.bitsPerDim;
    params.kmeans_n_iters = config_.kmeansNIterations;
    params.max_train_points_per_cluster = config_.maxTrainPointsPerCluster;
    params.fast_quantize_flag = config_.useFastQuantize;
    params.streaming_batch_size = config_.streamingBatchSize;
    params.force_streaming = config_.forceStreaming;

    if (getDeviceForAddress(x) >= 0) {
        auto dataset =
                raft::make_device_matrix_view<const float, int64_t>(x, n, dim_);
        cuvs_index_ =
                std::make_shared<cuvs::neighbors::ivf_rabitq::index<int64_t>>(
                        cuvs::neighbors::ivf_rabitq::build(
                                raft_handle, params, dataset));
    } else {
        auto dataset =
                raft::make_host_matrix_view<const float, int64_t>(x, n, dim_);
        cuvs_index_ =
                std::make_shared<cuvs::neighbors::ivf_rabitq::index<int64_t>>(
                        cuvs::neighbors::ivf_rabitq::build(
                                raft_handle, params, dataset));
    }
}

void CuvsIVFRaBitQ::search(
        Tensor<float, 2, true>& queries,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        uint32_t nprobe,
        IVFRaBitQSearchMode searchMode) {
    FAISS_ASSERT(cuvs_index_);
    const auto& raft_handle = resources_->getRaftHandleCurrentDevice();
    const auto n = queries.getSize(0);

    auto queriesView = raft::make_device_matrix_view<const float, int64_t>(
            queries.data(), n, dim_);
    auto distancesView = raft::make_device_matrix_view<float, int64_t>(
            outDistances.data(), n, k);
    auto indicesView = raft::make_device_matrix_view<int64_t, int64_t>(
            outIndices.data(), n, k);

    cuvs::neighbors::ivf_rabitq::search_params params;
    params.n_probes = nprobe;
    params.mode = toCuvsSearchMode(searchMode);
    cuvs::neighbors::ivf_rabitq::search(
            raft_handle,
            params,
            *cuvs_index_,
            queriesView,
            indicesView,
            distancesView);
}

void CuvsIVFRaBitQ::reset() {
    cuvs_index_.reset();
}

} // namespace gpu
} // namespace faiss
