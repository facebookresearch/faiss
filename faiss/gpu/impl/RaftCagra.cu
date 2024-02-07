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

#include <faiss/gpu/utils/DeviceUtils.h>
#include <cstddef>
#include <faiss/gpu/impl/RaftCagra.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/neighbors/cagra.cuh>

namespace faiss {
namespace gpu {

RaftCagra::RaftCagra(
        GpuResources* resources,
        int dim,
        idx_t intermediate_graph_degree,
        idx_t graph_degree,
        faiss::cagra_build_algo graph_build_algo,
        size_t nn_descent_niter,
        faiss::MetricType metric,
        float metricArg,
        IndicesOptions indicesOptions)
        : resources_(resources),
          dim_(dim),
          metric_(metric),
          metricArg_(metricArg),
          index_pams_() {
    FAISS_THROW_IF_NOT_MSG(
            metric == faiss::METRIC_L2,
            "CAGRA currently only supports L2 metric.");
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for RAFT CAGRA index");

    index_pams_.intermediate_graph_degree = intermediate_graph_degree;
    index_pams_.graph_degree = graph_degree;
    index_pams_.build_algo =
            static_cast<raft::neighbors::cagra::graph_build_algo>(
                    graph_build_algo);
    index_pams_.nn_descent_niter = nn_descent_niter;

    reset();
}

void RaftCagra::train(idx_t n, const float* x) {
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    if (getDeviceForAddress(x) >= 0) {
        raft_knn_index = raft::neighbors::cagra::build<float, idx_t>(
                raft_handle,
                index_pams_,
                raft::make_device_matrix_view<const float, idx_t>(x, n, dim_));
    } else {
        raft_knn_index = raft::neighbors::cagra::build<float, idx_t>(
                raft_handle,
                index_pams_,
                raft::make_host_matrix_view<const float, idx_t>(x, n, dim_));
    }
}

void RaftCagra::search(
        Tensor<float, 2, true>& queries,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        idx_t max_queries,
        idx_t itopk_size,
        idx_t max_iterations,
        faiss::cagra_search_algo graph_search_algo,
        idx_t team_size,
        idx_t search_width,
        idx_t min_iterations,
        idx_t thread_block_size,
        faiss::cagra_hash_mode hash_mode,
        idx_t hashmap_min_bitlen,
        float hashmap_max_fill_rate,
        idx_t num_random_samplings,
        idx_t rand_xor_mask) {
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    idx_t numQueries = queries.getSize(0);
    idx_t cols = queries.getSize(1);
    idx_t k_ = k;

    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);

    auto queries_view = raft::make_device_matrix_view<const float, idx_t>(
            queries.data(), numQueries, cols);
    auto distances_view = raft::make_device_matrix_view<float, idx_t>(
            outDistances.data(), numQueries, k_);
    auto indices_view = raft::make_device_matrix_view<idx_t, idx_t>(
            outIndices.data(), numQueries, k_);

    raft::neighbors::cagra::search_params search_pams;
    search_pams.max_queries = max_queries;
    search_pams.itopk_size = itopk_size;
    search_pams.max_iterations = max_iterations;
    search_pams.algo =
            static_cast<raft::neighbors::cagra::search_algo>(graph_search_algo);
    search_pams.team_size = team_size;
    search_pams.search_width = search_width;
    search_pams.min_iterations = min_iterations;
    search_pams.thread_block_size = thread_block_size;
    search_pams.hashmap_mode =
            static_cast<raft::neighbors::cagra::hash_mode>(hash_mode);
    search_pams.hashmap_min_bitlen = hashmap_min_bitlen;
    search_pams.hashmap_max_fill_rate = hashmap_max_fill_rate;
    search_pams.num_random_samplings = num_random_samplings;
    search_pams.rand_xor_mask = rand_xor_mask;

    raft::neighbors::cagra::search(
            raft_handle,
            search_pams,
            raft_knn_index.value(),
            queries_view,
            indices_view,
            distances_view);
}

void RaftCagra::reset() {
    raft_knn_index.reset();
}

} // namespace gpu
} // namespace faiss
