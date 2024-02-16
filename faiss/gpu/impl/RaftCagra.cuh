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

#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuResources.h>
#include <cstddef>
#include <faiss/gpu/utils/Tensor.cuh>

#include <faiss/MetricType.h>

#include <raft/neighbors/cagra_types.hpp>

namespace faiss {

enum class cagra_build_algo { IVF_PQ, NN_DESCENT };

enum class cagra_search_algo { SINGLE_CTA, MULTI_CTA };

enum class cagra_hash_mode { HASH, SMALL, AUTO };

namespace gpu {

class RaftCagra {
   public:
    RaftCagra(
            GpuResources* resources,
            int dim,
            idx_t intermediate_graph_degree,
            idx_t graph_degree,
            faiss::cagra_build_algo graph_build_algo,
            size_t nn_descent_niter,
            faiss::MetricType metric,
            float metricArg,
            IndicesOptions indicesOptions);

    ~RaftCagra() = default;

    void train(idx_t n, const float* x);

    void search(
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
            idx_t rand_xor_mask);

    void reset();

   private:
    /// Collection of GPU resources that we use
    GpuResources* resources_;

    /// Expected dimensionality of the vectors
    const int dim_;

    /// Metric type of the index
    faiss::MetricType metric_;

    /// Metric arg
    float metricArg_;

    /// Parameters to build RAFT CAGRA index
    raft::neighbors::cagra::index_params index_pams_;

    /// Instance of trained RAFT CAGRA index
    std::optional<raft::neighbors::cagra::index<float, idx_t>> raft_knn_index{
            std::nullopt};
};

} // namespace gpu
} // namespace faiss
