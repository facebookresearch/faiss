// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cuvs/core/dataset.h>
#include <cuvs/neighbors/cagra.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>

#include <cstddef>

namespace faiss {

/// Algorithm used to build underlying CAGRA graph
enum class cagra_build_algo { IVF_PQ, NN_DESCENT };

enum class cagra_search_algo {
    SINGLE_CTA = 0,
    MULTI_CTA = 1,
    MULTI_KERNEL = 2,
    AUTO = 100
};

enum class cagra_hash_mode { HASH = 0, SMALL = 1, AUTO = 100 };

namespace gpu {

struct IVFPQBuildCagraConfig;
struct IVFPQSearchCagraConfig;

template <typename data_t = float>
class CuvsCagra {
   public:
    CuvsCagra(
            GpuResources* resources,
            int dim,
            idx_t intermediate_graph_degree,
            idx_t graph_degree,
            faiss::cagra_build_algo graph_build_algo,
            size_t nn_descent_niter,
            bool store_dataset,
            faiss::MetricType metric,
            float metricArg,
            IndicesOptions indicesOptions,
            const IVFPQBuildCagraConfig* ivf_pq_params = nullptr,
            const IVFPQSearchCagraConfig* ivf_pq_search_params = nullptr,
            float refine_rate = 2.0f,
            bool guarantee_connectivity = false);

    CuvsCagra(
            GpuResources* resources,
            int dim,
            idx_t n,
            int graph_degree,
            const data_t* dataset,
            const idx_t* knn_graph,
            faiss::MetricType metric,
            float metricArg,
            IndicesOptions indicesOptions);

    ~CuvsCagra();

    void train(idx_t n, const data_t* x);

    void search(
            Tensor<data_t, 2, true>& queries,
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
            idx_t rand_xor_mask,
            const IDSelector* sel = nullptr);

    void reset();

    idx_t get_knngraph_degree() const;

    std::vector<idx_t> get_knngraph() const;

    const data_t* get_training_dataset() const;

   private:
    /// Collection of GPU resources that we use
    GpuResources* resources_;

    /// Training dataset
    const data_t* storage_;
    int n_;

    /// Expected dimensionality of the vectors
    const int dim_;

    /// Controls the underlying cuVS index if it should store the dataset in
    /// device memory. Default set to true for enabling search capabilities on
    /// the index.
    /// NB: This is also required to be set to true for deserializing
    /// an IndexHNSWCagra object.
    bool store_dataset_ = true;

    /// Metric type of the index
    faiss::MetricType metric_;

    /// Metric arg
    float metricArg_;

    /// Parameters to build cuVS CAGRA index
    faiss::cagra_build_algo graph_build_algo_;
    size_t intermediate_graph_degree_;
    size_t graph_degree_;

    /// Parameters to build CAGRA graph using IVF PQ
    const IVFPQBuildCagraConfig* ivf_pq_params_;
    const IVFPQSearchCagraConfig* ivf_pq_search_params_;
    float refine_rate_;

    /// Parameters to build CAGRA graph using NN Descent
    size_t nn_descent_niter_ = 20;

    /// release/26.10 has no C parameter for guarantee_connectivity and its
    /// CAGRA bridge omits IVF-PQ max_internal_batch_size. train() uses a
    /// narrow C++ graph-build fallback when either setting is requested.
    bool guarantee_connectivity_ = false;

    /// Device storage used when cuvsCagraIndexFromArgs receives host data.
    DeviceTensor<data_t, 2, true> ownedDataset_;

    /// Dataset object whose storage/view is referenced by the CAGRA index.
    cuvsDataset_t cuvs_dataset_{nullptr};

    /// Instance of trained cuVS CAGRA index
    cuvsCagraIndex_t cuvs_index{nullptr};
};
} // namespace gpu
} // namespace faiss
