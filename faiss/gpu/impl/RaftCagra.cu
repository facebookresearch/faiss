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

#include <faiss/gpu/utils/DeviceUtils.h>
#include <cstddef>
#include <cstdint>
#include <faiss/gpu/impl/RaftCagra.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft_runtime/neighbors/cagra.hpp>
#include <optional>
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
        IndicesOptions indicesOptions,
        std::optional<raft::neighbors::ivf_pq::index_params> ivf_pq_params,
        std::optional<raft::neighbors::ivf_pq::search_params>
                ivf_pq_search_params)
        : resources_(resources),
          dim_(dim),
          metric_(metric),
          metricArg_(metricArg),
          index_params_(),
          ivf_pq_params_(ivf_pq_params),
          ivf_pq_search_params_(ivf_pq_search_params) {
    FAISS_THROW_IF_NOT_MSG(
            metric == faiss::METRIC_L2 || metric == faiss::METRIC_INNER_PRODUCT,
            "CAGRA currently only supports L2 or Inner Product metric.");
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for RAFT CAGRA index");

    index_params_.intermediate_graph_degree = intermediate_graph_degree;
    index_params_.graph_degree = graph_degree;
    index_params_.build_algo =
            static_cast<raft::neighbors::cagra::graph_build_algo>(
                    graph_build_algo);
    index_params_.nn_descent_niter = nn_descent_niter;

    if (!ivf_pq_params_) {
        ivf_pq_params_ =
                std::make_optional<raft::neighbors::ivf_pq::index_params>();
    }
    if (!ivf_pq_search_params_) {
        ivf_pq_search_params_ =
                std::make_optional<raft::neighbors::ivf_pq::search_params>();
    }
    index_params_.metric = metric_ == faiss::METRIC_L2
            ? raft::distance::DistanceType::L2Expanded
            : raft::distance::DistanceType::InnerProduct;
    ivf_pq_params_->metric = metric_ == faiss::METRIC_L2
            ? raft::distance::DistanceType::L2Expanded
            : raft::distance::DistanceType::InnerProduct;

    reset();
}

RaftCagra::RaftCagra(
        GpuResources* resources,
        int dim,
        idx_t n,
        int graph_degree,
        const float* distances,
        const idx_t* knn_graph,
        faiss::MetricType metric,
        float metricArg,
        IndicesOptions indicesOptions)
        : resources_(resources),
          dim_(dim),
          metric_(metric),
          metricArg_(metricArg) {
    FAISS_THROW_IF_NOT_MSG(
            metric == faiss::METRIC_L2 || metric == faiss::METRIC_INNER_PRODUCT,
            "CAGRA currently only supports L2 or Inner Product metric.");
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for RAFT CAGRA index");

    auto distances_on_gpu = getDeviceForAddress(distances) >= 0;
    auto knn_graph_on_gpu = getDeviceForAddress(knn_graph) >= 0;

    FAISS_ASSERT(distances_on_gpu == knn_graph_on_gpu);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    if (distances_on_gpu && knn_graph_on_gpu) {
        raft_handle.sync_stream();
        // Copying to host so that raft::neighbors::cagra::index
        // creates an owning copy of the knn graph on device
        auto knn_graph_copy =
                raft::make_host_matrix<uint32_t, int64_t>(n, graph_degree);
        thrust::copy(
                thrust::device_ptr<const idx_t>(knn_graph),
                thrust::device_ptr<const idx_t>(knn_graph + (n * graph_degree)),
                knn_graph_copy.data_handle());

        auto distances_mds =
                raft::make_device_matrix_view<const float, int64_t>(
                        distances, n, dim);

        raft_knn_index = raft::neighbors::cagra::index<float, uint32_t>(
                raft_handle,
                metric_ == faiss::METRIC_L2
                        ? raft::distance::DistanceType::L2Expanded
                        : raft::distance::DistanceType::InnerProduct,
                distances_mds,
                raft::make_const_mdspan(knn_graph_copy.view()));
    } else if (!distances_on_gpu && !knn_graph_on_gpu) {
        // copy idx_t (int64_t) host knn_graph to uint32_t host knn_graph
        auto knn_graph_copy =
                raft::make_host_matrix<uint32_t, int64_t>(n, graph_degree);
        std::copy(
                knn_graph,
                knn_graph + (n * graph_degree),
                knn_graph_copy.data_handle());

        auto distances_mds = raft::make_host_matrix_view<const float, int64_t>(
                distances, n, dim);

        raft_knn_index = raft::neighbors::cagra::index<float, uint32_t>(
                raft_handle,
                metric_ == faiss::METRIC_L2
                        ? raft::distance::DistanceType::L2Expanded
                        : raft::distance::DistanceType::InnerProduct,
                distances_mds,
                raft::make_const_mdspan(knn_graph_copy.view()));
    } else {
        FAISS_THROW_MSG(
                "distances and knn_graph must both be in device or host memory");
    }
}

void RaftCagra::train(idx_t n, const float* x) {
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    if (index_params_.build_algo ==
        raft::neighbors::cagra::graph_build_algo::IVF_PQ) {
        std::optional<raft::host_matrix<uint32_t, int64_t>> knn_graph(
                raft::make_host_matrix<uint32_t, int64_t>(
                        n, index_params_.intermediate_graph_degree));
        if (getDeviceForAddress(x) >= 0) {
            auto dataset_d =
                    raft::make_device_matrix_view<const float, int64_t>(
                            x, n, dim_);
            raft::neighbors::cagra::build_knn_graph(
                    raft_handle,
                    dataset_d,
                    knn_graph->view(),
                    1.0f,
                    ivf_pq_params_,
                    ivf_pq_search_params_);
        } else {
            auto dataset_h = raft::make_host_matrix_view<const float, int64_t>(
                    x, n, dim_);
            raft::neighbors::cagra::build_knn_graph(
                    raft_handle,
                    dataset_h,
                    knn_graph->view(),
                    1.0f,
                    ivf_pq_params_,
                    ivf_pq_search_params_);
        }
        auto cagra_graph = raft::make_host_matrix<uint32_t, int64_t>(
                n, index_params_.graph_degree);

        raft::neighbors::cagra::optimize<uint32_t>(
                raft_handle, knn_graph->view(), cagra_graph.view());

        // free intermediate graph before trying to create the index
        knn_graph.reset();

        if (getDeviceForAddress(x) >= 0) {
            auto dataset_d =
                    raft::make_device_matrix_view<const float, int64_t>(
                            x, n, dim_);
            raft_knn_index = raft::neighbors::cagra::index<float, uint32_t>(
                    raft_handle,
                    metric_ == faiss::METRIC_L2
                            ? raft::distance::DistanceType::L2Expanded
                            : raft::distance::DistanceType::InnerProduct,
                    dataset_d,
                    raft::make_const_mdspan(cagra_graph.view()));
        } else {
            auto dataset_h = raft::make_host_matrix_view<const float, int64_t>(
                    x, n, dim_);
            raft_knn_index = raft::neighbors::cagra::index<float, uint32_t>(
                    raft_handle,
                    metric_ == faiss::METRIC_L2
                            ? raft::distance::DistanceType::L2Expanded
                            : raft::distance::DistanceType::InnerProduct,
                    dataset_h,
                    raft::make_const_mdspan(cagra_graph.view()));
        }

    } else {
        if (getDeviceForAddress(x) >= 0) {
            raft_knn_index = raft::runtime::neighbors::cagra::build(
                    raft_handle,
                    index_params_,
                    raft::make_device_matrix_view<const float, int64_t>(
                            x, n, dim_));
        } else {
            raft_knn_index = raft::runtime::neighbors::cagra::build(
                    raft_handle,
                    index_params_,
                    raft::make_host_matrix_view<const float, int64_t>(
                            x, n, dim_));
        }
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

    auto queries_view = raft::make_device_matrix_view<const float, int64_t>(
            queries.data(), numQueries, cols);
    auto distances_view = raft::make_device_matrix_view<float, int64_t>(
            outDistances.data(), numQueries, k_);
    auto indices_view = raft::make_device_matrix_view<idx_t, int64_t>(
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

    auto indices_copy = raft::make_device_matrix<uint32_t, int64_t>(
            raft_handle, numQueries, k_);

    raft::runtime::neighbors::cagra::search(
            raft_handle,
            search_pams,
            raft_knn_index.value(),
            queries_view,
            indices_copy.view(),
            distances_view);
    thrust::copy(
            raft::resource::get_thrust_policy(raft_handle),
            indices_copy.data_handle(),
            indices_copy.data_handle() + indices_copy.size(),
            indices_view.data_handle());
}

void RaftCagra::reset() {
    raft_knn_index.reset();
}

idx_t RaftCagra::get_knngraph_degree() const {
    FAISS_ASSERT(raft_knn_index.has_value());
    return static_cast<idx_t>(raft_knn_index.value().graph_degree());
}

std::vector<idx_t> RaftCagra::get_knngraph() const {
    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    auto device_graph = raft_knn_index.value().graph();

    std::vector<idx_t> host_graph(
            device_graph.extent(0) * device_graph.extent(1));

    raft_handle.sync_stream();

    thrust::copy(
            thrust::device_ptr<const uint32_t>(device_graph.data_handle()),
            thrust::device_ptr<const uint32_t>(
                    device_graph.data_handle() + device_graph.size()),
            host_graph.data());

    return host_graph;
}

std::vector<float> RaftCagra::get_training_dataset() const {
    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    auto device_dataset = raft_knn_index.value().dataset();

    std::vector<float> host_dataset(
            device_dataset.extent(0) * device_dataset.extent(1));

    RAFT_CUDA_TRY(cudaMemcpy2DAsync(
            host_dataset.data(),
            sizeof(float) * dim_,
            device_dataset.data_handle(),
            sizeof(float) * device_dataset.stride(0),
            sizeof(float) * dim_,
            device_dataset.extent(0),
            cudaMemcpyDefault,
            raft_handle.get_stream()));
    raft_handle.sync_stream();

    return host_dataset;
}

} // namespace gpu
} // namespace faiss
