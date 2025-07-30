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

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/CuvsCagra.cuh>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>

namespace faiss {
namespace gpu {

template <typename data_t>
CuvsCagra<data_t>::CuvsCagra(
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
        std::optional<cuvs::neighbors::ivf_pq::index_params> ivf_pq_params,
        std::optional<cuvs::neighbors::ivf_pq::search_params>
                ivf_pq_search_params,
        float refine_rate,
        bool guarantee_connectivity)
        : resources_(resources),
          dim_(dim),
          graph_build_algo_(graph_build_algo),
          nn_descent_niter_(nn_descent_niter),
          store_dataset_(store_dataset),
          metric_(metric),
          metricArg_(metricArg),
          index_params_(),
          ivf_pq_params_(ivf_pq_params),
          ivf_pq_search_params_(ivf_pq_search_params),
          refine_rate_(refine_rate),
          guarantee_connectivity_(guarantee_connectivity) {
    FAISS_THROW_IF_NOT_MSG(
            metric == faiss::METRIC_L2 || metric == faiss::METRIC_INNER_PRODUCT,
            "CAGRA currently only supports L2 or Inner Product metric.");
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS CAGRA index");

    index_params_.intermediate_graph_degree = intermediate_graph_degree;
    index_params_.graph_degree = graph_degree;
    index_params_.attach_dataset_on_build = store_dataset;
    index_params_.guarantee_connectivity = guarantee_connectivity;

    if (!ivf_pq_search_params_) {
        ivf_pq_search_params_ =
                std::make_optional<cuvs::neighbors::ivf_pq::search_params>();
    }
    index_params_.metric = metricFaissToCuvs(metric_, false);

    reset();
}

template <typename data_t>
CuvsCagra<data_t>::CuvsCagra(
        GpuResources* resources,
        int dim,
        idx_t n,
        int graph_degree,
        const data_t* dataset,
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
            "only INDICES_64_BIT is supported for cuVS CAGRA index");

    auto dataset_on_gpu = getDeviceForAddress(dataset) >= 0;
    auto knn_graph_on_gpu = getDeviceForAddress(knn_graph) >= 0;

    FAISS_ASSERT(dataset_on_gpu == knn_graph_on_gpu);

    storage_ = dataset;
    n_ = n;

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    if (dataset_on_gpu && knn_graph_on_gpu) {
        raft_handle.sync_stream();
        // Copying to host so that cuvs::neighbors::cagra::index
        // creates an owning copy of the knn graph on device
        auto knn_graph_copy =
                raft::make_host_matrix<uint32_t, int64_t>(n, graph_degree);
        thrust::copy(
                thrust::device_ptr<const idx_t>(knn_graph),
                thrust::device_ptr<const idx_t>(knn_graph + (n * graph_degree)),
                knn_graph_copy.data_handle());

        auto dataset_mds = raft::make_device_matrix_view<const data_t, int64_t>(
                dataset, n, dim);

        cuvs_index = std::make_shared<
                cuvs::neighbors::cagra::index<data_t, uint32_t>>(
                raft_handle,
                metricFaissToCuvs(metric_, false),
                dataset_mds,
                raft::make_const_mdspan(knn_graph_copy.view()));
    } else if (!dataset_on_gpu && !knn_graph_on_gpu) {
        // copy idx_t (int64_t) host knn_graph to uint32_t host knn_graph
        auto knn_graph_copy =
                raft::make_host_matrix<uint32_t, int64_t>(n, graph_degree);
        std::copy(
                knn_graph,
                knn_graph + (n * graph_degree),
                knn_graph_copy.data_handle());

        auto dataset_mds = raft::make_host_matrix_view<const data_t, int64_t>(
                dataset, n, dim);

        cuvs_index = std::make_shared<
                cuvs::neighbors::cagra::index<data_t, uint32_t>>(
                raft_handle,
                metricFaissToCuvs(metric_, false),
                dataset_mds,
                raft::make_const_mdspan(knn_graph_copy.view()));
    } else {
        FAISS_THROW_MSG(
                "dataset and knn_graph must both be in device or host memory");
    }
}

template <typename data_t>
void CuvsCagra<data_t>::train(idx_t n, const data_t* x) {
    storage_ = x;
    n_ = n;

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    if (!ivf_pq_params_) {
        ivf_pq_params_ = cuvs::neighbors::ivf_pq::index_params::from_dataset(
                raft::make_extents<uint32_t>(
                        static_cast<uint32_t>(n_), static_cast<uint32_t>(dim_)),
                metricFaissToCuvs(metric_, false));
    }
    if (graph_build_algo_ == faiss::cagra_build_algo::IVF_PQ) {
        cuvs::neighbors::cagra::graph_build_params::ivf_pq_params
                graph_build_params;
        graph_build_params.build_params = ivf_pq_params_.value();
        graph_build_params.build_params.metric =
                metricFaissToCuvs(metric_, false);
        graph_build_params.search_params = ivf_pq_search_params_.value();
        graph_build_params.refinement_rate = refine_rate_.value();
        index_params_.graph_build_params = graph_build_params;
        if (index_params_.graph_degree ==
            index_params_.intermediate_graph_degree) {
            index_params_.intermediate_graph_degree =
                    1.5 * index_params_.graph_degree;
        }
    } else {
        cuvs::neighbors::cagra::graph_build_params::nn_descent_params
                graph_build_params(index_params_.intermediate_graph_degree);
        graph_build_params.max_iterations = nn_descent_niter_;
        index_params_.graph_build_params = graph_build_params;
    }

    if (getDeviceForAddress(x) >= 0) {
        auto dataset = raft::make_device_matrix_view<const data_t, int64_t>(
                x, n, dim_);
        cuvs_index = std::make_shared<
                cuvs::neighbors::cagra::index<data_t, uint32_t>>(
                cuvs::neighbors::cagra::build(
                        raft_handle, index_params_, dataset));
    } else {
        auto dataset =
                raft::make_host_matrix_view<const data_t, int64_t>(x, n, dim_);
        cuvs_index = std::make_shared<
                cuvs::neighbors::cagra::index<data_t, uint32_t>>(
                cuvs::neighbors::cagra::build(
                        raft_handle, index_params_, dataset));
    }
}

template <typename data_t>
void CuvsCagra<data_t>::search(
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
        idx_t rand_xor_mask) {
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    idx_t numQueries = queries.getSize(0);
    idx_t cols = queries.getSize(1);
    idx_t k_ = k;

    FAISS_ASSERT(cuvs_index);
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);

    if (!store_dataset_) {
        if (getDeviceForAddress(storage_) >= 0) {
            auto dataset = raft::make_device_matrix_view<const data_t, int64_t>(
                    storage_, n_, dim_);
            cuvs_index->update_dataset(raft_handle, dataset);
        } else {
            auto dataset = raft::make_host_matrix_view<const data_t, int64_t>(
                    storage_, n_, dim_);
            cuvs_index->update_dataset(raft_handle, dataset);
        }
        store_dataset_ = true;
    }

    auto queries_view = raft::make_device_matrix_view<const data_t, int64_t>(
            queries.data(), numQueries, cols);
    auto distances_view = raft::make_device_matrix_view<float, int64_t>(
            outDistances.data(), numQueries, k_);
    auto indices_view = raft::make_device_matrix_view<idx_t, int64_t>(
            outIndices.data(), numQueries, k_);

    cuvs::neighbors::cagra::search_params search_pams;
    search_pams.max_queries = max_queries;
    search_pams.itopk_size = itopk_size;
    search_pams.max_iterations = max_iterations;
    search_pams.algo =
            static_cast<cuvs::neighbors::cagra::search_algo>(graph_search_algo);
    search_pams.team_size = team_size;
    search_pams.search_width = search_width;
    search_pams.min_iterations = min_iterations;
    search_pams.thread_block_size = thread_block_size;
    search_pams.hashmap_mode =
            static_cast<cuvs::neighbors::cagra::hash_mode>(hash_mode);
    search_pams.hashmap_min_bitlen = hashmap_min_bitlen;
    search_pams.hashmap_max_fill_rate = hashmap_max_fill_rate;
    search_pams.num_random_samplings = num_random_samplings;
    search_pams.rand_xor_mask = rand_xor_mask;

    auto indices_copy = raft::make_device_matrix<uint32_t, int64_t>(
            raft_handle, numQueries, k_);

    cuvs::neighbors::cagra::search(
            raft_handle,
            search_pams,
            *cuvs_index,
            queries_view,
            indices_copy.view(),
            distances_view);
    thrust::copy(
            raft::resource::get_thrust_policy(raft_handle),
            indices_copy.data_handle(),
            indices_copy.data_handle() + indices_copy.size(),
            indices_view.data_handle());
}

template <typename data_t>
void CuvsCagra<data_t>::reset() {
    cuvs_index.reset();
}

template <typename data_t>
idx_t CuvsCagra<data_t>::get_knngraph_degree() const {
    FAISS_ASSERT(cuvs_index);
    return static_cast<idx_t>(cuvs_index->graph_degree());
}

template <typename data_t>
std::vector<idx_t> CuvsCagra<data_t>::get_knngraph() const {
    FAISS_ASSERT(cuvs_index);
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    auto device_graph = cuvs_index->graph();

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

template <typename data_t>
const data_t* CuvsCagra<data_t>::get_training_dataset() const {
    return storage_;
}

template class CuvsCagra<float>;
template class CuvsCagra<half>;
template class CuvsCagra<int8_t>;
} // namespace gpu
} // namespace faiss
