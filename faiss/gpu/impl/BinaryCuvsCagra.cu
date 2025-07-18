// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <faiss/gpu/impl/BinaryCuvsCagra.cuh>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map.cuh>

namespace faiss {
namespace gpu {

BinaryCuvsCagra::BinaryCuvsCagra(
        GpuResources* resources,
        int dim,
        idx_t intermediate_graph_degree,
        idx_t graph_degree,
        bool store_dataset,
        IndicesOptions indicesOptions)
        : resources_(resources), dim_(dim), store_dataset_(store_dataset) {
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS CAGRA index");

    index_params_.intermediate_graph_degree = intermediate_graph_degree;
    index_params_.graph_degree = graph_degree;
    index_params_.attach_dataset_on_build = store_dataset;

    index_params_.metric = cuvs::distance::DistanceType::BitwiseHamming;

    reset();
}

BinaryCuvsCagra::BinaryCuvsCagra(
        GpuResources* resources,
        int dim,
        idx_t n,
        int graph_degree,
        const uint8_t* train_dataset,
        const idx_t* knn_graph,
        IndicesOptions indicesOptions)
        : resources_(resources), dim_(dim) {
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS CAGRA index");

    bool distances_on_gpu = getDeviceForAddress(train_dataset) >= 0;
    bool knn_graph_on_gpu = getDeviceForAddress(knn_graph) >= 0;

    FAISS_ASSERT(distances_on_gpu == knn_graph_on_gpu);

    storage_ = train_dataset;
    n_ = n;

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    if (distances_on_gpu && knn_graph_on_gpu) {
        raft_handle.sync_stream();
        // Copying to host so that cuvs::neighbors::cagra::index
        // creates an owning copy of the knn graph on device
        auto knn_graph_copy =
                raft::make_host_matrix<uint32_t, int64_t>(n, graph_degree);
        thrust::copy(
                thrust::device_ptr<const idx_t>(knn_graph),
                thrust::device_ptr<const idx_t>(knn_graph + (n * graph_degree)),
                knn_graph_copy.data_handle());

        auto dataset_mds =
                raft::make_device_matrix_view<const uint8_t, int64_t>(
                        train_dataset, n, dim / 8);

        cuvs_index = std::make_shared<
                cuvs::neighbors::cagra::index<uint8_t, uint32_t>>(
                raft_handle,
                cuvs::distance::DistanceType::BitwiseHamming,
                dataset_mds,
                raft::make_const_mdspan(knn_graph_copy.view()));
    } else if (!distances_on_gpu && !knn_graph_on_gpu) {
        // copy idx_t (int64_t) host knn_graph to uint32_t host knn_graph
        auto knn_graph_copy =
                raft::make_host_matrix<uint32_t, int64_t>(n, graph_degree);
        std::copy(
                knn_graph,
                knn_graph + (n * graph_degree),
                knn_graph_copy.data_handle());

        auto dataset_mds = raft::make_host_matrix_view<const uint8_t, int64_t>(
                train_dataset, n, dim / 8);

        cuvs_index = std::make_shared<
                cuvs::neighbors::cagra::index<uint8_t, uint32_t>>(
                raft_handle,
                cuvs::distance::DistanceType::BitwiseHamming,
                dataset_mds,
                raft::make_const_mdspan(knn_graph_copy.view()));
    } else {
        FAISS_THROW_MSG(
                "distances and knn_graph must both be in device or host memory");
    }
}

void BinaryCuvsCagra::train(idx_t n, const uint8_t* x) {
    storage_ = x;
    n_ = n;

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    // BitwiseHamming metric only supports CAGRA iterative search as the graph
    // building algorithm
    cuvs::neighbors::cagra::graph_build_params::iterative_search_params
            graph_build_params;
    index_params_.graph_build_params = graph_build_params;

    if (getDeviceForAddress(x) >= 0) {
        auto dataset = raft::make_device_matrix_view<const uint8_t, int64_t>(
                x, n, dim_ / 8);
        cuvs_index = std::make_shared<
                cuvs::neighbors::cagra::index<uint8_t, uint32_t>>(
                cuvs::neighbors::cagra::build(
                        raft_handle, index_params_, dataset));
    } else {
        auto dataset = raft::make_host_matrix_view<const uint8_t, int64_t>(
                x, n, dim_ / 8);
        cuvs_index = std::make_shared<
                cuvs::neighbors::cagra::index<uint8_t, uint32_t>>(
                cuvs::neighbors::cagra::build(
                        raft_handle, index_params_, dataset));
    }
}

void BinaryCuvsCagra::search(
        Tensor<uint8_t, 2, true>& queries,
        int k,
        Tensor<int, 2, true>& outDistances,
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
    auto distances_float =
            raft::make_device_matrix<float>(raft_handle, numQueries, k_);

    FAISS_ASSERT(cuvs_index);
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_ / 8);

    if (!store_dataset_) {
        if (getDeviceForAddress(storage_) >= 0) {
            auto dataset =
                    raft::make_device_matrix_view<const uint8_t, int64_t>(
                            storage_, n_, dim_ / 8);
            cuvs_index->update_dataset(raft_handle, dataset);
        } else {
            auto dataset = raft::make_host_matrix_view<const uint8_t, int64_t>(
                    storage_, n_, dim_ / 8);
            cuvs_index->update_dataset(raft_handle, dataset);
        }
        store_dataset_ = true;
    }

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

    auto queries_view = raft::make_device_matrix_view<const uint8_t, int64_t>(
            queries.data(), numQueries, cols);
    auto indices_copy = raft::make_device_matrix<uint32_t, int64_t>(
            raft_handle, numQueries, k_);
    auto distances_float_view = distances_float.view();

    cuvs::neighbors::cagra::search(
            raft_handle,
            search_pams,
            *cuvs_index,
            queries_view,
            indices_copy.view(),
            distances_float_view);

    thrust::copy(
            raft::resource::get_thrust_policy(raft_handle),
            indices_copy.data_handle(),
            indices_copy.data_handle() + indices_copy.size(),
            indices_view.data_handle());
    auto distances_view = raft::make_device_matrix_view(
            outDistances.data(),
            static_cast<int64_t>(numQueries),
            static_cast<int64_t>(k));

    raft::linalg::map_offset(
            raft_handle,
            distances_view,
            [distances_float_view, k_] __device__(size_t i) {
                int row_idx = i / k_;
                int col_idx = i % k_;
                return static_cast<int>(distances_float_view(row_idx, col_idx));
            });
}

void BinaryCuvsCagra::reset() {
    cuvs_index.reset();
}

idx_t BinaryCuvsCagra::get_knngraph_degree() const {
    FAISS_ASSERT(cuvs_index);
    return static_cast<idx_t>(cuvs_index->graph_degree());
}

std::vector<idx_t> BinaryCuvsCagra::get_knngraph() const {
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

const uint8_t* BinaryCuvsCagra::get_training_dataset() const {
    return storage_;
}
} // namespace gpu
} // namespace faiss
