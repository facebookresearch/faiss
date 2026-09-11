// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
#include <faiss/gpu/utils/CuvsFilterConvert.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/BinaryCuvsCagra.cuh>

#include <cuvs/neighbors/cagra.h>
#include <raft/core/device_resources.hpp>
#include <raft/linalg/map.cuh>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace faiss {
namespace gpu {

namespace {

cuvsDataset_t makeAndAttachPaddedDataset(
        GpuResources* resources,
        const uint8_t* data,
        int64_t rows,
        int64_t cols,
        cuvsCagraIndex_t index) {
    cuvsDataset_t paddedDataset =
            makeCuvsPaddedDataset(resources, data, rows, cols);
    CuvsUniquePtr<cuvsDataset, cuvsDatasetDestroy> paddedHolder(paddedDataset);
    cuvsCheck(
            cuvsCagraUpdateDataset(
                    cuvsResourcesFromGpuResources(resources),
                    paddedDataset,
                    index),
            "cuvsCagraUpdateDataset");
    return paddedHolder.release();
}

} // namespace

BinaryCuvsCagra::BinaryCuvsCagra(
        GpuResources* resources,
        int dim,
        idx_t intermediate_graph_degree,
        idx_t graph_degree,
        faiss::cagra_build_algo graph_build_algo,
        size_t nn_descent_niter,
        bool store_dataset,
        IndicesOptions indicesOptions)
        : resources_(resources),
          storage_(nullptr),
          n_(0),
          dim_(dim),
          store_dataset_(store_dataset),
          graph_build_algo_(graph_build_algo),
          intermediate_graph_degree_(intermediate_graph_degree),
          graph_degree_(graph_degree),
          nn_descent_niter_(nn_descent_niter) {
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS CAGRA index");

    if (graph_build_algo == faiss::cagra_build_algo::IVF_PQ) {
        fprintf(stderr,
                "WARNING: IVF_PQ is not supported for binary CAGRA. "
                "Defaulting to NN_DESCENT\n");
    }
}

BinaryCuvsCagra::BinaryCuvsCagra(
        GpuResources* resources,
        int dim,
        idx_t n,
        int graph_degree,
        const uint8_t* train_dataset,
        const idx_t* knn_graph,
        IndicesOptions indicesOptions)
        : resources_(resources),
          storage_(train_dataset),
          n_(n),
          dim_(dim),
          graph_build_algo_(faiss::cagra_build_algo::NN_DESCENT),
          intermediate_graph_degree_(graph_degree),
          graph_degree_(graph_degree) {
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS CAGRA index");

    const bool datasetOnGpu = getDeviceForAddress(train_dataset) >= 0;
    const bool graphOnGpu = getDeviceForAddress(knn_graph) >= 0;
    FAISS_THROW_IF_NOT_MSG(
            datasetOnGpu == graphOnGpu,
            "dataset and knn_graph must both be in device or host memory");

    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    std::vector<uint32_t> hostGraph(n * graph_degree);
    if (graphOnGpu) {
        raftHandle.sync_stream();
        thrust::copy(
                thrust::device_ptr<const idx_t>(knn_graph),
                thrust::device_ptr<const idx_t>(knn_graph + n * graph_degree),
                hostGraph.data());
    } else {
        std::transform(
                knn_graph,
                knn_graph + n * graph_degree,
                hostGraph.begin(),
                [](idx_t value) { return static_cast<uint32_t>(value); });
    }

    if (!datasetOnGpu || cuvsPaddedDatasetNeedsView(train_dataset, dim_ / 8)) {
        ownedDataset_ = DeviceTensor<uint8_t, 2, true>(
                resources_,
                AllocInfo(
                        AllocType::Other,
                        getCurrentDevice(),
                        MemorySpace::Device,
                        raftHandle.get_stream()),
                {n, dim_ / 8});
        raft::copy(
                ownedDataset_.data(),
                train_dataset,
                ownedDataset_.numElements(),
                raftHandle.get_stream());
        storage_ = ownedDataset_.data();
    }

    auto graphTensor =
            makeCuvsTensor(hostGraph.data(), (int64_t)n, (int64_t)graph_degree);
    auto datasetTensor = makeCuvsTensor(
            const_cast<uint8_t*>(storage_), (int64_t)n, (int64_t)(dim_ / 8));

    cuvsCagraIndex_t index = nullptr;
    cuvsCheck(cuvsCagraIndexCreate(&index), "cuvsCagraIndexCreate");
    CuvsUniquePtr<cuvsCagraIndex, cuvsCagraIndexDestroy> indexHolder(index);
    cuvsCheck(
            cuvsCagraIndexFromArgs(
                    cuvsResourcesFromGpuResources(resources_),
                    BitwiseHamming,
                    graphTensor.get(),
                    datasetTensor.get(),
                    index),
            "cuvsCagraIndexFromArgs");
    cuvs_dataset_ = makeAndAttachPaddedDataset(
            resources_,
            datasetOnGpu ? storage_ : train_dataset,
            (int64_t)n_,
            (int64_t)(dim_ / 8),
            index);
    raftHandle.sync_stream();
    cuvs_index = indexHolder.release();
}

BinaryCuvsCagra::~BinaryCuvsCagra() {
    if (cuvs_index) {
        cuvsCagraIndexDestroy(cuvs_index);
    }
    if (cuvs_dataset_) {
        cuvsDatasetDestroy(cuvs_dataset_);
    }
}

void BinaryCuvsCagra::train(idx_t n, const uint8_t* x) {
    reset();
    storage_ = x;
    n_ = n;

    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    const uint8_t* buildData = x;
    if (store_dataset_ && cuvsPaddedDatasetNeedsView(x, dim_ / 8)) {
        ownedDataset_ = DeviceTensor<uint8_t, 2, true>(
                resources_,
                AllocInfo(
                        AllocType::Other,
                        getCurrentDevice(),
                        MemorySpace::Device,
                        raftHandle.get_stream()),
                {n, dim_ / 8});
        raft::copy(
                ownedDataset_.data(),
                x,
                ownedDataset_.numElements(),
                raftHandle.get_stream());
        storage_ = ownedDataset_.data();
        buildData = storage_;
    }

    auto sourceTensor = makeCuvsTensor(
            const_cast<uint8_t*>(buildData), (int64_t)n, (int64_t)(dim_ / 8));
    cuvsDataset_t dataset = nullptr;
    if (store_dataset_) {
        dataset = makeCuvsPaddedDataset(
                resources_, buildData, (int64_t)n, (int64_t)(dim_ / 8));
    } else {
        cuvsCheck(
                cuvsDatasetMakeStandardView(
                        cuvsResourcesFromGpuResources(resources_),
                        sourceTensor.get(),
                        &dataset),
                "cuvsDatasetMakeStandardView");
    }
    CuvsUniquePtr<cuvsDataset, cuvsDatasetDestroy> datasetHolder(dataset);

    cuvsCagraIndexParams_t params = nullptr;
    cuvsCheck(
            cuvsCagraIndexParamsCreate(&params), "cuvsCagraIndexParamsCreate");
    CuvsUniquePtr<cuvsCagraIndexParams, cuvsCagraIndexParamsDestroy>
            paramsHolder(params);
    // The C factory preallocates an IVF-PQ block, but binary CAGRA always uses
    // NN-descent. Detach and destroy the unused block explicitly.
    std::unique_ptr<cuvsIvfPqParams> unusedIvfPqParams(
            static_cast<cuvsIvfPqParams*>(params->graph_build_params));
    params->graph_build_params = nullptr;
    params->metric = BitwiseHamming;
    params->intermediate_graph_degree = intermediate_graph_degree_;
    params->graph_degree = graph_degree_;
    params->build_algo = NN_DESCENT;
    params->nn_descent_niter = nn_descent_niter_;

    cuvsCagraIndex_t index = nullptr;
    cuvsCheck(cuvsCagraIndexCreate(&index), "cuvsCagraIndexCreate");
    CuvsUniquePtr<cuvsCagraIndex, cuvsCagraIndexDestroy> indexHolder(index);
    cuvsCheck(
            cuvsCagraBuild(
                    cuvsResourcesFromGpuResources(resources_),
                    params,
                    dataset,
                    index),
            "cuvsCagraBuild");
    cuvs_dataset_ = datasetHolder.release();
    cuvs_index = indexHolder.release();
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
        idx_t rand_xor_mask,
        const IDSelector* sel) {
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    const idx_t numQueries = queries.getSize(0);
    const idx_t cols = queries.getSize(1);

    FAISS_ASSERT(cuvs_index);
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_ / 8);

    if (!store_dataset_) {
        if (cuvsPaddedDatasetNeedsView(storage_, dim_ / 8)) {
            ownedDataset_ = DeviceTensor<uint8_t, 2, true>(
                    resources_,
                    AllocInfo(
                            AllocType::Other,
                            getCurrentDevice(),
                            MemorySpace::Device,
                            raftHandle.get_stream()),
                    {n_, dim_ / 8});
            raft::copy(
                    ownedDataset_.data(),
                    storage_,
                    ownedDataset_.numElements(),
                    raftHandle.get_stream());
            storage_ = ownedDataset_.data();
        }
        auto paddedDataset = makeAndAttachPaddedDataset(
                resources_,
                storage_,
                (int64_t)n_,
                (int64_t)(dim_ / 8),
                cuvs_index);
        if (cuvs_dataset_) {
            cuvsCheck(cuvsDatasetDestroy(cuvs_dataset_), "cuvsDatasetDestroy");
        }
        cuvs_dataset_ = paddedDataset;
        store_dataset_ = true;
    }

    cuvsCagraSearchParams_t params = nullptr;
    cuvsCheck(
            cuvsCagraSearchParamsCreate(&params),
            "cuvsCagraSearchParamsCreate");
    CuvsUniquePtr<cuvsCagraSearchParams, cuvsCagraSearchParamsDestroy>
            paramsHolder(params);
    params->max_queries = max_queries;
    params->itopk_size = itopk_size;
    params->max_iterations = max_iterations;
    params->algo = static_cast<cuvsCagraSearchAlgo>(graph_search_algo);
    params->team_size = team_size;
    params->search_width = search_width;
    params->min_iterations = min_iterations;
    params->thread_block_size = thread_block_size;
    params->hashmap_mode = static_cast<cuvsCagraHashMode>(hash_mode);
    params->hashmap_min_bitlen = hashmap_min_bitlen;
    params->hashmap_max_fill_rate = hashmap_max_fill_rate;
    params->num_random_samplings = num_random_samplings;
    params->rand_xor_mask = rand_xor_mask;

    auto indexCopy = raft::make_device_matrix<uint32_t, int64_t>(
            raftHandle, numQueries, k);
    auto distanceCopy =
            raft::make_device_matrix<float, int64_t>(raftHandle, numQueries, k);
    auto queryTensor =
            makeCuvsTensor(queries.data(), (int64_t)numQueries, (int64_t)cols);
    auto indexTensor = makeCuvsTensor(
            indexCopy.data_handle(), (int64_t)numQueries, (int64_t)k);
    auto distanceTensor = makeCuvsTensor(
            distanceCopy.data_handle(), (int64_t)numQueries, (int64_t)k);
    CuvsFilter filter(resources_, sel, n_);

    cuvsCheck(
            cuvsCagraSearch(
                    cuvsResourcesFromGpuResources(resources_),
                    params,
                    cuvs_index,
                    queryTensor.get(),
                    indexTensor.get(),
                    distanceTensor.get(),
                    filter.get()),
            "cuvsCagraSearch");
    sanitizeCuvsIndices(
            resources_,
            indexCopy.data_handle(),
            outIndices.data(),
            indexCopy.size(),
            n_);

    auto distancesView = raft::make_device_matrix_view(
            outDistances.data(), (int64_t)numQueries, (int64_t)k);
    auto distanceCopyView = distanceCopy.view();
    raft::linalg::map_offset(
            raftHandle,
            distancesView,
            [distanceCopyView, k] __device__(size_t i) {
                const int row = i / k;
                const int col = i % k;
                return static_cast<int>(distanceCopyView(row, col));
            });
}

void BinaryCuvsCagra::reset() {
    if (cuvs_index) {
        cuvsCheck(cuvsCagraIndexDestroy(cuvs_index), "cuvsCagraIndexDestroy");
        cuvs_index = nullptr;
    }
    if (cuvs_dataset_) {
        cuvsCheck(cuvsDatasetDestroy(cuvs_dataset_), "cuvsDatasetDestroy");
        cuvs_dataset_ = nullptr;
    }
}

idx_t BinaryCuvsCagra::get_knngraph_degree() const {
    FAISS_ASSERT(cuvs_index);
    int64_t degree = 0;
    cuvsCheck(
            cuvsCagraIndexGetGraphDegree(cuvs_index, &degree),
            "cuvsCagraIndexGetGraphDegree");
    return static_cast<idx_t>(degree);
}

std::vector<idx_t> BinaryCuvsCagra::get_knngraph() const {
    FAISS_ASSERT(cuvs_index);
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();

    CuvsOutputTensor graphTensor;
    cuvsCheck(
            cuvsCagraIndexGetGraph(cuvs_index, graphTensor.get()),
            "cuvsCagraIndexGetGraph");
    const uint32_t* graphData = graphTensor.data<uint32_t>();
    const size_t graphSize = graphTensor.extent(0) * graphTensor.extent(1);

    std::vector<idx_t> hostGraph(graphSize);
    raftHandle.sync_stream();
    thrust::copy(
            thrust::device_ptr<const uint32_t>(graphData),
            thrust::device_ptr<const uint32_t>(graphData + graphSize),
            hostGraph.data());
    return hostGraph;
}

const uint8_t* BinaryCuvsCagra::get_training_dataset() const {
    return storage_;
}
} // namespace gpu
} // namespace faiss
