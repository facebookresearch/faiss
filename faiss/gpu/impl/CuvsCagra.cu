// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/CuvsFilterConvert.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/CuvsCagra.cuh>

#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/bitset.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

namespace faiss {
namespace gpu {

namespace {

cuvs::distance::DistanceType cppMetric(faiss::MetricType metric) {
    return static_cast<cuvs::distance::DistanceType>(
            metricFaissToCuvs(metric, false));
}

void configureCppIvfPqParams(
        cuvs::neighbors::ivf_pq::index_params& params,
        const IVFPQBuildCagraConfig* config,
        faiss::MetricType metric) {
    if (!config) {
        return;
    }

    params.metric = cppMetric(metric);
    params.n_lists = config->n_lists;
    params.kmeans_n_iters = config->kmeans_n_iters;
    params.kmeans_trainset_fraction = config->kmeans_trainset_fraction;
    params.pq_bits = config->pq_bits;
    params.pq_dim = config->pq_dim;
    params.codebook_kind = static_cast<cuvs::neighbors::ivf_pq::codebook_gen>(
            config->codebook_kind);
    params.force_random_rotation = config->force_random_rotation;
    params.conservative_memory_allocation =
            config->conservative_memory_allocation;
}

void configureCppIvfPqSearchParams(
        cuvs::neighbors::ivf_pq::search_params& params,
        const IVFPQSearchCagraConfig* config) {
    if (!config) {
        return;
    }

    params.n_probes = config->n_probes;
    params.lut_dtype = config->lut_dtype;
    params.internal_distance_dtype = config->internal_distance_dtype;
    params.preferred_shmem_carveout = config->preferred_shmem_carveout;
    params.max_internal_batch_size = config->max_internal_batch_size;
}

void configureCIvfPqParams(
        cuvsIvfPqIndexParams_t params,
        const IVFPQBuildCagraConfig& config,
        faiss::MetricType metric) {
    params->metric = metricFaissToCuvs(metric, false);
    params->n_lists = config.n_lists;
    params->kmeans_n_iters = config.kmeans_n_iters;
    params->kmeans_trainset_fraction = config.kmeans_trainset_fraction;
    params->pq_bits = config.pq_bits;
    params->pq_dim = config.pq_dim;
    params->codebook_kind =
            static_cast<cuvsIvfPqCodebookGen>(config.codebook_kind);
    params->force_random_rotation = config.force_random_rotation;
    params->conservative_memory_allocation =
            config.conservative_memory_allocation;
}

void configureCIvfPqSearchParams(
        cuvsIvfPqSearchParams_t params,
        const IVFPQSearchCagraConfig& config) {
    params->n_probes = config.n_probes;
    params->lut_dtype = config.lut_dtype;
    params->internal_distance_dtype = config.internal_distance_dtype;
    params->preferred_shmem_carveout = config.preferred_shmem_carveout;
    params->max_internal_batch_size = config.max_internal_batch_size;
}

template <typename T>
cuvsDataset_t makeAndAttachPaddedDataset(
        GpuResources* resources,
        const T* data,
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
        const IVFPQBuildCagraConfig* ivf_pq_params,
        const IVFPQSearchCagraConfig* ivf_pq_search_params,
        float refine_rate,
        bool guarantee_connectivity)
        : resources_(resources),
          storage_(nullptr),
          n_(0),
          dim_(dim),
          store_dataset_(store_dataset),
          metric_(metric),
          metricArg_(metricArg),
          graph_build_algo_(graph_build_algo),
          intermediate_graph_degree_(intermediate_graph_degree),
          graph_degree_(graph_degree),
          ivf_pq_params_(ivf_pq_params),
          ivf_pq_search_params_(ivf_pq_search_params),
          refine_rate_(refine_rate),
          nn_descent_niter_(nn_descent_niter),
          guarantee_connectivity_(guarantee_connectivity) {
    FAISS_THROW_IF_NOT_MSG(
            metric == faiss::METRIC_L2 || metric == faiss::METRIC_INNER_PRODUCT,
            "CAGRA currently only supports L2 or Inner Product metric.");
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS CAGRA index");
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
          storage_(dataset),
          n_(n),
          dim_(dim),
          metric_(metric),
          metricArg_(metricArg),
          graph_build_algo_(faiss::cagra_build_algo::IVF_PQ),
          intermediate_graph_degree_(graph_degree),
          graph_degree_(graph_degree),
          ivf_pq_params_(nullptr),
          ivf_pq_search_params_(nullptr),
          refine_rate_(2.0f) {
    FAISS_THROW_IF_NOT_MSG(
            metric == faiss::METRIC_L2 || metric == faiss::METRIC_INNER_PRODUCT,
            "CAGRA currently only supports L2 or Inner Product metric.");
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == faiss::gpu::INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS CAGRA index");

    const bool datasetOnGpu = getDeviceForAddress(dataset) >= 0;
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

    if (!datasetOnGpu || cuvsPaddedDatasetNeedsView(dataset, dim_)) {
        ownedDataset_ = DeviceTensor<data_t, 2, true>(
                resources_,
                AllocInfo(
                        AllocType::Other,
                        getCurrentDevice(),
                        MemorySpace::Device,
                        raftHandle.get_stream()),
                {n, dim_});
        raft::copy(
                ownedDataset_.data(),
                dataset,
                ownedDataset_.numElements(),
                raftHandle.get_stream());
        storage_ = ownedDataset_.data();
    }

    auto graphTensor =
            makeCuvsTensor(hostGraph.data(), (int64_t)n, (int64_t)graph_degree);
    auto datasetTensor = makeCuvsTensor(
            const_cast<data_t*>(storage_), (int64_t)n, (int64_t)dim_);

    cuvsCagraIndex_t index = nullptr;
    cuvsCheck(cuvsCagraIndexCreate(&index), "cuvsCagraIndexCreate");
    CuvsUniquePtr<cuvsCagraIndex, cuvsCagraIndexDestroy> indexHolder(index);
    cuvsCheck(
            cuvsCagraIndexFromArgs(
                    cuvsResourcesFromGpuResources(resources_),
                    metricFaissToCuvs(metric_, false),
                    graphTensor.get(),
                    datasetTensor.get(),
                    index),
            "cuvsCagraIndexFromArgs");
    cuvs_dataset_ = makeAndAttachPaddedDataset(
            resources_,
            datasetOnGpu ? storage_ : dataset,
            (int64_t)n_,
            (int64_t)dim_,
            index);
    raftHandle.sync_stream();
    cuvs_index = indexHolder.release();
}

template <typename data_t>
CuvsCagra<data_t>::~CuvsCagra() {
    if (cuvs_index) {
        cuvsCagraIndexDestroy(cuvs_index);
    }
    if (cuvs_dataset_) {
        cuvsDatasetDestroy(cuvs_dataset_);
    }
}

template <typename data_t>
void CuvsCagra<data_t>::train(idx_t n, const data_t* x) {
    reset();
    storage_ = x;
    n_ = n;

    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();

    // Compatibility note: release/26.10 has no C API field corresponding to
    // cagra::index_params::guarantee_connectivity, and its CAGRA bridge does
    // not propagate IVF-PQ search_params::max_internal_batch_size. Use C++
    // only to build graphs needing either setting, then put the result back
    // behind a C API index handle.
    const bool needsCppGraphBuild = guarantee_connectivity_ ||
            (graph_build_algo_ == faiss::cagra_build_algo::IVF_PQ &&
             ivf_pq_search_params_);
    if (needsCppGraphBuild) {
        cuvs::neighbors::cagra::index_params params;
        params.metric = cppMetric(metric_);
        params.intermediate_graph_degree = intermediate_graph_degree_;
        params.graph_degree = graph_degree_;
        params.attach_dataset_on_build = false;
        params.guarantee_connectivity = guarantee_connectivity_;

        if (graph_build_algo_ == faiss::cagra_build_algo::IVF_PQ) {
            cuvs::neighbors::cagra::graph_build_params::ivf_pq_params build(
                    raft::make_extents<uint32_t>(
                            static_cast<uint32_t>(n_),
                            static_cast<uint32_t>(dim_)),
                    cppMetric(metric_));
            configureCppIvfPqParams(
                    build.build_params, ivf_pq_params_, metric_);
            configureCppIvfPqSearchParams(
                    build.search_params, ivf_pq_search_params_);
            build.refinement_rate = refine_rate_;
            params.graph_build_params = build;
            if (params.graph_degree == params.intermediate_graph_degree) {
                params.intermediate_graph_degree =
                        static_cast<size_t>(1.5 * params.graph_degree);
            }
        } else {
            cuvs::neighbors::cagra::graph_build_params::nn_descent_params build(
                    params.intermediate_graph_degree);
            build.max_iterations = nn_descent_niter_;
            build.metric = cppMetric(metric_);
            params.graph_build_params = build;
        }

        ownedDataset_ = DeviceTensor<data_t, 2, true>(
                resources_,
                AllocInfo(
                        AllocType::Other,
                        getCurrentDevice(),
                        MemorySpace::Device,
                        raftHandle.get_stream()),
                {n, dim_});
        raft::copy(
                ownedDataset_.data(),
                x,
                ownedDataset_.numElements(),
                raftHandle.get_stream());
        storage_ = ownedDataset_.data();

        auto sourceView = raft::make_device_matrix_view<const data_t, int64_t>(
                storage_, n, dim_);
        auto paddedDataset = cuvs::neighbors::make_device_padded_dataset(
                raftHandle, sourceView);
        auto cppIndex = cuvs::neighbors::cagra::build(
                raftHandle, params, paddedDataset->as_dataset_view());
        auto graph = cppIndex.graph();
        auto graphTensor = makeCuvsTensor(
                graph.data_handle(),
                static_cast<int64_t>(graph.extent(0)),
                static_cast<int64_t>(graph.extent(1)));
        auto datasetTensor =
                makeCuvsTensor(ownedDataset_.data(), (int64_t)n, (int64_t)dim_);

        cuvsCagraIndex_t index = nullptr;
        cuvsCheck(cuvsCagraIndexCreate(&index), "cuvsCagraIndexCreate");
        CuvsUniquePtr<cuvsCagraIndex, cuvsCagraIndexDestroy> indexHolder(index);
        cuvsCheck(
                cuvsCagraIndexFromArgs(
                        cuvsResourcesFromGpuResources(resources_),
                        metricFaissToCuvs(metric_, false),
                        graphTensor.get(),
                        datasetTensor.get(),
                        index),
                "cuvsCagraIndexFromArgs");
        cuvs_dataset_ = makeAndAttachPaddedDataset(
                resources_, storage_, (int64_t)n_, (int64_t)dim_, index);
        raftHandle.sync_stream();
        cuvs_index = indexHolder.release();
        store_dataset_ = true;
        return;
    }

    const data_t* buildData = x;
    if (store_dataset_ && cuvsPaddedDatasetNeedsView(x, dim_)) {
        ownedDataset_ = DeviceTensor<data_t, 2, true>(
                resources_,
                AllocInfo(
                        AllocType::Other,
                        getCurrentDevice(),
                        MemorySpace::Device,
                        raftHandle.get_stream()),
                {n, dim_});
        raft::copy(
                ownedDataset_.data(),
                x,
                ownedDataset_.numElements(),
                raftHandle.get_stream());
        storage_ = ownedDataset_.data();
        buildData = storage_;
    }

    auto sourceTensor = makeCuvsTensor(
            const_cast<data_t*>(buildData), (int64_t)n, (int64_t)dim_);
    cuvsDataset_t dataset = nullptr;
    if (store_dataset_) {
        dataset = makeCuvsPaddedDataset(
                resources_, buildData, (int64_t)n, (int64_t)dim_);
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
    // The C factory owns and preallocates this nested IVF-PQ block. Detach it
    // while selecting the build algorithm so destruction follows C ownership.
    std::unique_ptr<cuvsIvfPqParams> ivfPqParams(
            static_cast<cuvsIvfPqParams*>(params->graph_build_params));
    params->graph_build_params = nullptr;
    ivfPqParams->refinement_rate = refine_rate_;
    params->metric = metricFaissToCuvs(metric_, false);
    params->intermediate_graph_degree = intermediate_graph_degree_;
    params->graph_degree = graph_degree_;
    params->nn_descent_niter = nn_descent_niter_;

    cuvsIvfPqIndexParams_t ivfBuildParams = nullptr;
    cuvsIvfPqSearchParams_t ivfSearchParams = nullptr;
    std::unique_ptr<
            cuvsIvfPqIndexParams,
            CuvsDeleter<cuvsIvfPqIndexParams, cuvsIvfPqIndexParamsDestroy>>
            ivfBuildHolder;
    std::unique_ptr<
            cuvsIvfPqSearchParams,
            CuvsDeleter<cuvsIvfPqSearchParams, cuvsIvfPqSearchParamsDestroy>>
            ivfSearchHolder;
    if (graph_build_algo_ == faiss::cagra_build_algo::IVF_PQ) {
        params->build_algo = IVF_PQ;
        if (params->graph_degree == params->intermediate_graph_degree) {
            params->intermediate_graph_degree =
                    static_cast<size_t>(1.5 * params->graph_degree);
        }
        if (ivf_pq_params_) {
            cuvsCheck(
                    cuvsIvfPqIndexParamsCreate(&ivfBuildParams),
                    "cuvsIvfPqIndexParamsCreate");
            ivfBuildHolder.reset(ivfBuildParams);
            configureCIvfPqParams(ivfBuildParams, *ivf_pq_params_, metric_);
            ivfPqParams->ivf_pq_build_params = ivfBuildParams;
        }
        if (ivf_pq_search_params_) {
            cuvsCheck(
                    cuvsIvfPqSearchParamsCreate(&ivfSearchParams),
                    "cuvsIvfPqSearchParamsCreate");
            ivfSearchHolder.reset(ivfSearchParams);
            configureCIvfPqSearchParams(
                    ivfSearchParams, *ivf_pq_search_params_);
            ivfPqParams->ivf_pq_search_params = ivfSearchParams;
        }
        params->graph_build_params = ivfPqParams.release();
    } else {
        params->build_algo = NN_DESCENT;
        params->graph_build_params = nullptr;
    }

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
        idx_t rand_xor_mask,
        const IDSelector* sel) {
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    const idx_t numQueries = queries.getSize(0);
    const idx_t cols = queries.getSize(1);

    FAISS_ASSERT(cuvs_index);
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);

    if (!store_dataset_) {
        if (cuvsPaddedDatasetNeedsView(storage_, dim_)) {
            ownedDataset_ = DeviceTensor<data_t, 2, true>(
                    resources_,
                    AllocInfo(
                            AllocType::Other,
                            getCurrentDevice(),
                            MemorySpace::Device,
                            raftHandle.get_stream()),
                    {n_, dim_});
            raft::copy(
                    ownedDataset_.data(),
                    storage_,
                    ownedDataset_.numElements(),
                    raftHandle.get_stream());
            storage_ = ownedDataset_.data();
        }
        auto paddedDataset = makeAndAttachPaddedDataset(
                resources_, storage_, (int64_t)n_, (int64_t)dim_, cuvs_index);
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
    auto queryTensor =
            makeCuvsTensor(queries.data(), (int64_t)numQueries, (int64_t)cols);
    auto indexTensor = makeCuvsTensor(
            indexCopy.data_handle(), (int64_t)numQueries, (int64_t)k);
    auto distanceTensor = makeCuvsTensor(
            outDistances.data(), (int64_t)numQueries, (int64_t)k);
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
}

template <typename data_t>
void CuvsCagra<data_t>::reset() {
    if (cuvs_index) {
        cuvsCheck(cuvsCagraIndexDestroy(cuvs_index), "cuvsCagraIndexDestroy");
        cuvs_index = nullptr;
    }
    if (cuvs_dataset_) {
        cuvsCheck(cuvsDatasetDestroy(cuvs_dataset_), "cuvsDatasetDestroy");
        cuvs_dataset_ = nullptr;
    }
}

template <typename data_t>
idx_t CuvsCagra<data_t>::get_knngraph_degree() const {
    FAISS_ASSERT(cuvs_index);

    int64_t degree = 0;
    cuvsCheck(
            cuvsCagraIndexGetGraphDegree(cuvs_index, &degree),
            "cuvsCagraIndexGetGraphDegree");
    return static_cast<idx_t>(degree);
}

template <typename data_t>
std::vector<idx_t> CuvsCagra<data_t>::get_knngraph() const {
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

template <typename data_t>
const data_t* CuvsCagra<data_t>::get_training_dataset() const {
    return storage_;
}

template class CuvsCagra<float>;
template class CuvsCagra<half>;
template class CuvsCagra<int8_t>;
} // namespace gpu
} // namespace faiss
