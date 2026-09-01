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

#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <faiss/gpu/impl/CuvsCagra.cuh>
#include <optional>
#include <type_traits>

#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/resource/multi_gpu.hpp>
// clang-format off
#if __has_include(<cuvs/neighbors/cagra_optimize.hpp>)
// Some cuVS versions put helpers::optimize in its own header.
#include <cuvs/neighbors/cagra_optimize.hpp>
#endif
// TEMPORARY. cuVS has no public API to prune a graph that is already on the
// device, so we reach into its internals when they happen to be available.
// Delete this and the branch it guards once that API ships upstream.
#if defined(FAISS_CAGRA_DEVICE_OPTIMIZE)
#include <neighbors/detail/cagra/graph_core.cuh>
#endif
// clang-format on

namespace {

// all_neighbors emits int64 indices, optimize consumes uint32. Narrowing on
// device lets the graph stay there.
__global__ void kern_narrow_indices(
        const int64_t* __restrict__ src,
        uint32_t* __restrict__ dst,
        size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        dst[i] = static_cast<uint32_t>(src[i]);
    }
}

} // anonymous namespace

namespace faiss {
namespace gpu {

GpuIndexCagra::GpuIndexCagra(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        GpuIndexCagraConfig config)
        : GpuIndex(provider->getResources(), dims, metric, 0.0f, config),
          cagraConfig_(config) {
    this->is_trained = false;
}

void GpuIndexCagra::train_ex(idx_t n, const void* x, NumericType numeric_type) {
    if (cagraConfig_.devices.size() > 1) {
        FAISS_THROW_IF_NOT_MSG(
                numeric_type == NumericType::Float32,
                "multi-GPU CAGRA build only supports Float32");
        trainAllNeighbors_(n, static_cast<const float*>(x));
        return;
    }

    FAISS_THROW_IF_MSG(
            cagraConfig_.build_algo == graph_build_algo::BRUTE_FORCE,
            "graph_build_algo::BRUTE_FORCE is only available on the multi-GPU "
            "build path (set GpuIndexCagraConfig::devices)");

    numeric_type_ = numeric_type;
    bool index_is_initialized = !std::holds_alternative<std::monostate>(index_);

    DeviceScope scope(config_.device);
    if (this->is_trained) {
        FAISS_ASSERT(index_is_initialized);
        return;
    }

    // CuvsCagra not initialized
    FAISS_ASSERT(!index_is_initialized);

    std::optional<cuvs::neighbors::ivf_pq::index_params> ivf_pq_params =
            std::nullopt;
    std::optional<cuvs::neighbors::ivf_pq::search_params> ivf_pq_search_params =
            std::nullopt;
    if (cagraConfig_.ivf_pq_params != nullptr) {
        ivf_pq_params =
                std::make_optional<cuvs::neighbors::ivf_pq::index_params>();
        ivf_pq_params->n_lists = cagraConfig_.ivf_pq_params->n_lists;
        ivf_pq_params->kmeans_n_iters =
                cagraConfig_.ivf_pq_params->kmeans_n_iters;
        ivf_pq_params->kmeans_trainset_fraction =
                cagraConfig_.ivf_pq_params->kmeans_trainset_fraction;
        ivf_pq_params->pq_bits = cagraConfig_.ivf_pq_params->pq_bits;
        ivf_pq_params->pq_dim = cagraConfig_.ivf_pq_params->pq_dim;
        ivf_pq_params->codebook_kind =
                static_cast<cuvs::neighbors::ivf_pq::codebook_gen>(
                        cagraConfig_.ivf_pq_params->codebook_kind);
        ivf_pq_params->force_random_rotation =
                cagraConfig_.ivf_pq_params->force_random_rotation;
        ivf_pq_params->conservative_memory_allocation =
                cagraConfig_.ivf_pq_params->conservative_memory_allocation;
    }
    if (cagraConfig_.ivf_pq_search_params != nullptr) {
        ivf_pq_search_params =
                std::make_optional<cuvs::neighbors::ivf_pq::search_params>();
        ivf_pq_search_params->n_probes =
                cagraConfig_.ivf_pq_search_params->n_probes;
        ivf_pq_search_params->lut_dtype =
                cagraConfig_.ivf_pq_search_params->lut_dtype;
        ivf_pq_search_params->preferred_shmem_carveout =
                cagraConfig_.ivf_pq_search_params->preferred_shmem_carveout;
        ivf_pq_search_params->max_internal_batch_size =
                cagraConfig_.ivf_pq_search_params->max_internal_batch_size;
    }

    if (numeric_type == NumericType::Float32) {
        index_ = std::make_shared<CuvsCagra<float>>(
                this->resources_.get(),
                this->d,
                cagraConfig_.intermediate_graph_degree,
                cagraConfig_.graph_degree,
                static_cast<faiss::cagra_build_algo>(cagraConfig_.build_algo),
                cagraConfig_.nn_descent_niter,
                cagraConfig_.store_dataset,
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT,
                ivf_pq_params,
                ivf_pq_search_params,
                cagraConfig_.refine_rate,
                cagraConfig_.guarantee_connectivity);
        std::get<std::shared_ptr<CuvsCagra<float>>>(index_)->train(
                n, static_cast<const float*>(x));
    } else if (numeric_type == NumericType::Float16) {
        index_ = std::make_shared<CuvsCagra<half>>(
                this->resources_.get(),
                this->d,
                cagraConfig_.intermediate_graph_degree,
                cagraConfig_.graph_degree,
                static_cast<faiss::cagra_build_algo>(cagraConfig_.build_algo),
                cagraConfig_.nn_descent_niter,
                cagraConfig_.store_dataset,
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT,
                ivf_pq_params,
                ivf_pq_search_params,
                cagraConfig_.refine_rate,
                cagraConfig_.guarantee_connectivity);
        std::get<std::shared_ptr<CuvsCagra<half>>>(index_)->train(
                n, static_cast<const half*>(x));
    } else if (numeric_type == NumericType::Int8) {
        index_ = std::make_shared<CuvsCagra<int8_t>>(
                this->resources_.get(),
                this->d,
                cagraConfig_.intermediate_graph_degree,
                cagraConfig_.graph_degree,
                static_cast<faiss::cagra_build_algo>(cagraConfig_.build_algo),
                cagraConfig_.nn_descent_niter,
                cagraConfig_.store_dataset,
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT,
                ivf_pq_params,
                ivf_pq_search_params,
                cagraConfig_.refine_rate,
                cagraConfig_.guarantee_connectivity);
        std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)->train(
                n, static_cast<const int8_t*>(x));
    } else {
        FAISS_THROW_MSG("GpuIndexCagra::train unsupported data type");
    }

    this->is_trained = true;
    this->ntotal = n;
}

void GpuIndexCagra::train(idx_t n, const float* x) {
    train_ex(n, static_cast<const void*>(x), NumericType::Float32);
}

void GpuIndexCagra::add_ex(idx_t n, const void* x, NumericType numeric_type) {
    train_ex(n, x, numeric_type);
}

void GpuIndexCagra::add(idx_t n, const float* x) {
    add_ex(n, x, NumericType::Float32);
}

bool GpuIndexCagra::addImplRequiresIDs_() const {
    return false;
};

void GpuIndexCagra::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_MSG("adding vectors is not supported by GpuIndexCagra.");
};

void GpuIndexCagra::addImpl_ex_(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        const idx_t* ids) {
    GpuIndex::addImpl_ex_(n, x, numeric_type, ids);
}

void GpuIndexCagra::searchImpl_ex_(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    FAISS_ASSERT(
            this->is_trained &&
            !std::holds_alternative<std::monostate>(index_));
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT_MSG(
            numeric_type == numeric_type_,
            "Inconsistent numeric type for train and search");

    SearchParametersCagra* params;
    if (search_params) {
        params = dynamic_cast<SearchParametersCagra*>(
                const_cast<SearchParameters*>(search_params));
    } else {
        params = new SearchParametersCagra{};
    }

    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(const_cast<idx_t*>(labels), {n, k});

    if (numeric_type == NumericType::Float32) {
        Tensor<float, 2, true> queries(
                const_cast<float*>(static_cast<const float*>(x)), {n, this->d});

        std::get<std::shared_ptr<CuvsCagra<float>>>(index_)->search(
                queries,
                k,
                outDistances,
                outLabels,
                params->max_queries,
                params->itopk_size,
                params->max_iterations,
                static_cast<faiss::cagra_search_algo>(params->algo),
                params->team_size,
                params->search_width,
                params->min_iterations,
                params->thread_block_size,
                static_cast<faiss::cagra_hash_mode>(params->hashmap_mode),
                params->hashmap_min_bitlen,
                params->hashmap_max_fill_rate,
                params->num_random_samplings,
                params->seed,
                params->sel);

    } else if (numeric_type == NumericType::Float16) {
        Tensor<half, 2, true> queries(
                const_cast<half*>(static_cast<const half*>(x)), {n, this->d});

        std::get<std::shared_ptr<CuvsCagra<half>>>(index_)->search(
                queries,
                k,
                outDistances,
                outLabels,
                params->max_queries,
                params->itopk_size,
                params->max_iterations,
                static_cast<faiss::cagra_search_algo>(params->algo),
                params->team_size,
                params->search_width,
                params->min_iterations,
                params->thread_block_size,
                static_cast<faiss::cagra_hash_mode>(params->hashmap_mode),
                params->hashmap_min_bitlen,
                params->hashmap_max_fill_rate,
                params->num_random_samplings,
                params->seed,
                params->sel);
    } else if (numeric_type == NumericType::Int8) {
        Tensor<int8_t, 2, true> queries(
                const_cast<int8_t*>(static_cast<const int8_t*>(x)),
                {n, this->d});

        std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)->search(
                queries,
                k,
                outDistances,
                outLabels,
                params->max_queries,
                params->itopk_size,
                params->max_iterations,
                static_cast<faiss::cagra_search_algo>(params->algo),
                params->team_size,
                params->search_width,
                params->min_iterations,
                params->thread_block_size,
                static_cast<faiss::cagra_hash_mode>(params->hashmap_mode),
                params->hashmap_min_bitlen,
                params->hashmap_max_fill_rate,
                params->num_random_samplings,
                params->seed,
                params->sel);
    } else {
        FAISS_THROW_MSG("GpuIndexCagra::searchImpl_ unsupported data type");
    }

    if (not search_params) {
        delete params;
    }
}

void GpuIndexCagra::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    searchImpl_ex_(
            n,
            static_cast<const void*>(x),
            NumericType::Float32,
            k,
            distances,
            labels,
            search_params);
}

void GpuIndexCagra::trainAllNeighbors_(idx_t n, const float* x) {
    FAISS_THROW_IF_MSG(is_trained, "index is already trained");

    const auto& devices = cagraConfig_.devices;
    const auto& an_config = cagraConfig_.all_neighbors_params;

    numeric_type_ = NumericType::Float32;
    idx_t graph_degree = static_cast<idx_t>(cagraConfig_.graph_degree);
    idx_t intermediate_degree =
            static_cast<idx_t>(cagraConfig_.intermediate_graph_degree);
    const size_t knn_count = (size_t)n * intermediate_degree;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_phase = t_start;

    // Allocated off the clique so it survives the clique's destruction below.
    CUDA_VERIFY(cudaSetDevice(devices[0]));
    raft::device_resources single_gpu_res;
    auto d_knn = raft::make_device_matrix<uint32_t, int64_t>(
            single_gpu_res, n, intermediate_degree);

    // Build the knn graph across all devices. The clique is scoped: optimize
    // hangs intermittently if multi-GPU state is still alive when it runs.
    {
        raft::device_resources_snmg clique(devices);

        cuvs::neighbors::all_neighbors::all_neighbors_params an_params;
        int num_gpus = static_cast<int>(devices.size());
        an_params.n_clusters = an_config.n_clusters > 0
                ? an_config.n_clusters
                : static_cast<size_t>(std::max(num_gpus * 2, 4));
        an_params.overlap_factor = an_config.overlap_factor;
        an_params.metric = metricFaissToCuvs(this->metric_type, false);

        const char* algo_name = "nn_descent";
        switch (cagraConfig_.build_algo) {
            case graph_build_algo::BRUTE_FORCE: {
                // Exact kNN via tiled GEMM, O(N^2 D) per cluster
                cuvs::neighbors::all_neighbors::graph_build_params::
                        brute_force_params bf_params;
                an_params.graph_build_params = bf_params;
                algo_name = "brute_force";
                break;
            }
            case graph_build_algo::IVF_PQ: {
                // cuVS derives good IVF-PQ params from the dataset shape,
                // so only override what it cannot infer. See
                // AllNeighborsCagraConfig::ivf_pq_size_from_cluster.
                idx_t sizing_rows = n;
                if (an_config.ivf_pq_size_from_cluster) {
                    sizing_rows = std::max<idx_t>(
                            1,
                            (idx_t)((double)n * an_params.overlap_factor /
                                    (double)an_params.n_clusters));
                }
                auto dataset_ext = raft::make_extents<int64_t>(
                        sizing_rows, static_cast<int64_t>(this->d));
                cuvs::neighbors::all_neighbors::graph_build_params::
                        ivf_pq_params ivfpq_params(dataset_ext);
                ivfpq_params.refinement_rate = an_config.refinement_rate;
                // Bounds search memory; batches the search rather than
                // changing its result.
                if (an_config.ivf_pq_search_batch_size > 0) {
                    ivfpq_params.search_params.max_internal_batch_size =
                            an_config.ivf_pq_search_batch_size;
                }
                fprintf(stderr,
                        "  [trainAllNeighbors] IVF-PQ sized from %s "
                        "(%ld rows): n_lists=%u n_probes=%u "
                        "kmeans_trainset_fraction=%.4f refine=%.2f\n",
                        an_config.ivf_pq_size_from_cluster ? "cluster"
                                                           : "full dataset",
                        (long)sizing_rows,
                        ivfpq_params.build_params.n_lists,
                        ivfpq_params.search_params.n_probes,
                        ivfpq_params.build_params.kmeans_trainset_fraction,
                        ivfpq_params.refinement_rate);
                an_params.graph_build_params = ivfpq_params;
                algo_name = "ivf_pq";
                break;
            }
            case graph_build_algo::NN_DESCENT: {
                cuvs::neighbors::all_neighbors::graph_build_params::
                        nn_descent_params nn_params;
                nn_params.graph_degree = intermediate_degree;
                nn_params.max_iterations = cagraConfig_.nn_descent_niter;
                an_params.graph_build_params = nn_params;
                break;
            }
            default:
                FAISS_THROW_MSG(
                        "multi-GPU CAGRA build supports build_algo IVF_PQ, "
                        "NN_DESCENT or BRUTE_FORCE");
        }

        auto dataset = raft::make_host_matrix_view<const float, int64_t>(
                x, n, static_cast<int64_t>(this->d));

        auto d_indices = raft::make_device_matrix<int64_t, int64_t>(
                clique, n, static_cast<int64_t>(intermediate_degree));

        fprintf(stderr,
                "  [trainAllNeighbors] Building kNN graph: n=%ld, "
                "intermediate_degree=%ld, n_clusters=%zu, overlap=%zu, "
                "algo=%s, refinement_rate=%.2f\n",
                (long)n,
                (long)intermediate_degree,
                an_params.n_clusters,
                an_params.overlap_factor,
                algo_name,
                cagraConfig_.build_algo == graph_build_algo::IVF_PQ
                        ? an_config.refinement_rate
                        : 0.0f);

        cuvs::neighbors::all_neighbors::build(
                clique, an_params, dataset, d_indices.view());

        auto t_now = std::chrono::high_resolution_clock::now();
        fprintf(stderr,
                "  [trainAllNeighbors] all_neighbors build: %.2f seconds\n",
                std::chrono::duration<double>(t_now - t_phase).count());
        t_phase = t_now;

        CUDA_VERIFY(cudaSetDevice(devices[0]));
        constexpr int kThreads = 256;
        const int blocks = (int)std::min<size_t>(
                (knn_count + kThreads - 1) / kThreads, 65535);
        kern_narrow_indices<<<blocks, kThreads>>>(
                d_indices.data_handle(), d_knn.data_handle(), knn_count);
        CUDA_VERIFY(cudaGetLastError());
        CUDA_VERIFY(cudaDeviceSynchronize());
    } // clique + d_indices destroyed here, freeing all GPU resources

    auto t_now = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "  [trainAllNeighbors] narrow to uint32 (device): %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_phase).count());
    t_phase = t_now;

    // Phase 3: prune the knn graph into a CAGRA graph.
    const size_t out_count = (size_t)n * graph_degree;
    std::vector<uint32_t> h_cagra(out_count);

#if !defined(FAISS_CAGRA_DEVICE_OPTIMIZE)
    // Public API: host matrices only, so the graph makes a round trip it does
    // not need. Some cuVS versions also read the input as host memory whatever
    // accessor you hand them, so passing device pointers here is not an option.
    auto h_knn =
            raft::make_host_matrix<uint32_t, int64_t>(n, intermediate_degree);
    raft::copy(
            h_knn.data_handle(),
            d_knn.data_handle(),
            knn_count,
            raft::resource::get_cuda_stream(single_gpu_res));
    raft::resource::sync_stream(single_gpu_res);

    auto h_cagra_view = raft::make_host_matrix_view<uint32_t, int64_t>(
            h_cagra.data(), n, graph_degree);
    cuvs::neighbors::cagra::helpers::optimize(
            single_gpu_res, h_knn.view(), h_cagra_view);
    const char* optimize_where = "host (public API)";
#else
    auto d_cagra = raft::make_device_matrix<uint32_t, int64_t>(
            single_gpu_res, n, graph_degree);

    cuvs::neighbors::cagra::detail::graph::optimize<uint32_t>(
            single_gpu_res,
            d_knn.view(),
            d_cagra.view(),
            cagraConfig_.guarantee_connectivity);
    raft::resource::sync_stream(single_gpu_res);
    raft::copy(
            h_cagra.data(),
            d_cagra.data_handle(),
            out_count,
            raft::resource::get_cuda_stream(single_gpu_res));
    raft::resource::sync_stream(single_gpu_res);
    const char* optimize_where = "device-resident";
#endif

    t_now = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "  [trainAllNeighbors] optimize (%s): %.2f seconds\n",
            optimize_where,
            std::chrono::duration<double>(t_now - t_phase).count());
    t_phase = t_now;

    merged_knngraph_.resize(out_count);
    merged_knngraph_degree_ = graph_degree;

#pragma omp parallel for
    for (idx_t i = 0; i < (idx_t)out_count; i++) {
        merged_knngraph_[i] = static_cast<idx_t>(h_cagra[i]);
    }

    multi_gpu_dataset_ = x;
    this->is_trained = true;
    this->ntotal = n;

    t_now = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "  [trainAllNeighbors] D2H + widen: %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_phase).count());
    fprintf(stderr,
            "  [trainAllNeighbors] Total: %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_start).count());
}

void GpuIndexCagra::copyFrom_ex(
        const faiss::IndexHNSWCagra* index,
        NumericType numeric_type) {
    FAISS_ASSERT(index);
    numeric_type_ = numeric_type;

    DeviceScope scope(config_.device);

    GpuIndex::copyFrom(index);

    auto hnsw = index->hnsw;

    // copy level 0 to a dense knn graph matrix
    std::vector<idx_t> knn_graph;
    knn_graph.resize(index->ntotal * hnsw.nb_neighbors(0));

#pragma omp parallel for
    for (size_t i = 0; i < index->ntotal; ++i) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            // knn_graph.push_back(hnsw.neighbors[j]);
            knn_graph[i * hnsw.nb_neighbors(0) + (j - begin)] =
                    hnsw.neighbors[j];
        }
    }

    if (numeric_type == NumericType::Float32) {
        auto base_index = dynamic_cast<IndexFlat*>(index->storage);
        FAISS_ASSERT(base_index);
        auto dataset = base_index->get_xb();
        fprintf(stderr,
                "WARNING: GpuIndexCagra::copyFrom uses non-owning CPU storage. "
                "Keep the source IndexHNSWCagra alive for the lifetime of the "
                "GpuIndexCagra.\n");

        index_ = std::make_shared<CuvsCagra<float>>(
                this->resources_.get(),
                this->d,
                index->ntotal,
                hnsw.nb_neighbors(0),
                dataset,
                knn_graph.data(),
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT);
    } else if (numeric_type == NumericType::Float16) {
        auto base_index = dynamic_cast<IndexScalarQuantizer*>(index->storage);
        FAISS_ASSERT(base_index);
        auto dataset = reinterpret_cast<half*>(base_index->codes.data());
        fprintf(stderr,
                "WARNING: GpuIndexCagra::copyFrom uses non-owning CPU storage. "
                "Keep the source IndexHNSWCagra alive for the lifetime of the "
                "GpuIndexCagra.\n");

        index_ = std::make_shared<CuvsCagra<half>>(
                this->resources_.get(),
                this->d,
                index->ntotal,
                hnsw.nb_neighbors(0),
                dataset,
                knn_graph.data(),
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT);
    } else if (numeric_type == NumericType::Int8) {
        auto base_index = dynamic_cast<IndexScalarQuantizer*>(index->storage);
        FAISS_ASSERT(base_index);
        auto dataset = (uint8_t*)base_index->codes.data();
        fprintf(stderr,
                "WARNING: GpuIndexCagra::copyFrom uses non-owning CPU storage. "
                "Keep the source IndexHNSWCagra alive for the lifetime of the "
                "GpuIndexCagra.\n");

        int8_t* decoded_train_dataset = new int8_t[index->ntotal * index->d];
        for (int i = 0; i < index->ntotal * this->d; i++) {
            decoded_train_dataset[i] = dataset[i] - 128;
        }

        index_ = std::make_shared<CuvsCagra<int8_t>>(
                this->resources_.get(),
                this->d,
                index->ntotal,
                hnsw.nb_neighbors(0),
                decoded_train_dataset,
                knn_graph.data(),
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT);
        delete[] decoded_train_dataset;
    } else {
        FAISS_THROW_MSG("GpuIndexCagra::copyFrom unsupported data type");
    }

    this->is_trained = true;
}

void GpuIndexCagra::copyFrom(const faiss::IndexHNSWCagra* index) {
    copyFrom_ex(index, NumericType::Float32);
}

void GpuIndexCagra::copyTo(faiss::IndexHNSWCagra* index) const {
    if (!merged_knngraph_.empty()) {
        copyToMultiGpu_(index);
        return;
    }
    FAISS_ASSERT(
            !std::holds_alternative<std::monostate>(index_) &&
            this->is_trained && index);
    DeviceScope scope(config_.device);

    //
    // Index information
    //
    GpuIndex::copyTo(index);
    index->hnsw.is_similarity = is_similarity_metric(this->metric_type);
    // This needs to be zeroed out as this implementation adds vectors to the
    // cpuIndex instead of copying fields
    index->ntotal = 0;
    index->set_numeric_type(numeric_type_);

    idx_t graph_degree;

    if (numeric_type_ == NumericType::Float32) {
        graph_degree = std::get<std::shared_ptr<CuvsCagra<float>>>(index_)
                               ->get_knngraph_degree();
    } else if (numeric_type_ == NumericType::Float16) {
        graph_degree = std::get<std::shared_ptr<CuvsCagra<half>>>(index_)
                               ->get_knngraph_degree();
    } else if (numeric_type_ == NumericType::Int8) {
        graph_degree = std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)
                               ->get_knngraph_degree();
    } else {
        FAISS_THROW_MSG("GpuIndexCagra::copyTo unsupported data type");
    }

    auto M = graph_degree / 2;
    if (index->storage and index->own_fields) {
        delete index->storage;
    }

    // storage depends on numerictype
    if (numeric_type_ == NumericType::Float32) {
        if (this->metric_type == METRIC_L2) {
            index->storage = new IndexFlatL2(index->d);
        } else if (this->metric_type == METRIC_INNER_PRODUCT) {
            index->storage = new IndexFlatIP(index->d);
        }
    } else if (numeric_type_ == NumericType::Float16) {
        auto qtype = ScalarQuantizer::QT_fp16;
        index->storage =
                new IndexScalarQuantizer(index->d, qtype, this->metric_type);
    } else if (numeric_type_ == NumericType::Int8) {
        auto qtype = ScalarQuantizer::QT_8bit_direct_signed;
        index->storage =
                new IndexScalarQuantizer(index->d, qtype, this->metric_type);
    }

    index->own_fields = true;
    index->keep_max_size_level0 = true;
    index->hnsw.reset();
    index->hnsw.assign_probas.clear();
    index->hnsw.cum_nneighbor_per_level.clear();
    index->hnsw.set_default_probas(M, 1.0 / log(M));

    auto n_train = this->ntotal;
    bool allocation = false;

    if (numeric_type_ == NumericType::Float32) {
        float* train_dataset;
        const float* dataset =
                std::get<std::shared_ptr<CuvsCagra<float>>>(index_)
                        ->get_training_dataset();
        if (getDeviceForAddress(dataset) >= 0) {
            train_dataset = new float[n_train * index->d];
            allocation = true;
            raft::copy(
                    train_dataset,
                    dataset,
                    n_train * index->d,
                    this->resources_->getRaftHandleCurrentDevice()
                            .get_stream());
        } else {
            train_dataset = const_cast<float*>(dataset);
        }

        // turn off as level 0 is copied from CAGRA graph
        index->init_level0 = false;
        if (!index->base_level_only) {
            index->add(n_train, train_dataset);
        } else {
            index->hnsw.prepare_level_tab(n_train, false);
            index->storage->add(n_train, train_dataset);
            index->ntotal = n_train;
        }
        if (allocation) {
            delete[] train_dataset;
        }
    } else if (numeric_type_ == NumericType::Float16) {
        half* train_dataset;
        const half* dataset = std::get<std::shared_ptr<CuvsCagra<half>>>(index_)
                                      ->get_training_dataset();
        if (getDeviceForAddress(dataset) >= 0) {
            train_dataset = new half[n_train * index->d];
            allocation = true;
            raft::copy(
                    train_dataset,
                    dataset,
                    n_train * index->d,
                    this->resources_->getRaftHandleCurrentDevice()
                            .get_stream());
        } else {
            train_dataset = const_cast<half*>(dataset);
        }

        index->init_level0 = false;
        if (!index->base_level_only) {
            FAISS_THROW_MSG(
                    "Only base level copy is supported for FP16 types in GpuIndexCagra::copyTo");
        } else {
            index->hnsw.prepare_level_tab(n_train, false);
            index->storage->add_sa_codes(
                    n_train, (uint8_t*)train_dataset, nullptr);
            index->ntotal = n_train;
        }

        if (allocation) {
            delete[] train_dataset;
        }
    } else if (numeric_type_ == NumericType::Int8) {
        int8_t* train_dataset;
        const int8_t* dataset =
                std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)
                        ->get_training_dataset();
        if (getDeviceForAddress(dataset) >= 0) {
            train_dataset = new int8_t[n_train * index->d];
            allocation = true;
            raft::copy(
                    train_dataset,
                    dataset,
                    n_train * index->d,
                    this->resources_->getRaftHandleCurrentDevice()
                            .get_stream());
        } else {
            train_dataset = const_cast<int8_t*>(dataset);
        }

        index->init_level0 = false;
        if (!index->base_level_only) {
            FAISS_THROW_MSG(
                    "Only base level copy is supported for Int8 types in GpuIndexCagra::copyTo");
        } else {
            index->hnsw.prepare_level_tab(n_train, false);
            // Directly update train_dataset with encoding of
            // Quantizer8bitDirectSigned
            for (int64_t i = 0; i < ((int64_t)n_train) * index->d; ++i) {
                train_dataset[i] = static_cast<uint8_t>(
                        static_cast<int>(train_dataset[i]) + 128);
            }

            index->storage->add_sa_codes(
                    n_train,
                    reinterpret_cast<uint8_t*>(train_dataset),
                    nullptr);

            index->ntotal = n_train;
        }

        if (allocation) {
            delete[] train_dataset;
        } else {
            // Recover after appending
            for (int64_t i = 0; i < ((int64_t)n_train) * index->d; ++i) {
                train_dataset[i] = static_cast<int8_t>(
                        static_cast<int>(train_dataset[i]) - 128);
            }
        }
    }

    auto graph = get_knngraph();

#pragma omp parallel for
    for (idx_t i = 0; i < n_train; i++) {
        size_t begin, end;
        index->hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            index->hnsw.neighbors[j] = graph[i * graph_degree + (j - begin)];
        }
    }

    // turn back on to allow new vectors to be added to level 0
    index->init_level0 = true;
}

void GpuIndexCagra::buildHnswUpperLevelsGpu_(
        faiss::IndexHNSWCagra* index,
        int max_lvl) const {
    auto& hnsw = index->hnsw;
    const idx_t n_train = this->ntotal;
    // Levels >= 1 all carry M neighbours; level 0 carries 2*M.
    const int upper_degree = (int)hnsw.nb_neighbors(1);

    // BRUTE_FORCE only exists on the multi-GPU path.
    auto sub_algo = cagraConfig_.build_algo == graph_build_algo::IVF_PQ
            ? faiss::cagra_build_algo::IVF_PQ
            : faiss::cagra_build_algo::NN_DESCENT;

    const int sub_intermediate = cagraConfig_.gpu_hnsw_intermediate_degree > 0
            ? (int)cagraConfig_.gpu_hnsw_intermediate_degree
            : 2 * upper_degree;

    // Search starts at entry_point and immediately reads its neighbours at
    // max_level, so the two must agree. Track the highest populated level
    // rather than assuming max_lvl has any nodes.
    int top_lvl = 0;
    HNSW::storage_idx_t top_entry = 0;

    std::vector<idx_t> ids;
    std::vector<float> sub_x;
    for (int lvl = 1; lvl <= max_lvl; lvl++) {
        ids.clear();
        for (idx_t i = 0; i < n_train; i++) {
            if (hnsw.levels[i] - 1 >= lvl) {
                ids.push_back(i);
            }
        }
        const idx_t n_lvl = (idx_t)ids.size();
        if (n_lvl < 1) {
            continue;
        }
        top_lvl = lvl;
        top_entry = static_cast<HNSW::storage_idx_t>(ids[0]);
        if (n_lvl == 1) {
            // Single node: no edges to add, but it is still a valid entry.
            continue;
        }

        if (n_lvl <= upper_degree + 1) {
            // Too small for a meaningful proximity graph: connect all-to-all.
            for (idx_t a = 0; a < n_lvl; a++) {
                size_t begin, end;
                hnsw.neighbor_range(ids[a], lvl, &begin, &end);
                size_t slot = begin;
                for (idx_t b = 0; b < n_lvl && slot < end; b++) {
                    if (b != a) {
                        hnsw.neighbors[slot++] = ids[b];
                    }
                }
            }
            continue;
        }

        sub_x.resize((size_t)n_lvl * this->d);
#pragma omp parallel for
        for (idx_t a = 0; a < n_lvl; a++) {
            memcpy(sub_x.data() + (size_t)a * this->d,
                   multi_gpu_dataset_ + (size_t)ids[a] * this->d,
                   this->d * sizeof(float));
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<idx_t> sub_graph;
        {
            CuvsCagra<float> sub(
                    this->resources_.get(),
                    this->d,
                    sub_intermediate,
                    upper_degree,
                    sub_algo,
                    cagraConfig_.nn_descent_niter,
                    /*store_dataset=*/false,
                    this->metric_type,
                    this->metric_arg,
                    INDICES_64_BIT,
                    std::nullopt,
                    std::nullopt,
                    cagraConfig_.refine_rate,
                    cagraConfig_.gpu_hnsw_guarantee_connectivity);
            sub.train(n_lvl, sub_x.data());
            sub_graph = sub.get_knngraph();
        }

        // Map subgraph-local ids back to global ids.
#pragma omp parallel for
        for (idx_t a = 0; a < n_lvl; a++) {
            size_t begin, end;
            hnsw.neighbor_range(ids[a], lvl, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                idx_t local = sub_graph[(size_t)a * upper_degree + (j - begin)];
                hnsw.neighbors[j] =
                        (local >= 0 && local < n_lvl) ? ids[local] : -1;
            }
        }

        fprintf(stderr,
                "  [copyTo] level %d: %ld nodes, igd=%d gc=%d, "
                "CAGRA subgraph %.2f s\n",
                lvl,
                (long)n_lvl,
                sub_intermediate,
                (int)cagraConfig_.gpu_hnsw_guarantee_connectivity,
                std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t0)
                        .count());
    }
    hnsw.entry_point = top_entry;
    hnsw.max_level = top_lvl;
}

void GpuIndexCagra::copyToMultiGpu_(faiss::IndexHNSWCagra* index) const {
    FAISS_ASSERT(this->is_trained && index && !merged_knngraph_.empty());
    FAISS_THROW_IF_NOT_MSG(
            numeric_type_ == NumericType::Float32,
            "Multi-GPU copyTo only supports Float32");

    GpuIndex::copyTo(index);
    index->ntotal = 0;
    index->set_numeric_type(numeric_type_);

    auto graph_degree = merged_knngraph_degree_;
    auto M = graph_degree / 2;

    if (index->storage && index->own_fields) {
        delete index->storage;
        index->storage = nullptr;
    }
    if (this->metric_type == METRIC_L2) {
        index->storage = new IndexFlatL2(index->d);
    } else if (this->metric_type == METRIC_INNER_PRODUCT) {
        index->storage = new IndexFlatIP(index->d);
    } else {
        FAISS_THROW_MSG(
                "Multi-GPU copyTo only supports METRIC_L2 and METRIC_INNER_PRODUCT");
    }
    index->own_fields = true;
    index->keep_max_size_level0 = true;
    index->hnsw.reset();
    index->hnsw.assign_probas.clear();
    index->hnsw.cum_nneighbor_per_level.clear();
    index->hnsw.set_default_probas(M, 1.0 / log(M));

    auto n_train = this->ntotal;

    index->init_level0 = false;
    if (index->base_level_only) {
        index->hnsw.prepare_level_tab(n_train, false);
        index->storage->add(n_train, multi_gpu_dataset_);
        index->ntotal = n_train;
    } else if (cagraConfig_.gpu_hnsw_upper_levels) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int max_lvl = index->hnsw.prepare_level_tab(n_train, false);
        index->storage->add(n_train, multi_gpu_dataset_);
        index->ntotal = n_train;
        buildHnswUpperLevelsGpu_(index, max_lvl);
        fprintf(stderr,
                "  [copyTo] GPU upper levels: %.2f seconds\n",
                std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t0)
                        .count());
    } else {
        index->add(n_train, multi_gpu_dataset_);
    }

#pragma omp parallel for
    for (idx_t i = 0; i < n_train; i++) {
        size_t begin, end;
        index->hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            index->hnsw.neighbors[j] =
                    merged_knngraph_[i * graph_degree + (j - begin)];
        }
    }

    index->init_level0 = true;
}

void GpuIndexCagra::reset() {
    DeviceScope scope(config_.device);

    bool had_multi_gpu = !merged_knngraph_.empty();
    merged_knngraph_.clear();
    merged_knngraph_degree_ = 0;
    multi_gpu_dataset_ = nullptr;

    if (!std::holds_alternative<std::monostate>(index_)) {
        std::visit(
                [](auto& index_ptr) {
                    using IndexPtrT = std::decay_t<decltype(index_ptr)>;
                    if constexpr (std::is_same_v<IndexPtrT, std::monostate>) {
                        FAISS_THROW_MSG(
                                "CuvsCagra not initialized when calling GpuIndexCagra::reset");
                    } else {
                        return index_ptr->reset();
                    }
                },
                index_);
        this->ntotal = 0;
        this->is_trained = false;
    } else if (had_multi_gpu) {
        this->ntotal = 0;
        this->is_trained = false;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

std::vector<idx_t> GpuIndexCagra::get_knngraph() const {
    FAISS_ASSERT(this->is_trained);

    if (!merged_knngraph_.empty()) {
        return merged_knngraph_;
    }

    FAISS_ASSERT(!std::holds_alternative<std::monostate>(index_));
    return std::visit(
            [](auto&& index_ptr) -> std::vector<idx_t> {
                using IndexPtrT = std::decay_t<decltype(index_ptr)>;

                if constexpr (std::is_same_v<IndexPtrT, std::monostate>) {
                    FAISS_THROW_MSG(
                            "CuvsCagra not initialized when calling GpuIndexCagra::get_knngraph");
                } else {
                    return index_ptr->get_knngraph();
                }
            },
            index_);
}

faiss::NumericType GpuIndexCagra::get_numeric_type() const {
    return numeric_type_;
}

} // namespace gpu
} // namespace faiss
