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
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <faiss/gpu/impl/CuvsCagra.cuh>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>

#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/cagra_optimize.hpp>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace {

template <int MAX_DEGREE>
__global__ void kern_detour_count(
        const uint32_t* __restrict__ knn_graph,
        const uint32_t graph_size,
        const uint32_t graph_degree,
        const uint32_t output_degree,
        const uint32_t batch_size,
        const uint32_t batch_id,
        uint8_t* __restrict__ detour_count) {
    __shared__ uint32_t smem[MAX_DEGREE];

    const uint64_t iA = blockIdx.x + (uint64_t)batch_size * batch_id;
    if (iA >= graph_size)
        return;

    for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
        smem[k] = 0;
        if (knn_graph[k + (uint64_t)graph_degree * iA] == (uint32_t)iA)
            smem[k] = graph_degree;
    }
    __syncthreads();

    for (uint32_t kAD = 0; kAD < graph_degree - 1; kAD++) {
        const uint64_t iD = knn_graph[kAD + (uint64_t)graph_degree * iA];
        if (iD >= graph_size)
            continue;
        for (uint32_t kDB = threadIdx.x; kDB < graph_degree;
             kDB += blockDim.x) {
            const uint64_t iB_cand =
                    knn_graph[kDB + (uint64_t)graph_degree * iD];
            for (uint32_t kAB = kAD + 1; kAB < graph_degree; kAB++) {
                const uint64_t iB =
                        knn_graph[kAB + (uint64_t)graph_degree * iA];
                if (iB == iB_cand) {
                    atomicAdd(smem + kAB, 1);
                    break;
                }
            }
        }
        __syncthreads();
    }

    for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
        detour_count[k + (uint64_t)graph_degree * iA] =
                static_cast<uint8_t>(min(smem[k], 255u));
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

void GpuIndexCagra::trainMultiGpu(
        idx_t n,
        const float* x,
        std::vector<GpuResourcesProvider*>& providers,
        std::vector<int>& devices,
        idx_t stitch_per_shard,
        int stitch_k,
        int stitch_mode) {
    FAISS_THROW_IF_NOT_MSG(!is_trained, "index is already trained");
    FAISS_THROW_IF_NOT_MSG(!devices.empty(), "must provide at least one GPU");
    FAISS_THROW_IF_NOT_MSG(stitch_k >= 1, "stitch_k must be >= 1");

    numeric_type_ = NumericType::Float32;
    int num_shards = static_cast<int>(devices.size());
    idx_t shard_size = (n + num_shards - 1) / num_shards;

    int total_cross = (num_shards > 1) ? stitch_k * (num_shards - 1) : 0;

    auto t_phase = std::chrono::high_resolution_clock::now();
    auto t_start = t_phase;

    // Use cuVS native multi-GPU CAGRA build via SNMG
    raft::device_resources_snmg clique(devices);

    cuvs::neighbors::mg_index_params<cuvs::neighbors::cagra::index_params>
            mg_params;
    mg_params.mode = cuvs::neighbors::SHARDED;
    mg_params.intermediate_graph_degree =
            cagraConfig_.intermediate_graph_degree;
    mg_params.graph_degree = cagraConfig_.graph_degree;
    mg_params.guarantee_connectivity = cagraConfig_.guarantee_connectivity;
    mg_params.metric = metricFaissToCuvs(this->metric_type, false);

    if (cagraConfig_.build_algo == graph_build_algo::IVF_PQ) {
        cuvs::neighbors::cagra::graph_build_params::ivf_pq_params
                graph_build_params;
        if (cagraConfig_.ivf_pq_params) {
            graph_build_params.build_params.n_lists =
                    cagraConfig_.ivf_pq_params->n_lists;
            graph_build_params.build_params.kmeans_n_iters =
                    cagraConfig_.ivf_pq_params->kmeans_n_iters;
            graph_build_params.build_params.kmeans_trainset_fraction =
                    cagraConfig_.ivf_pq_params->kmeans_trainset_fraction;
            graph_build_params.build_params.pq_bits =
                    cagraConfig_.ivf_pq_params->pq_bits;
            graph_build_params.build_params.pq_dim =
                    cagraConfig_.ivf_pq_params->pq_dim;
            graph_build_params.build_params.codebook_kind =
                    static_cast<cuvs::neighbors::ivf_pq::codebook_gen>(
                            cagraConfig_.ivf_pq_params->codebook_kind);
            graph_build_params.build_params.force_random_rotation =
                    cagraConfig_.ivf_pq_params->force_random_rotation;
            graph_build_params.build_params.conservative_memory_allocation =
                    cagraConfig_.ivf_pq_params->conservative_memory_allocation;
        }
        if (cagraConfig_.ivf_pq_search_params) {
            graph_build_params.search_params.n_probes =
                    cagraConfig_.ivf_pq_search_params->n_probes;
            graph_build_params.search_params.lut_dtype =
                    cagraConfig_.ivf_pq_search_params->lut_dtype;
            graph_build_params.search_params.preferred_shmem_carveout =
                    cagraConfig_.ivf_pq_search_params->preferred_shmem_carveout;
        }
        graph_build_params.build_params.metric =
                metricFaissToCuvs(this->metric_type, false);
        graph_build_params.refinement_rate = cagraConfig_.refine_rate;
        mg_params.graph_build_params = graph_build_params;
        if (mg_params.graph_degree == mg_params.intermediate_graph_degree) {
            mg_params.intermediate_graph_degree = 1.5 * mg_params.graph_degree;
        }
    } else {
        cuvs::neighbors::cagra::graph_build_params::nn_descent_params
                graph_build_params(mg_params.intermediate_graph_degree);
        graph_build_params.max_iterations = cagraConfig_.nn_descent_niter;
        graph_build_params.metric = metricFaissToCuvs(this->metric_type, false);
        mg_params.graph_build_params = graph_build_params;
    }

    auto dataset = raft::make_host_matrix_view<const float, int64_t>(
            x, n, static_cast<int64_t>(this->d));

    auto mg_idx = cuvs::neighbors::cagra::build(clique, mg_params, dataset);

    auto t_now = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "  [trainMultiGpu] SNMG CAGRA build: %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_phase).count());
    t_phase = t_now;

    // Extract per-shard graphs into expanded layout: [CAGRA neighbors |
    // cross-shard slots]
    idx_t graph_degree = static_cast<idx_t>(cagraConfig_.graph_degree);
    idx_t expanded_degree = graph_degree + total_cross;
    merged_knngraph_.assign(n * expanded_degree, -1);

    for (int s = 0; s < num_shards; s++) {
        idx_t offset = static_cast<idx_t>(s) * shard_size;
        idx_t actual_size = std::min(shard_size, n - offset);

        auto& shard_idx = mg_idx.ann_interfaces_[s].index_.value();
        auto device_graph = shard_idx.graph();

        const auto& dev_res =
                raft::resource::set_current_device_to_rank(clique, s);
        FAISS_THROW_IF_NOT_FMT(
                device_graph.extent(1) == graph_degree,
                "Shard %d has graph_degree %ld, expected %ld",
                s,
                (long)device_graph.extent(1),
                (long)graph_degree);

        std::vector<uint32_t> host_graph(actual_size * graph_degree);
        raft::resource::sync_stream(dev_res);
        thrust::copy(
                thrust::device_ptr<const uint32_t>(device_graph.data_handle()),
                thrust::device_ptr<const uint32_t>(
                        device_graph.data_handle() +
                        actual_size * graph_degree),
                host_graph.data());

#pragma omp parallel for
        for (idx_t i = 0; i < actual_size; i++) {
            for (idx_t j = 0; j < graph_degree; j++) {
                merged_knngraph_[(offset + i) * expanded_degree + j] =
                        static_cast<idx_t>(host_graph[i * graph_degree + j]) +
                        offset;
            }
        }
    }

    merged_knngraph_degree_ = expanded_degree;

    t_now = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "  [trainMultiGpu] Graph extract + merge: %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_phase).count());
    t_phase = t_now;

    // Cross-shard stitching: appends cross-shard edges after CAGRA neighbors.
    // Mode 0: CPU HNSW search (approximate). Mode 1: GPU brute-force (exact).
    if (num_shards > 1) {
        if (stitch_mode == 1 && stitch_per_shard == 0) {
            stitch_per_shard = 100000;
            fprintf(stderr,
                    "  [trainMultiGpu] GPU brute-force mode: defaulting "
                    "stitch_per_shard to %ld (all-vectors is O(N^2))\n",
                    (long)stitch_per_shard);
        }

        fprintf(stderr,
                "  [trainMultiGpu] Stitching: mode=%s, stitch_per_shard=%ld, "
                "stitch_k=%d, total_cross=%d, expanded_degree=%ld\n",
                stitch_mode == 1 ? "GPU-brute-force" : "CPU-HNSW",
                (long)stitch_per_shard,
                stitch_k,
                total_cross,
                (long)expanded_degree);

        if (stitch_mode == 1) {
            // GPU brute-force stitching: exact cross-shard neighbors
            for (int j = 0; j < num_shards; j++) {
                idx_t j_offset = static_cast<idx_t>(j) * shard_size;
                idx_t j_size = std::min(shard_size, n - j_offset);
                if (j_size == 0)
                    continue;

                GpuIndexFlatConfig flat_config;
                flat_config.device = devices[j];
                GpuIndexFlatL2 flat_idx(providers[j], this->d, flat_config);
                flat_idx.add(j_size, x + j_offset * this->d);

                for (int i = 0; i < num_shards; i++) {
                    if (i == j)
                        continue;
                    idx_t i_offset = static_cast<idx_t>(i) * shard_size;
                    idx_t i_size = std::min(shard_size, n - i_offset);
                    if (i_size == 0)
                        continue;

                    int target_pos = (j < i) ? j : j - 1;
                    idx_t slot_base = graph_degree +
                            static_cast<idx_t>(target_pos) * stitch_k;

                    idx_t num_queries = i_size;
                    const float* query_data = x + i_offset * this->d;
                    std::vector<idx_t> sampled_indices;
                    std::vector<float> sampled_vectors;

                    if (stitch_per_shard > 0 && stitch_per_shard < i_size) {
                        num_queries = stitch_per_shard;
                        sampled_indices.resize(i_size);
                        std::iota(
                                sampled_indices.begin(),
                                sampled_indices.end(),
                                idx_t(0));
                        std::mt19937 rng(42 + i * num_shards + j);
                        for (idx_t s = 0; s < stitch_per_shard; s++) {
                            std::uniform_int_distribution<idx_t> dist(
                                    s, i_size - 1);
                            std::swap(
                                    sampled_indices[s],
                                    sampled_indices[dist(rng)]);
                        }
                        sampled_indices.resize(stitch_per_shard);

                        sampled_vectors.resize(stitch_per_shard * this->d);
#pragma omp parallel for
                        for (idx_t s = 0; s < stitch_per_shard; s++) {
                            memcpy(sampled_vectors.data() + s * this->d,
                                   x +
                                           (i_offset + sampled_indices[s]) *
                                                   this->d,
                                   this->d * sizeof(float));
                        }
                        query_data = sampled_vectors.data();
                    }

                    std::vector<float> distances(num_queries * stitch_k);
                    std::vector<idx_t> labels(num_queries * stitch_k);
                    flat_idx.search(
                            num_queries,
                            query_data,
                            stitch_k,
                            distances.data(),
                            labels.data());

#pragma omp parallel for
                    for (idx_t s = 0; s < num_queries; s++) {
                        idx_t orig_idx = (!sampled_indices.empty())
                                ? sampled_indices[s]
                                : s;
                        idx_t global_id = i_offset + orig_idx;
                        for (int k = 0; k < stitch_k; k++) {
                            idx_t local_nb = labels[s * stitch_k + k];
                            merged_knngraph_
                                    [global_id * expanded_degree + slot_base +
                                     k] = (local_nb >= 0)
                                    ? (local_nb + j_offset)
                                    : local_nb;
                        }
                    }
                }
                fprintf(stderr,
                        "    Shard %d/%d GPU-stitched\n",
                        j + 1,
                        num_shards);
            }
        } else {
            // CPU HNSW stitching (Approach C)
            auto M_temp = graph_degree / 2;
            for (int j = 0; j < num_shards; j++) {
                idx_t j_offset = static_cast<idx_t>(j) * shard_size;
                idx_t j_size = std::min(shard_size, n - j_offset);
                if (j_size == 0)
                    continue;

                IndexHNSWCagra temp_idx;
                temp_idx.d = this->d;
                temp_idx.metric_type = this->metric_type;
                temp_idx.base_level_only = true;
                temp_idx.num_base_level_search_entrypoints = 32;
                if (this->metric_type == METRIC_L2) {
                    temp_idx.storage = new IndexFlatL2(this->d);
                } else {
                    temp_idx.storage = new IndexFlatIP(this->d);
                }
                temp_idx.own_fields = true;
                temp_idx.keep_max_size_level0 = true;
                temp_idx.hnsw.reset();
                temp_idx.hnsw.assign_probas.clear();
                temp_idx.hnsw.cum_nneighbor_per_level.clear();
                temp_idx.hnsw.set_default_probas(M_temp, 1.0 / log(M_temp));
                temp_idx.init_level0 = false;
                temp_idx.hnsw.prepare_level_tab(j_size, false);
                temp_idx.storage->add(j_size, x + j_offset * this->d);
                temp_idx.ntotal = j_size;

#pragma omp parallel for
                for (idx_t i = 0; i < j_size; i++) {
                    size_t begin, end;
                    temp_idx.hnsw.neighbor_range(i, 0, &begin, &end);
                    for (size_t k = begin; k < end; k++) {
                        idx_t global_nb = merged_knngraph_
                                [(j_offset + i) * expanded_degree +
                                 (k - begin)];
                        temp_idx.hnsw.neighbors[k] = global_nb - j_offset;
                    }
                }

                temp_idx.hnsw.efSearch = 64;

                for (int i = 0; i < num_shards; i++) {
                    if (i == j)
                        continue;
                    idx_t i_offset = static_cast<idx_t>(i) * shard_size;
                    idx_t i_size = std::min(shard_size, n - i_offset);
                    if (i_size == 0)
                        continue;

                    int target_pos = (j < i) ? j : j - 1;
                    idx_t slot_base = graph_degree +
                            static_cast<idx_t>(target_pos) * stitch_k;

                    idx_t num_queries = i_size;
                    const float* query_data = x + i_offset * this->d;
                    std::vector<idx_t> sampled_indices;
                    std::vector<float> sampled_vectors;

                    if (stitch_per_shard > 0 && stitch_per_shard < i_size) {
                        num_queries = stitch_per_shard;
                        sampled_indices.resize(i_size);
                        std::iota(
                                sampled_indices.begin(),
                                sampled_indices.end(),
                                idx_t(0));
                        std::mt19937 rng(42 + i * num_shards + j);
                        for (idx_t s = 0; s < stitch_per_shard; s++) {
                            std::uniform_int_distribution<idx_t> dist(
                                    s, i_size - 1);
                            std::swap(
                                    sampled_indices[s],
                                    sampled_indices[dist(rng)]);
                        }
                        sampled_indices.resize(stitch_per_shard);

                        sampled_vectors.resize(stitch_per_shard * this->d);
#pragma omp parallel for
                        for (idx_t s = 0; s < stitch_per_shard; s++) {
                            memcpy(sampled_vectors.data() + s * this->d,
                                   x +
                                           (i_offset + sampled_indices[s]) *
                                                   this->d,
                                   this->d * sizeof(float));
                        }
                        query_data = sampled_vectors.data();
                    }

                    std::vector<float> distances(num_queries * stitch_k);
                    std::vector<idx_t> labels(num_queries * stitch_k);
                    temp_idx.search(
                            num_queries,
                            query_data,
                            stitch_k,
                            distances.data(),
                            labels.data());

#pragma omp parallel for
                    for (idx_t s = 0; s < num_queries; s++) {
                        idx_t orig_idx = (!sampled_indices.empty())
                                ? sampled_indices[s]
                                : s;
                        idx_t global_id = i_offset + orig_idx;
                        for (int k = 0; k < stitch_k; k++) {
                            idx_t local_nb = labels[s * stitch_k + k];
                            merged_knngraph_
                                    [global_id * expanded_degree + slot_base +
                                     k] = (local_nb >= 0)
                                    ? (local_nb + j_offset)
                                    : local_nb;
                        }
                    }
                }
                fprintf(stderr,
                        "    Shard %d/%d stitched\n",
                        j + 1,
                        num_shards);
            }
        } // end else (CPU HNSW)
    }

    t_now = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "  [trainMultiGpu] Stitching: %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_phase).count());
    fprintf(stderr,
            "  [trainMultiGpu] Total: %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_start).count());

    multi_gpu_dataset_ = x;
    this->is_trained = true;
    this->ntotal = n;
}

void GpuIndexCagra::trainAllNeighbors(
        idx_t n,
        const float* x,
        std::vector<int>& devices,
        int n_clusters_override,
        int overlap_factor_override,
        bool multi_gpu_optimize,
        int build_algo,
        float refinement_rate,
        int ivfpq_search_batch) {
    FAISS_THROW_IF_NOT_MSG(!is_trained, "index is already trained");
    FAISS_THROW_IF_NOT_MSG(!devices.empty(), "must provide at least one GPU");

    numeric_type_ = NumericType::Float32;
    idx_t graph_degree = static_cast<idx_t>(cagraConfig_.graph_degree);
    idx_t intermediate_degree =
            static_cast<idx_t>(cagraConfig_.intermediate_graph_degree);

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_phase = t_start;

    // Phase 1+2: Build kNN graph (multi-GPU) then D2H + cast.
    // clique and d_indices are scoped so SNMG resources are fully destroyed
    // before Phase 3, avoiding CUDA context interference with single-GPU
    // optimize.
    auto h_knn =
            raft::make_host_matrix<uint32_t, int64_t>(n, intermediate_degree);
    {
        raft::device_resources_snmg clique(devices);

        cuvs::neighbors::all_neighbors::all_neighbors_params an_params;
        int num_gpus = static_cast<int>(devices.size());
        an_params.n_clusters = n_clusters_override > 0
                ? static_cast<size_t>(n_clusters_override)
                : std::max(num_gpus * 2, 4);
        an_params.overlap_factor = overlap_factor_override > 0
                ? static_cast<size_t>(overlap_factor_override)
                : 2;
        an_params.metric = metricFaissToCuvs(this->metric_type, false);

        const char* algo_name = "nn_descent";
        if (build_algo == 1) {
            // Brute-force: exact kNN via tiled GEMM, O(N²D) per cluster
            cuvs::neighbors::all_neighbors::graph_build_params::
                    brute_force_params bf_params;
            an_params.graph_build_params = bf_params;
            algo_name = "brute_force";
        } else if (build_algo == 2) {
            // IVF-PQ: approximate kNN with refinement
            auto dataset_ext = raft::make_extents<int64_t>(
                    n, static_cast<int64_t>(this->d));
            cuvs::neighbors::all_neighbors::graph_build_params::ivf_pq_params
                    ivfpq_params(dataset_ext);
            ivfpq_params.refinement_rate = refinement_rate;
            // Bound the IVF-PQ search workspace to avoid GPU OOM at scale. The
            // cuVS default max_internal_batch_size is 128*1024, whose search
            // buffers can exceed H100 memory for dense 100M clusters. Capping
            // it batches the search into smaller query chunks (results
            // unchanged).
            if (ivfpq_search_batch > 0) {
                ivfpq_params.search_params.max_internal_batch_size =
                        static_cast<uint32_t>(ivfpq_search_batch);
            }
            an_params.graph_build_params = ivfpq_params;
            algo_name = "ivf_pq";
        } else {
            // Default: NN-descent
            cuvs::neighbors::all_neighbors::graph_build_params::
                    nn_descent_params nn_params;
            nn_params.graph_degree = intermediate_degree;
            nn_params.max_iterations = cagraConfig_.nn_descent_niter;
            an_params.graph_build_params = nn_params;
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
                build_algo == 2 ? refinement_rate : 0.0f);

        cuvs::neighbors::all_neighbors::build(
                clique, an_params, dataset, d_indices.view());

        auto t_now = std::chrono::high_resolution_clock::now();
        fprintf(stderr,
                "  [trainAllNeighbors] all_neighbors build: %.2f seconds\n",
                std::chrono::duration<double>(t_now - t_phase).count());
        t_phase = t_now;

        // D2H + cast int64 -> uint32
        std::vector<int64_t> h_indices_i64(n * intermediate_degree);
        raft::copy(
                h_indices_i64.data(),
                d_indices.data_handle(),
                n * intermediate_degree,
                raft::resource::get_cuda_stream(clique));
        raft::resource::sync_stream(clique);

#pragma omp parallel for
        for (idx_t i = 0; i < n * intermediate_degree; i++) {
            h_knn.data_handle()[i] = static_cast<uint32_t>(h_indices_i64[i]);
        }
    } // clique + d_indices destroyed here, freeing all GPU resources

    auto t_now = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "  [trainAllNeighbors] D2H + cast: %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_phase).count());
    t_phase = t_now;

    // Phase 3: Optimize kNN graph into CAGRA graph
    auto h_cagra = raft::make_host_matrix<uint32_t, int64_t>(n, graph_degree);
    memset(h_cagra.data_handle(), 0xff, n * graph_degree * sizeof(uint32_t));

    if (multi_gpu_optimize && devices.size() > 1) {
        // S2: Multi-GPU detour counting + CPU pruning/reverse/merge
        int num_gpus = static_cast<int>(devices.size());
        fprintf(stderr,
                "  [trainAllNeighbors] Multi-GPU optimize: %d GPUs, "
                "n=%ld, %ld->%ld\n",
                num_gpus,
                (long)n,
                (long)intermediate_degree,
                (long)graph_degree);

        // Phase 3a: Multi-GPU detour counting
        FAISS_THROW_IF_NOT_MSG(
                intermediate_degree <= 1024,
                "intermediate_graph_degree must be <= 1024 for "
                "multi-GPU optimize");

        auto h_detour = raft::make_host_matrix<uint8_t, int64_t>(
                n, intermediate_degree);
        memset(h_detour.data_handle(),
               0xff,
               n * intermediate_degree * sizeof(uint8_t));

        idx_t chunk = (n + num_gpus - 1) / num_gpus;

#pragma omp parallel for num_threads(num_gpus)
        for (int g = 0; g < num_gpus; g++) {
            idx_t my_start = g * chunk;
            idx_t my_end = std::min(my_start + chunk, n);
            idx_t my_n = my_end - my_start;
            if (my_n <= 0)
                continue;

            CUDA_VERIFY(cudaSetDevice(devices[g]));
            cudaStream_t stream;
            CUDA_VERIFY(cudaStreamCreate(&stream));

            uint32_t* d_knn_graph;
            CUDA_VERIFY(cudaMalloc(
                    &d_knn_graph,
                    (size_t)n * intermediate_degree * sizeof(uint32_t)));
            CUDA_VERIFY(cudaMemcpyAsync(
                    d_knn_graph,
                    h_knn.data_handle(),
                    (size_t)n * intermediate_degree * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream));

            uint8_t* d_detour;
            CUDA_VERIFY(cudaMalloc(
                    &d_detour,
                    (size_t)n * intermediate_degree * sizeof(uint8_t)));
            CUDA_VERIFY(cudaMemsetAsync(
                    d_detour,
                    0xff,
                    (size_t)n * intermediate_degree * sizeof(uint8_t),
                    stream));

            // Launch detour counting for this GPU's node range
            constexpr uint32_t BATCH = 262144;
            dim3 threads(32, 1, 1);
            uint32_t total_batches =
                    (static_cast<uint32_t>(my_n) + BATCH - 1) / BATCH;

            for (uint32_t ib = 0; ib < total_batches; ib++) {
                uint32_t batch_start =
                        static_cast<uint32_t>(my_start) + ib * BATCH;
                uint32_t batch_n = std::min(
                        BATCH, static_cast<uint32_t>(my_end) - batch_start);
                dim3 blocks(batch_n, 1, 1);
                kern_detour_count<1024><<<blocks, threads, 0, stream>>>(
                        d_knn_graph,
                        static_cast<uint32_t>(n),
                        static_cast<uint32_t>(intermediate_degree),
                        static_cast<uint32_t>(graph_degree),
                        BATCH,
                        batch_start / BATCH,
                        d_detour);
            }

            CUDA_VERIFY(cudaGetLastError());
            CUDA_VERIFY(cudaMemcpyAsync(
                    h_detour.data_handle() + my_start * intermediate_degree,
                    d_detour + my_start * intermediate_degree,
                    (size_t)my_n * intermediate_degree * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost,
                    stream));

            CUDA_VERIFY(cudaStreamSynchronize(stream));
            CUDA_VERIFY(cudaFree(d_knn_graph));
            CUDA_VERIFY(cudaFree(d_detour));
            CUDA_VERIFY(cudaStreamDestroy(stream));
        }

        auto t_opt1 = std::chrono::high_resolution_clock::now();
        fprintf(stderr,
                "  [trainAllNeighbors] multi-GPU detour counting: "
                "%.2f seconds\n",
                std::chrono::duration<double>(t_opt1 - t_phase).count());

        // Phase 3b: CPU pruning (select top-graph_degree by lowest detour
        // count)
        auto* output_ptr = h_cagra.data_handle();
#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            idx_t pk = 0;
            uint32_t num_detour = 0;
            for (uint32_t l = 0;
                 l < (uint32_t)intermediate_degree && pk < graph_degree;
                 l++) {
                uint32_t next_num_detour = std::numeric_limits<uint32_t>::max();
                for (idx_t k = 0; k < intermediate_degree; k++) {
                    uint32_t dc =
                            h_detour.data_handle()[i * intermediate_degree + k];
                    if (dc > num_detour)
                        next_num_detour = std::min(dc, next_num_detour);
                    if (dc != num_detour)
                        continue;

                    uint32_t candidate =
                            h_knn.data_handle()[i * intermediate_degree + k];
                    bool dup = false;
                    for (idx_t dk = 0; dk < pk; dk++) {
                        if (candidate == output_ptr[i * graph_degree + dk]) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup && candidate < (uint32_t)n) {
                        output_ptr[i * graph_degree + pk] = candidate;
                        pk++;
                    }
                    if (pk >= graph_degree)
                        break;
                }
                if (pk >= graph_degree)
                    break;
                if (next_num_detour == std::numeric_limits<uint32_t>::max())
                    break;
                num_detour = next_num_detour;
            }
        }

        auto t_opt2 = std::chrono::high_resolution_clock::now();
        fprintf(stderr,
                "  [trainAllNeighbors] CPU pruning: %.2f seconds\n",
                std::chrono::duration<double>(t_opt2 - t_opt1).count());

        // Phase 3c: Build reverse graph and merge into output
        // (matches graph_core.cuh Phase 4+5 logic)
        std::vector<uint32_t> rev_graph(n * graph_degree, UINT32_MAX);
        std::vector<uint32_t> rev_count(n, 0);

        // Build reverse adjacency: for each edge A→B, record B←A
        for (idx_t i = 0; i < n; i++) {
            for (idx_t k = 0; k < graph_degree; k++) {
                uint32_t dest = output_ptr[i * graph_degree + k];
                if (dest >= (uint32_t)n)
                    continue;
                uint32_t slot = rev_count[dest];
                if (slot < (uint32_t)graph_degree) {
                    rev_graph[dest * graph_degree + slot] =
                            static_cast<uint32_t>(i);
                    rev_count[dest]++;
                }
            }
        }

        // Merge reverse edges into output graph (replace tail edges)
#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            uint32_t num_protected = std::max<uint32_t>(graph_degree / 2, 1);
            uint32_t kr =
                    std::min(rev_count[i], static_cast<uint32_t>(graph_degree));
            while (kr > 0) {
                kr--;
                uint32_t rev_node = rev_graph[i * graph_degree + kr];
                if (rev_node >= (uint32_t)n)
                    continue;

                // Check if already in neighbor list
                bool found = false;
                for (idx_t j = 0; j < graph_degree; j++) {
                    if (output_ptr[i * graph_degree + j] == rev_node) {
                        found = true;
                        break;
                    }
                }
                if (found)
                    continue;

                // Shift tail edges and insert at protected boundary
                for (idx_t j = graph_degree - 1; j > (idx_t)num_protected;
                     j--) {
                    output_ptr[i * graph_degree + j] =
                            output_ptr[i * graph_degree + j - 1];
                }
                output_ptr[i * graph_degree + num_protected] = rev_node;
            }
        }

        auto t_opt3 = std::chrono::high_resolution_clock::now();
        fprintf(stderr,
                "  [trainAllNeighbors] reverse graph: %.2f seconds\n",
                std::chrono::duration<double>(t_opt3 - t_opt2).count());
    } else {
        // Single-GPU optimize (original path)
        cudaSetDevice(devices[0]);
        cudaDeviceSynchronize();
        raft::device_resources single_gpu_res;
        cuvs::neighbors::cagra::helpers::optimize(
                single_gpu_res, h_knn.view(), h_cagra.view());
    }

    t_now = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "  [trainAllNeighbors] optimize: %.2f seconds\n",
            std::chrono::duration<double>(t_now - t_phase).count());
    t_phase = t_now;

    // Phase 4: Store as merged_knngraph_ for copyTo()
    merged_knngraph_.resize(n * graph_degree);
    merged_knngraph_degree_ = graph_degree;

#pragma omp parallel for
    for (idx_t i = 0; i < n * graph_degree; i++) {
        merged_knngraph_[i] = static_cast<idx_t>(h_cagra.data_handle()[i]);
    }

    multi_gpu_dataset_ = x;
    this->is_trained = true;
    this->ntotal = n;

    t_now = std::chrono::high_resolution_clock::now();
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
    if (!index->base_level_only) {
        index->add(n_train, multi_gpu_dataset_);
    } else {
        index->hnsw.prepare_level_tab(n_train, false);
        index->storage->add(n_train, multi_gpu_dataset_);
        index->ntotal = n_train;
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
