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

#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <cstddef>
#include <faiss/gpu/impl/CuvsCagra.cuh>
#include <optional>

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

void GpuIndexCagra::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);
    if (this->is_trained) {
        FAISS_ASSERT(index_);
        return;
    }

    FAISS_ASSERT(!index_);

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
    }
    index_ = std::make_shared<CuvsCagra>(
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
            cagraConfig_.refine_rate);

    index_->train(n, x);

    this->is_trained = true;
    this->ntotal = n;
}

void GpuIndexCagra::add(idx_t n, const float* x) {
    train(n, x);
}

bool GpuIndexCagra::addImplRequiresIDs_() const {
    return false;
};

void GpuIndexCagra::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_MSG("adding vectors is not supported by GpuIndexCagra.");
};

void GpuIndexCagra::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    FAISS_ASSERT(this->is_trained && index_);
    FAISS_ASSERT(n > 0);

    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, this->d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(const_cast<idx_t*>(labels), {n, k});

    SearchParametersCagra* params;
    if (search_params) {
        params = dynamic_cast<SearchParametersCagra*>(
                const_cast<SearchParameters*>(search_params));
    } else {
        params = new SearchParametersCagra{};
    }

    index_->search(
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
            params->seed);

    if (not search_params) {
        delete params;
    }
}

void GpuIndexCagra::copyFrom(const faiss::IndexHNSWCagra* index) {
    FAISS_ASSERT(index);

    DeviceScope scope(config_.device);

    GpuIndex::copyFrom(index);

    auto base_index = dynamic_cast<IndexFlat*>(index->storage);
    FAISS_ASSERT(base_index);
    auto distances = base_index->get_xb();

    auto hnsw = index->hnsw;
    // copy level 0 to a dense knn graph matrix
    std::vector<idx_t> knn_graph;
    knn_graph.reserve(index->ntotal * hnsw.nb_neighbors(0));

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

    index_ = std::make_shared<CuvsCagra>(
            this->resources_.get(),
            this->d,
            index->ntotal,
            hnsw.nb_neighbors(0),
            distances,
            knn_graph.data(),
            this->metric_type,
            this->metric_arg,
            INDICES_64_BIT);

    this->is_trained = true;
}

void GpuIndexCagra::copyTo(faiss::IndexHNSWCagra* index) const {
    FAISS_ASSERT(index_ && this->is_trained && index);

    DeviceScope scope(config_.device);

    //
    // Index information
    //
    GpuIndex::copyTo(index);
    // This needs to be zeroed out as this implementation adds vectors to the
    // cpuIndex instead of copying fields
    index->ntotal = 0;

    auto graph_degree = index_->get_knngraph_degree();
    auto M = graph_degree / 2;
    if (index->storage and index->own_fields) {
        delete index->storage;
    }

    if (this->metric_type == METRIC_L2) {
        index->storage = new IndexFlatL2(index->d);
    } else if (this->metric_type == METRIC_INNER_PRODUCT) {
        index->storage = new IndexFlatIP(index->d);
    }
    index->own_fields = true;
    index->keep_max_size_level0 = true;
    index->hnsw.reset();
    index->hnsw.assign_probas.clear();
    index->hnsw.cum_nneighbor_per_level.clear();
    index->hnsw.set_default_probas(M, 1.0 / log(M));

    auto n_train = this->ntotal;
    float* train_dataset;
    auto dataset = index_->get_training_dataset();
    bool allocation = false;
    if (getDeviceForAddress(dataset) >= 0) {
        train_dataset = new float[n_train * index->d];
        allocation = true;
        raft::copy(
                train_dataset,
                dataset,
                n_train * index->d,
                this->resources_->getRaftHandleCurrentDevice().get_stream());
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

void GpuIndexCagra::reset() {
    DeviceScope scope(config_.device);

    if (index_) {
        index_->reset();
        this->ntotal = 0;
        this->is_trained = false;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

std::vector<idx_t> GpuIndexCagra::get_knngraph() const {
    FAISS_ASSERT(index_ && this->is_trained);

    return index_->get_knngraph();
}

} // namespace gpu
} // namespace faiss
