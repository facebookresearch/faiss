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

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>

#include <faiss/gpu/GpuIndexBinaryCagra.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/BinaryCuvsCagra.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <cstddef>

#include <optional>
#include "GpuResources.h"

namespace faiss {
namespace gpu {

/// Default CPU search size for which we use paged copies
constexpr size_t kMinPageSize = (size_t)256 * 1024 * 1024;

GpuIndexBinaryCagra::GpuIndexBinaryCagra(
        GpuResourcesProvider* provider,
        int dims,
        GpuIndexCagraConfig config)
        : IndexBinary(dims),
          resources_(provider->getResources()),
          cagraConfig_(std::move(config)) {
    DeviceScope scope(cagraConfig_.device);
    FAISS_THROW_IF_NOT_FMT(
            this->d % 8 == 0,
            "vector dimension (number of bits) "
            "must be divisible by 8 (passed %d)",
            this->d);

    this->is_trained = false;
}

GpuIndexBinaryCagra::~GpuIndexBinaryCagra() {}

int GpuIndexBinaryCagra::getDevice() const {
    return cagraConfig_.device;
}

std::shared_ptr<GpuResources> GpuIndexBinaryCagra::getResources() {
    return resources_;
}

void GpuIndexBinaryCagra::train(idx_t n, const uint8_t* x) {
    DeviceScope scope(cagraConfig_.device);
    if (this->is_trained) {
        FAISS_ASSERT(index_);
        return;
    }

    FAISS_ASSERT(!index_);

    index_ = std::make_shared<BinaryCuvsCagra>(
            this->resources_.get(),
            this->d,
            cagraConfig_.intermediate_graph_degree,
            cagraConfig_.graph_degree,
            cagraConfig_.store_dataset,
            INDICES_64_BIT);

    index_->train(n, x);

    this->is_trained = true;
    this->ntotal = n;
}

void GpuIndexBinaryCagra::train(
        idx_t n,
        const void* x,
        NumericType numeric_type) {
    IndexBinary::train(n, x, numeric_type);
}

void GpuIndexBinaryCagra::add(idx_t n, const uint8_t* x) {
    train(n, x);
}

void GpuIndexBinaryCagra::add(
        idx_t n,
        const void* x,
        NumericType numeric_type) {
    IndexBinary::add(n, x, numeric_type);
}

void GpuIndexBinaryCagra::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int* distances,
        faiss::idx_t* labels,
        const SearchParameters* params) const {
    DeviceScope scope(cagraConfig_.device);
    auto stream = resources_->getDefaultStream(cagraConfig_.device);

    if (n == 0) {
        return;
    }

    FAISS_THROW_IF_NOT_MSG(!params, "params not implemented");

    // validateKSelect(k);

    // The input vectors may be too large for the GPU, but we still
    // assume that the output distances and labels are not.
    // Go ahead and make space for output distances and labels on the
    // GPU.
    // If we reach a point where all inputs are too big, we can add
    // another level of tiling.
    // In order to remain consistent with other IndexBinary, we must return
    // distances as integers despite cuVS search functions requiring the
    // distances to be float. This can lead to up to 2x the allocation of
    // distances.
    auto outDistances = toDeviceTemporary<int, 2>(
            resources_.get(), cagraConfig_.device, distances, stream, {n, k});

    auto outIndices = toDeviceTemporary<idx_t, 2>(
            resources_.get(), cagraConfig_.device, labels, stream, {n, k});

    bool usePaged = false;

    if (getDeviceForAddress(x) == -1) {
        // It is possible that the user is querying for a vector set size
        // `x` that won't fit on the GPU.
        // In this case, we will have to handle paging of the data from CPU
        // -> GPU.
        // Currently, we don't handle the case where the output data won't
        // fit on the GPU (e.g., n * k is too large for the GPU memory).
        size_t dataSize = n * (this->d / 8) * sizeof(uint8_t);

        if (dataSize >= kMinPageSize) {
            searchFromCpuPaged_(
                    n, x, k, outDistances.data(), outIndices.data(), params);
            usePaged = true;
        }
    }

    if (!usePaged) {
        searchNonPaged_(
                n, x, k, outDistances.data(), outIndices.data(), params);
    }

    // Copy back if necessary
    fromDevice<int, 2>(outDistances, distances, stream);
    fromDevice<idx_t, 2>(outIndices, labels, stream);
}

void GpuIndexBinaryCagra::search(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        int* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IndexBinary::search(n, x, numeric_type, k, distances, labels, params);
}

void GpuIndexBinaryCagra::searchNonPaged_(
        idx_t n,
        const uint8_t* x,
        int k,
        int* outDistancesData,
        idx_t* outIndicesData,
        const SearchParameters* params) const {
    auto stream = resources_->getDefaultStream(cagraConfig_.device);

    // Make sure arguments are on the device we desire; use temporary
    // memory allocations to move it if necessary
    auto vecs = toDeviceTemporary<uint8_t, 2>(
            resources_.get(),
            cagraConfig_.device,
            const_cast<uint8_t*>(x),
            stream,
            {n, (this->d / 8)});

    searchImpl_(n, vecs.data(), k, outDistancesData, outIndicesData, params);
}

void GpuIndexBinaryCagra::searchFromCpuPaged_(
        idx_t n,
        const uint8_t* x,
        int k,
        int* outDistancesData,
        idx_t* outIndicesData,
        const SearchParameters* params) const {
    Tensor<int, 2, true> outDistances(outDistancesData, {n, k});
    Tensor<idx_t, 2, true> outIndices(outIndicesData, {n, k});

    idx_t vectorSize = sizeof(uint8_t) * (this->d / 8);

    // Just page without overlapping copy with compute (as GpuIndexFlat does)
    auto batchSize =
            utils::nextHighestPowerOf2(((idx_t)kMinPageSize / vectorSize));

    for (idx_t cur = 0; cur < n; cur += batchSize) {
        auto num = std::min(batchSize, n - cur);

        auto outDistancesSlice = outDistances.narrowOutermost(cur, num);
        auto outIndicesSlice = outIndices.narrowOutermost(cur, num);

        searchNonPaged_(
                num,
                x + cur * (this->d / 8),
                k,
                outDistancesSlice.data(),
                outIndicesSlice.data(),
                params);
    }
}

void GpuIndexBinaryCagra::searchImpl_(
        idx_t n,
        const uint8_t* x,
        int k,
        int* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    FAISS_ASSERT(this->is_trained && index_);
    FAISS_ASSERT(n > 0);

    Tensor<uint8_t, 2, true> queries(const_cast<uint8_t*>(x), {n, this->d / 8});

    Tensor<int, 2, true> outDistances(distances, {n, k});
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

void GpuIndexBinaryCagra::copyFrom(const faiss::IndexBinaryHNSW* index) {
    FAISS_ASSERT(index);

    DeviceScope scope(cagraConfig_.device);

    this->d = index->d;
    FAISS_THROW_IF_NOT(this->d % 8 == 0);
    this->code_size = index->d / 8;
    this->ntotal = index->ntotal;
    this->is_trained = index->is_trained;

    IndexBinaryFlat* flat_storage =
            dynamic_cast<IndexBinaryFlat*>(index->storage);
    FAISS_ASSERT(flat_storage);

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

    index_ = std::make_shared<BinaryCuvsCagra>(
            this->resources_.get(),
            this->d,
            index->ntotal,
            hnsw.nb_neighbors(0),
            flat_storage->xb.data(),
            knn_graph.data(),
            INDICES_64_BIT);

    this->is_trained = true;
}

void GpuIndexBinaryCagra::copyTo(faiss::IndexBinaryHNSW* index) const {
    FAISS_ASSERT(index_ && this->is_trained && index);

    DeviceScope scope(cagraConfig_.device);

    //
    // Index information
    //
    index->d = this->d;
    FAISS_THROW_IF_NOT(this->d % 8 == 0);
    index->code_size = this->d / 8;
    index->is_trained = this->is_trained;
    // This needs to be zeroed out as this implementation adds vectors to the
    // cpuIndex instead of copying fields
    index->ntotal = 0;

    auto graph_degree = index_->get_knngraph_degree();
    auto M = graph_degree / 2;
    if (index->storage and index->own_fields) {
        delete index->storage;
    }

    index->storage = new IndexBinaryFlat(index->d);
    index->own_fields = true;
    index->keep_max_size_level0 = true;
    index->hnsw.reset();
    index->hnsw.assign_probas.clear();
    index->hnsw.cum_nneighbor_per_level.clear();
    index->hnsw.set_default_probas(M, 1.0 / log(M));

    auto n_train = this->ntotal;
    auto stream = resources_->getDefaultStream(cagraConfig_.device);

    auto train_data = toHost<uint8_t, 2>(
            const_cast<uint8_t*>(index_->get_training_dataset()),
            stream,
            {idx_t(n_train), this->d / 8});

    // turn off as level 0 is copied from CAGRA graph
    index->init_level0 = false;
    index->add(n_train, train_data.data());

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
    index->keep_max_size_level0 = false;
}

void GpuIndexBinaryCagra::reset() {
    DeviceScope scope(cagraConfig_.device);

    if (index_) {
        index_->reset();
        this->ntotal = 0;
        this->is_trained = false;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

std::vector<idx_t> GpuIndexBinaryCagra::get_knngraph() const {
    FAISS_ASSERT(index_ && this->is_trained);

    return index_->get_knngraph();
}

} // namespace gpu
} // namespace faiss
