// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, 6sense Insights Inc.
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
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexHNSW.h>
#include <faiss/gpu/impl/GpuHnswTypes.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/GpuHnswBuildVanilla.cuh>
#include <faiss/gpu/impl/GpuHnswSearch.cuh>

#include <cstdio>
#include <memory>
#include <stdexcept>

namespace faiss {
namespace gpu {

namespace {

// Upload the filtered-search bitset (deletes / TTL / partition) into the
// scratch slot's device buffer on the search stream. gpu_hnsw_search reads
// sp.bitset_data only to decide the filtered path; the bytes themselves come
// from sc.d_bitset. The host BitsetView bytes referenced by sp.bitset_data
// must stay alive until the stream is synchronized (the caller synchronizes
// before returning, which happens in every search path here).
inline void upload_bitset_if_needed(
        GpuHnswSearchScratch& sc,
        const GpuHnswSearchParams& sp,
        int nq,
        cudaStream_t stream) {
    if (sp.bitset_data == nullptr || sp.bitset_nbits <= 0) {
        return;
    }
    size_t bytes = static_cast<size_t>((sp.bitset_nbits + 7) / 8);
    sc.ensure_filter(nq, bytes);
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            sc.d_bitset,
            sp.bitset_data,
            bytes,
            cudaMemcpyHostToDevice,
            stream));
}

} // namespace

GpuIndexHNSW::GpuIndexHNSW(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        GpuIndexHNSWConfig config)
        : GpuIndex(provider->getResources(), dims, metric, 0.0f, config),
          hnswConfig_(config) {
    FAISS_THROW_IF_NOT_MSG(
            metric == faiss::METRIC_L2 ||
                    metric == faiss::METRIC_INNER_PRODUCT,
            "GpuIndexHNSW supports METRIC_L2 and METRIC_INNER_PRODUCT only "
            "(cosine = normalize + inner product)");
    this->is_trained = false;
}

GpuIndexHNSW::GpuIndexHNSW(
        GpuResourcesProvider* provider,
        const faiss::IndexHNSW* index,
        GpuIndexHNSWConfig config)
        : GpuIndex(
                  provider->getResources(),
                  index->d,
                  index->metric_type,
                  0.0f,
                  config),
          hnswConfig_(config) {
    this->is_trained = false;
    copyFrom(index);
}

GpuIndexHNSW::~GpuIndexHNSW() = default;

void GpuIndexHNSW::copyFrom(const faiss::IndexHNSW* index) {
    FAISS_THROW_IF_NOT_MSG(index, "index must not be null");
    FAISS_THROW_IF_NOT_MSG(index->ntotal > 0, "index must not be empty");
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == faiss::METRIC_L2 ||
                    index->metric_type == faiss::METRIC_INNER_PRODUCT,
            "GpuIndexHNSW supports METRIC_L2 and METRIC_INNER_PRODUCT only "
            "(cosine = normalize + inner product)");

    DeviceScope scope(config_.device);

    this->d = index->d;
    this->metric_type = index->metric_type;
    this->ntotal = index->ntotal;

    bool use_ip = index->metric_type == faiss::METRIC_INNER_PRODUCT;
    use_ip_ = use_ip;

    if (dynamic_cast<const faiss::IndexScalarQuantizer*>(index->storage)) {
        deviceIndex_ = from_index_hnsw_sq(*index, use_ip, config_.device);
    } else {
        deviceIndex_ = from_index_hnsw_flat(*index, use_ip, config_.device);
    }

    this->is_trained = true;
}

void GpuIndexHNSW::copyTo(faiss::IndexHNSW* /*index*/) const {
    FAISS_THROW_MSG(
            "GpuIndexHNSW is search-only and does not support copyTo(). "
            "The GPU index uploads a CPU-built graph and does not retain a "
            "reconstructable CPU copy of the link structure. Keep the source "
            "faiss::IndexHNSW to obtain a CPU index.");
}

void GpuIndexHNSW::reset() {
    deviceIndex_.reset();
    this->ntotal = 0;
    this->is_trained = false;
    use_ip_ = false;
}

void GpuIndexHNSW::setSearchParams(const GpuHnswSearchParams& params) const {
    std::lock_guard<std::mutex> lock(searchParamsMutex_);
    directSearchParams_ = params;
    hasDirectSearchParams_ = true;
}

bool GpuIndexHNSW::addImplRequiresIDs_() const {
    return false;
}

void GpuIndexHNSW::addImpl_(idx_t, const float*, const idx_t*) {
    FAISS_THROW_MSG(
            "GpuIndexHNSW does not support add(). "
            "Build on CPU with IndexHNSW, then call copyFrom().");
}

void GpuIndexHNSW::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    FAISS_THROW_IF_NOT_MSG(
            this->is_trained && deviceIndex_,
            "Index not loaded. Call copyFrom() first.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n must be > 0");

    auto& idx = *deviceIndex_;

    GpuHnswSearchParams sp;
    bool got_params = false;

    // Prefer direct params set via setSearchParams() — avoids dynamic_cast.
    // These are sticky: they stay in effect for every subsequent search until
    // overwritten, so a later search never silently falls back to defaults.
    {
        std::lock_guard<std::mutex> lock(searchParamsMutex_);
        if (hasDirectSearchParams_) {
            sp = directSearchParams_;
            got_params = true;
        }
    }

    // Fallback: try dynamic_cast from SearchParameters.
    if (!got_params && search_params) {
        auto* params =
                dynamic_cast<const SearchParametersGpuHNSW*>(search_params);
        if (params) {
            sp.ef = params->ef;
            sp.search_width = params->search_width;
            sp.max_iterations = params->max_iterations;
            sp.thread_block_size = params->thread_block_size;
            got_params = true;
        }
    }

    ScratchPoolGuard guard(*idx.scratch_pool);
    auto* slot = guard.get();
    auto& sc = slot->scratch;
    cudaStream_t stream = slot->stream;

    int nq = static_cast<int>(n);
    int dim = static_cast<int>(idx.dim);
    sc.ensure(nq, k, dim, static_cast<int>(idx.n_rows));

    // The parent GpuIndex::search copied host queries into `x` (and will later
    // copy outputs back) on the resources' default stream. Our scratch-pool
    // slot has its own private stream, so make it wait for that H2D/D2D copy to
    // complete before we read `x` — otherwise the search can race ahead and
    // operate on partially-written query vectors.
    cudaStream_t defaultStream = resources_->getDefaultStream(config_.device);
    cudaEvent_t inputReady;
    GPU_HNSW_CUDA_CHECK(
            cudaEventCreateWithFlags(&inputReady, cudaEventDisableTiming));
    GPU_HNSW_CUDA_CHECK(cudaEventRecord(inputReady, defaultStream));
    GPU_HNSW_CUDA_CHECK(cudaStreamWaitEvent(stream, inputReady, 0));
    GPU_HNSW_CUDA_CHECK(cudaEventDestroy(inputReady));

    // D2D: query vectors (GpuIndex::search passes device pointers)
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            sc.d_queries,
            x,
            static_cast<size_t>(nq) * dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));

    // A filter may arrive through sticky setSearchParams() on this standard
    // search() path too, so upload it before searching — mirroring searchHost /
    // searchHostInt8. Without this, sp.bitset_data being non-null selects the
    // filtered kernel while sc.d_bitset is still unallocated (illegal access).
    upload_bitset_if_needed(sc, sp, nq, stream);

    gpu_hnsw_search(stream, sp, idx, sc, nq, k);

    // D2D: distances (output is a device pointer from GpuIndex::search)
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            distances,
            sc.d_distances,
            static_cast<size_t>(nq) * k * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));

    // Labels: D2H stage (uint64_t→idx_t conversion), then H2D back
    auto tmp = std::make_unique<uint64_t[]>(nq * k);
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            tmp.get(),
            sc.d_neighbors,
            static_cast<size_t>(nq) * k * sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            stream));
    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto h_labels = std::make_unique<idx_t[]>(nq * k);
    for (int i = 0; i < nq * k; i++) {
        h_labels[i] = (tmp[i] == UINT64_MAX) ? -1 : static_cast<idx_t>(tmp[i]);
    }

    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            labels,
            h_labels.get(),
            static_cast<size_t>(nq) * k * sizeof(idx_t),
            cudaMemcpyHostToDevice,
            stream));
    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void GpuIndexHNSW::searchHost(
        idx_t n,
        const float* x_host,
        int k,
        float* distances_host,
        idx_t* labels_host,
        const GpuHnswSearchParams& sp) const {
    FAISS_THROW_IF_NOT_MSG(
            this->is_trained && deviceIndex_,
            "Index not loaded. Call copyFrom() first.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n must be > 0");

    GPU_HNSW_CUDA_CHECK(cudaSetDevice(config_.device));
    DeviceScope scope(config_.device);
    auto& idx = *deviceIndex_;

    ScratchPoolGuard guard(*idx.scratch_pool);
    auto* slot = guard.get();
    auto& sc = slot->scratch;
    cudaStream_t stream = slot->stream;

    int nq = static_cast<int>(n);
    int dim = static_cast<int>(idx.dim);
    sc.ensure(nq, k, dim, static_cast<int>(idx.n_rows));

    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            sc.d_queries,
            x_host,
            static_cast<size_t>(nq) * dim * sizeof(float),
            cudaMemcpyDefault,
            stream));

    upload_bitset_if_needed(sc, sp, nq, stream);

    gpu_hnsw_search(stream, sp, idx, sc, nq, k);

    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            distances_host,
            sc.d_distances,
            static_cast<size_t>(nq) * k * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));

    auto tmp = std::make_unique<uint64_t[]>(nq * k);
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            tmp.get(),
            sc.d_neighbors,
            static_cast<size_t>(nq) * k * sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            stream));

    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < nq * k; i++) {
        labels_host[i] =
                (tmp[i] == UINT64_MAX) ? -1 : static_cast<idx_t>(tmp[i]);
    }
}

void GpuIndexHNSW::searchHostInt8(
        idx_t n,
        const int8_t* x_host,
        int k,
        float* distances_host,
        idx_t* labels_host,
        const GpuHnswSearchParams& sp) const {
    FAISS_THROW_IF_NOT_MSG(
            this->is_trained && deviceIndex_,
            "Index not loaded. Call copyFrom() first.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n must be > 0");

    auto& idx = *deviceIndex_;
    // If dim is not divisible by 4, DP4A cannot be used; fall back to fp32 path.
    if (idx.dim % 4 != 0) {
        auto fp32_fallback = std::make_unique<float[]>(
                static_cast<size_t>(n) * idx.dim);
        for (int64_t i = 0; i < static_cast<int64_t>(n) * idx.dim; i++) {
            fp32_fallback[i] = static_cast<float>(x_host[i]);
        }
        searchHost(n, fp32_fallback.get(), k, distances_host, labels_host, sp);
        return;
    }

    GPU_HNSW_CUDA_CHECK(cudaSetDevice(config_.device));
    DeviceScope scope(config_.device);

    ScratchPoolGuard guard(*idx.scratch_pool);
    auto* slot = guard.get();
    auto& sc = slot->scratch;
    cudaStream_t stream = slot->stream;

    int nq = static_cast<int>(n);
    int dim = static_cast<int>(idx.dim);
    int64_t nelem = static_cast<int64_t>(nq) * dim;

    sc.ensure(nq, k, dim, static_cast<int>(idx.n_rows), /*use_i8_queries=*/true);

    // Upload int8 queries directly — dataset on GPU is already in signed int8
    // (upload_int8_dataset applies codes[i]-128, reversing FAISS's +128 bias,
    // yielding the original signed user values). Queries arrive as the same
    // signed int8 user values; no shift is needed.
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            sc.d_queries_i8,
            x_host,
            nelem * sizeof(int8_t),
            cudaMemcpyHostToDevice,
            stream));

    // Also upload fp32 queries to d_queries for upper-layer greedy search.
    auto fp32_q = std::make_unique<float[]>(nelem);
    for (int64_t i = 0; i < nelem; i++) {
        fp32_q[i] = static_cast<float>(x_host[i]);
    }
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            sc.d_queries,
            fp32_q.get(),
            nelem * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));

    // Synchronize to ensure host buffers (fp32_q) are not freed
    // while the async copies are in flight.
    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));

    upload_bitset_if_needed(sc, sp, nq, stream);

    gpu_hnsw_search(stream, sp, idx, sc, nq, k);

    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            distances_host,
            sc.d_distances,
            static_cast<size_t>(nq) * k * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));

    auto tmp = std::make_unique<uint64_t[]>(nq * k);
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            tmp.get(),
            sc.d_neighbors,
            static_cast<size_t>(nq) * k * sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            stream));

    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < nq * k; i++) {
        labels_host[i] =
                (tmp[i] == UINT64_MAX) ? -1 : static_cast<idx_t>(tmp[i]);
    }
}

} // namespace gpu
} // namespace faiss
