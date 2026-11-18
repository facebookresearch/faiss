// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <faiss/gpu/GpuIndexIVFRaBitQ.h>

#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/CuvsIVFRaBitQ.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <algorithm>
#include <limits>

namespace faiss {
namespace gpu {

GpuIndexIVFRaBitQ::GpuIndexIVFRaBitQ(
        GpuResourcesProvider* provider,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        GpuIndexIVFRaBitQConfig config)
        : GpuIndex(provider->getResources(), dims, metric, 0.0f, config),
          nlist_(nlist),
          ivfRabitqConfig_(config) {
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "GpuIndexIVFRaBitQ requires nlist > 0");
    FAISS_THROW_IF_NOT_MSG(
            nlist <= std::numeric_limits<uint32_t>::max(),
            "GpuIndexIVFRaBitQ nlist must fit in uint32_t");
    FAISS_THROW_IF_NOT_MSG(
            config.bitsPerDim >= 1 && config.bitsPerDim <= 9,
            "GpuIndexIVFRaBitQ bitsPerDim must be in [1, 9]");
    FAISS_THROW_IF_NOT_MSG(
            config.kmeansNIterations > 0,
            "GpuIndexIVFRaBitQ kmeansNIterations must be greater than zero");
    FAISS_THROW_IF_NOT_MSG(
            config.maxTrainPointsPerCluster > 0,
            "GpuIndexIVFRaBitQ maxTrainPointsPerCluster must be greater than "
            "zero");
    FAISS_THROW_IF_NOT_MSG(
            config.streamingBatchSize > 0,
            "GpuIndexIVFRaBitQ streamingBatchSize must be greater than zero");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2, "GpuIndexIVFRaBitQ supports METRIC_L2 only");
    FAISS_THROW_IF_NOT_MSG(
            should_use_cuvs(config),
            "GpuIndexIVFRaBitQ requires a supported GPU and "
            "GpuIndexIVFRaBitQConfig::use_cuvs = true");
    validateNProbe(config.nprobe, true);
    FAISS_THROW_IF_NOT_MSG(
            config.searchMode >= IVFRaBitQSearchMode::LUT16 &&
                    config.searchMode <= IVFRaBitQSearchMode::QUANT8,
            "invalid IVF-RaBitQ search mode");
    this->is_trained = false;
}

GpuIndexIVFRaBitQ::~GpuIndexIVFRaBitQ() {
    reset();
}

void GpuIndexIVFRaBitQ::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);
    FAISS_THROW_IF_NOT_MSG(
            n > 0, "GpuIndexIVFRaBitQ cannot train on an empty dataset");
    FAISS_THROW_IF_NOT_MSG(
            n <= static_cast<idx_t>(std::numeric_limits<uint32_t>::max()),
            "GpuIndexIVFRaBitQ supports at most UINT32_MAX vectors");

    if (is_trained) {
        return;
    }

    FAISS_THROW_IF_NOT_MSG(
            x, "GpuIndexIVFRaBitQ requires non-null training data");
    FAISS_THROW_IF_NOT_MSG(
            n >= nlist_, "GpuIndexIVFRaBitQ requires at least nlist vectors");

    // Publish the index and its metadata only after a successful build.
    auto index = std::make_shared<CuvsIVFRaBitQ>(
            resources_.get(), d, nlist_, metric_type, ivfRabitqConfig_);

    const int inputDevice = getDeviceForAddress(x);
    if (inputDevice >= 0 && inputDevice != config_.device) {
        auto stream = resources_->getDefaultStreamCurrentDevice();
        auto dataset = toDeviceTemporary<float, 2>(
                resources_.get(),
                config_.device,
                const_cast<float*>(x),
                stream,
                {n, d});
        index->train(n, dataset.data());
    } else {
        // Keep host input on the host so cuVS can use its streaming build.
        index->train(n, x);
    }
    index_ = std::move(index);
    ntotal = n;
    is_trained = true;
}

void GpuIndexIVFRaBitQ::add(idx_t n, const float* x) {
    FAISS_THROW_IF_MSG(
            is_trained,
            "GpuIndexIVFRaBitQ does not support incremental additions; "
            "call reset() before building a new index");
    train(n, x);
}

void GpuIndexIVFRaBitQ::add_ex(
        idx_t n,
        const void* x,
        NumericType numeric_type) {
    FAISS_THROW_IF_NOT_MSG(
            numeric_type == NumericType::Float32,
            "GpuIndexIVFRaBitQ only supports Float32 input");
    add(n, static_cast<const float*>(x));
}

void GpuIndexIVFRaBitQ::add_with_ids(idx_t, const float*, const idx_t*) {
    FAISS_THROW_MSG("GpuIndexIVFRaBitQ does not support custom IDs");
}

void GpuIndexIVFRaBitQ::add_with_ids_ex(
        idx_t,
        const void*,
        NumericType,
        const idx_t*) {
    FAISS_THROW_MSG("GpuIndexIVFRaBitQ does not support custom IDs");
}

void GpuIndexIVFRaBitQ::search_ex(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(n >= 0, "GpuIndexIVFRaBitQ requires n >= 0");
    FAISS_THROW_IF_NOT_MSG(
            k > 0 && k <= getMaxKSelection(true),
            "GpuIndexIVFRaBitQ requires k in [1, 16384]");
    FAISS_THROW_IF_NOT_MSG(
            n == 0 || (x && distances && labels),
            "GpuIndexIVFRaBitQ requires non-null query and result pointers");
    GpuIndex::search_ex(n, x, numeric_type, k, distances, labels, params);
}

bool GpuIndexIVFRaBitQ::addImplRequiresIDs_() const {
    return false;
}

void GpuIndexIVFRaBitQ::addImpl_(idx_t, const float*, const idx_t*) {
    FAISS_THROW_MSG(
            "GpuIndexIVFRaBitQ does not support incremental additions; "
            "build the index with train() or the first add() call");
}

void GpuIndexIVFRaBitQ::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    FAISS_ASSERT(is_trained && index_);
    FAISS_THROW_IF_NOT_MSG(k > 0, "GpuIndexIVFRaBitQ requires k > 0");

    const auto* params = search_params
            ? dynamic_cast<const SearchParametersIVFRaBitQ*>(search_params)
            : nullptr;
    FAISS_THROW_IF_NOT_MSG(
            !search_params || params,
            "GpuIndexIVFRaBitQ requires SearchParametersIVFRaBitQ");
    FAISS_THROW_IF_MSG(
            params && params->sel,
            "GpuIndexIVFRaBitQ does not support IDSelector filtering");

    uint32_t nprobe = params ? params->nprobe : ivfRabitqConfig_.nprobe;
    const auto searchMode =
            params ? params->searchMode : ivfRabitqConfig_.searchMode;
    validateNProbe(nprobe, true);
    nprobe = std::min(nprobe, static_cast<uint32_t>(nlist_));

    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(labels, {n, k});
    index_->search(queries, k, outDistances, outLabels, nprobe, searchMode);
}

void GpuIndexIVFRaBitQ::reset() {
    DeviceScope scope(config_.device);
    if (index_) {
        // cuVS buffers retain the build-time allocation stream. Searches may
        // use a newer resources stream, so finish them before releasing
        // storage.
        resources_->syncDefaultStreamCurrentDevice();
        index_.reset();
    }
    ntotal = 0;
    is_trained = false;
}

} // namespace gpu
} // namespace faiss
