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

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/CuvsIVFRaBitQ.cuh>

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
            metric == METRIC_L2, "GpuIndexIVFRaBitQ supports METRIC_L2 only");
    FAISS_THROW_IF_NOT_MSG(
            should_use_cuvs(config),
            "GpuIndexIVFRaBitQ requires a supported GPU and "
            "GpuIndexIVFRaBitQConfig::use_cuvs = true");
    this->is_trained = false;
}

GpuIndexIVFRaBitQ::~GpuIndexIVFRaBitQ() = default;

void GpuIndexIVFRaBitQ::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);
    FAISS_THROW_IF_NOT_MSG(
            n > 0, "GpuIndexIVFRaBitQ cannot train on an empty dataset");

    if (is_trained) {
        return;
    }

    index_ = std::make_shared<CuvsIVFRaBitQ>(
            resources_.get(), d, nlist_, metric_type, ivfRabitqConfig_);
    index_->train(n, x);
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
    FAISS_THROW_IF_NOT_MSG(
            nprobe > 0, "GpuIndexIVFRaBitQ nprobe must be greater than zero");
    nprobe = std::min(nprobe, static_cast<uint32_t>(nlist_));

    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(labels, {n, k});
    index_->search(queries, k, outDistances, outLabels, nprobe, searchMode);
}

void GpuIndexIVFRaBitQ::reset() {
    DeviceScope scope(config_.device);
    index_.reset();
    ntotal = 0;
    is_trained = false;
}

} // namespace gpu
} // namespace faiss
