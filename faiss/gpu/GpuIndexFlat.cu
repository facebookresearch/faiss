/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <limits>

#if defined USE_NVIDIA_CUVS
#include <faiss/gpu/impl/CuvsFlatIndex.cuh>
#endif

namespace faiss {
namespace gpu {

GpuIndexFlat::GpuIndexFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexFlat* index,
        GpuIndexFlatConfig config)
        : GpuIndex(
                  provider->getResources(),
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  config),
          flatConfig_(config) {
    // Flat index doesn't need training
    this->is_trained = true;

    copyFrom(index);
}

GpuIndexFlat::GpuIndexFlat(
        std::shared_ptr<GpuResources> resources,
        const faiss::IndexFlat* index,
        GpuIndexFlatConfig config)
        : GpuIndex(
                  resources,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  config),
          flatConfig_(config) {
    // Flat index doesn't need training
    this->is_trained = true;

    copyFrom(index);
}

GpuIndexFlat::GpuIndexFlat(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        GpuIndexFlatConfig config)
        : GpuIndex(provider->getResources(), dims, metric, 0, config),
          flatConfig_(config) {
    DeviceScope scope(config_.device);

    // Flat index doesn't need training
    this->is_trained = true;

    // Construct index
    resetIndex_(dims);
}

GpuIndexFlat::GpuIndexFlat(
        std::shared_ptr<GpuResources> resources,
        int dims,
        faiss::MetricType metric,
        GpuIndexFlatConfig config)
        : GpuIndex(resources, dims, metric, 0, config), flatConfig_(config) {
    DeviceScope scope(config_.device);

    // Flat index doesn't need training
    this->is_trained = true;

    // Construct index
    resetIndex_(dims);
}

GpuIndexFlat::~GpuIndexFlat() {}

void GpuIndexFlat::resetIndex_(int dims) {
#if defined USE_NVIDIA_CUVS

    if (should_use_cuvs(config_)) {
        data_.reset(new CuvsFlatIndex(
                resources_.get(),
                dims,
                flatConfig_.useFloat16,
                config_.memorySpace));
    } else
#else
    if (should_use_cuvs(config_)) {
        FAISS_THROW_MSG(
                "cuVS has not been compiled into the current version so it cannot be used.");
    } else
#endif
    {
        data_.reset(new FlatIndex(
                resources_.get(),
                dims,
                flatConfig_.useFloat16,
                config_.memorySpace));
    }
}

void GpuIndexFlat::copyFrom(const faiss::IndexFlat* index) {
    DeviceScope scope(config_.device);

    GpuIndex::copyFrom(index);

    data_.reset();
    resetIndex_(this->d);

    // The index could be empty
    if (index->ntotal > 0) {
        data_->add(
                index->get_xb(),
                index->ntotal,
                resources_->getDefaultStream(config_.device));
    }
}

void GpuIndexFlat::copyTo(faiss::IndexFlat* index) const {
    DeviceScope scope(config_.device);

    GpuIndex::copyTo(index);
    index->code_size = sizeof(float) * this->d;

    FAISS_ASSERT(data_);
    FAISS_ASSERT(data_->getSize() == this->ntotal);
    index->codes.resize(this->ntotal * index->code_size);

    if (this->ntotal > 0) {
        // FIXME: there is an extra GPU allocation here and copy if the flat
        // index is already float32
        reconstruct_n(0, this->ntotal, index->get_xb());
    }
}

size_t GpuIndexFlat::getNumVecs() const {
    return this->ntotal;
}

void GpuIndexFlat::reset() {
    DeviceScope scope(config_.device);

    // Free the underlying memory
    data_->reset();
    this->ntotal = 0;
}

void GpuIndexFlat::train(idx_t n, const float* x) {
    // nothing to do
}

void GpuIndexFlat::train(idx_t n, const void* x, NumericType numeric_type) {
    GpuIndex::train(n, x, numeric_type);
}

void GpuIndexFlat::add(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    if (n == 0) {
        // nothing to add
        return;
    }

    // To avoid multiple re-allocations, ensure we have enough storage
    // available
    data_->reserve(n, resources_->getDefaultStream(config_.device));

    // If we're not operating in float16 mode, we don't need the input
    // data to be resident on our device; we can add directly.
    if (!flatConfig_.useFloat16) {
        addImpl_(n, x, nullptr);
    } else {
        // Otherwise, perform the paging
        GpuIndex::add(n, x);
    }
}

void GpuIndexFlat::add(idx_t n, const void* x, NumericType numeric_type) {
    GpuIndex::add(n, x, numeric_type);
}

bool GpuIndexFlat::addImplRequiresIDs_() const {
    return false;
}

void GpuIndexFlat::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    // current device already set
    // n already validated
    FAISS_ASSERT(data_);
    FAISS_ASSERT(n > 0);

    // We do not support add_with_ids
    FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");

    data_->add(x, n, resources_->getDefaultStream(config_.device));
    this->ntotal += n;
}

void GpuIndexFlat::addImpl_(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        const idx_t* ids) {
    GpuIndex::addImpl_(n, x, numeric_type, ids);
}

void GpuIndexFlat::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // current device already set
    // n/k already validated
    auto stream = resources_->getDefaultStream(config_.device);

    // Input and output data are already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, this->d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(labels, {n, k});

    data_->query(
            queries, k, metric_type, metric_arg, outDistances, outLabels, true);
}

void GpuIndexFlat::searchImpl_(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    GpuIndex::searchImpl_(n, x, numeric_type, k, distances, labels, params);
}

void GpuIndexFlat::reconstruct(idx_t key, float* out) const {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_FMT(
            key < this->ntotal,
            "index %zu out of bounds (ntotal %zu)",
            key,
            this->ntotal);
    auto stream = resources_->getDefaultStream(config_.device);

    // FIXME: `out` may already be on the device, in which case this is an
    // unneeded allocation
    DeviceTensor<float, 2, true> vec(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {1, this->d});

    FAISS_ASSERT(data_);
    data_->reconstruct(key, 1, vec);

    fromDevice(vec.data(), out, this->d, stream);
}

void GpuIndexFlat::reconstruct_n(idx_t i0, idx_t n, float* out) const {
    DeviceScope scope(config_.device);

    if (n == 0) {
        // nothing to do
        return;
    }

    FAISS_THROW_IF_NOT_FMT(
            i0 < this->ntotal,
            "start index (%zu) out of bounds (ntotal %zu)",
            i0,
            this->ntotal);
    FAISS_THROW_IF_NOT_FMT(
            i0 + n - 1 < this->ntotal,
            "max index requested (%zu) out of bounds (ntotal %zu)",
            i0 + n - 1,
            this->ntotal);
    auto stream = resources_->getDefaultStream(config_.device);

    auto outDevice = toDeviceTemporary<float, 2>(
            resources_.get(), config_.device, out, stream, {n, this->d});

    FAISS_ASSERT(data_);
    data_->reconstruct(i0, n, outDevice);

    fromDevice<float, 2>(outDevice, out, stream);
}

void GpuIndexFlat::reconstruct_batch(idx_t n, const idx_t* keys, float* out)
        const {
    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    if (n == 0) {
        // nothing to do
        return;
    }

    auto keysDevice = toDeviceTemporary<faiss::idx_t, 1>(
            resources_.get(),
            config_.device,
            const_cast<idx_t*>(keys),
            stream,
            {n});

    auto outDevice = toDeviceTemporary<float, 2>(
            resources_.get(), config_.device, out, stream, {n, this->d});

    FAISS_ASSERT(data_);
    data_->reconstruct(keysDevice, outDevice);

    // If the output is on the host, copy back if needed
    fromDevice<float, 2>(outDevice, out, stream);
}

void GpuIndexFlat::compute_residual(const float* x, float* residual, idx_t key)
        const {
    compute_residual_n(1, x, residual, &key);
}

void GpuIndexFlat::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    if (n == 0) {
        // nothing to do
        return;
    }

    auto vecsDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(xs),
            stream,
            {n, this->d});
    auto idsDevice = toDeviceTemporary<idx_t, 1>(
            resources_.get(),
            config_.device,
            const_cast<idx_t*>(keys),
            stream,
            {n});
    auto residualDevice = toDeviceTemporary<float, 2>(
            resources_.get(), config_.device, residuals, stream, {n, this->d});

    FAISS_ASSERT(data_);
    data_->computeResidual(vecsDevice, idsDevice, residualDevice);

    // If the output is on the host, copy back if needed
    fromDevice<float, 2>(residualDevice, residuals, stream);
}

//
// GpuIndexFlatL2
//

GpuIndexFlatL2::GpuIndexFlatL2(
        GpuResourcesProvider* provider,
        faiss::IndexFlatL2* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, index, config) {}

GpuIndexFlatL2::GpuIndexFlatL2(
        std::shared_ptr<GpuResources> resources,
        faiss::IndexFlatL2* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, index, config) {}

GpuIndexFlatL2::GpuIndexFlatL2(
        GpuResourcesProvider* provider,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, dims, faiss::METRIC_L2, config) {}

GpuIndexFlatL2::GpuIndexFlatL2(
        std::shared_ptr<GpuResources> resources,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, dims, faiss::METRIC_L2, config) {}

void GpuIndexFlatL2::copyFrom(faiss::IndexFlat* index) {
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == metric_type,
            "Cannot copy a GpuIndexFlatL2 from an index of "
            "different metric_type");

    GpuIndexFlat::copyFrom(index);
}

void GpuIndexFlatL2::copyTo(faiss::IndexFlat* index) {
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == metric_type,
            "Cannot copy a GpuIndexFlatL2 to an index of "
            "different metric_type");

    GpuIndexFlat::copyTo(index);
}

//
// GpuIndexFlatIP
//

GpuIndexFlatIP::GpuIndexFlatIP(
        GpuResourcesProvider* provider,
        faiss::IndexFlatIP* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, index, config) {}

GpuIndexFlatIP::GpuIndexFlatIP(
        std::shared_ptr<GpuResources> resources,
        faiss::IndexFlatIP* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, index, config) {}

GpuIndexFlatIP::GpuIndexFlatIP(
        GpuResourcesProvider* provider,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, dims, faiss::METRIC_INNER_PRODUCT, config) {}

GpuIndexFlatIP::GpuIndexFlatIP(
        std::shared_ptr<GpuResources> resources,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, dims, faiss::METRIC_INNER_PRODUCT, config) {}

void GpuIndexFlatIP::copyFrom(faiss::IndexFlat* index) {
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == metric_type,
            "Cannot copy a GpuIndexFlatIP from an index of "
            "different metric_type");

    GpuIndexFlat::copyFrom(index);
}

void GpuIndexFlatIP::copyTo(faiss::IndexFlat* index) {
    // The passed in index must be IP
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == metric_type,
            "Cannot copy a GpuIndexFlatIP to an index of "
            "different metric_type");

    GpuIndexFlat::copyTo(index);
}

} // namespace gpu
} // namespace faiss
