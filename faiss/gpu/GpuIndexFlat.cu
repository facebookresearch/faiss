/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <limits>

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
    // Flat index doesn't need training
    this->is_trained = true;

    // Construct index
    DeviceScope scope(config_.device);
    data_.reset(new FlatIndex(
            resources_.get(),
            dims,
            flatConfig_.useFloat16,
            flatConfig_.storeTransposed,
            config_.memorySpace));
}

GpuIndexFlat::GpuIndexFlat(
        std::shared_ptr<GpuResources> resources,
        int dims,
        faiss::MetricType metric,
        GpuIndexFlatConfig config)
        : GpuIndex(resources, dims, metric, 0, config), flatConfig_(config) {
    // Flat index doesn't need training
    this->is_trained = true;

    // Construct index
    DeviceScope scope(config_.device);
    data_.reset(new FlatIndex(
            resources_.get(),
            dims,
            flatConfig_.useFloat16,
            flatConfig_.storeTransposed,
            config_.memorySpace));
}

GpuIndexFlat::~GpuIndexFlat() {}

void GpuIndexFlat::copyFrom(const faiss::IndexFlat* index) {
    DeviceScope scope(config_.device);

    GpuIndex::copyFrom(index);

    // GPU code has 32 bit indices
    FAISS_THROW_IF_NOT_FMT(
            index->ntotal <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %zu indices; "
            "attempting to copy CPU index with %zu parameters",
            (size_t)std::numeric_limits<int>::max(),
            (size_t)index->ntotal);

    data_.reset();
    data_.reset(new FlatIndex(
            resources_.get(),
            this->d,
            flatConfig_.useFloat16,
            flatConfig_.storeTransposed,
            config_.memorySpace));

    // The index could be empty
    if (index->ntotal > 0) {
        data_->add(
                index->xb.data(),
                index->ntotal,
                resources_->getDefaultStream(config_.device));
    }
}

void GpuIndexFlat::copyTo(faiss::IndexFlat* index) const {
    DeviceScope scope(config_.device);

    GpuIndex::copyTo(index);

    FAISS_ASSERT(data_);
    FAISS_ASSERT(data_->getSize() == this->ntotal);
    index->xb.resize(this->ntotal * this->d);

    auto stream = resources_->getDefaultStream(config_.device);

    if (this->ntotal > 0) {
        if (flatConfig_.useFloat16) {
            auto vecFloat32 = data_->getVectorsFloat32Copy(stream);
            fromDevice(vecFloat32, index->xb.data(), stream);
        } else {
            fromDevice(data_->getVectorsFloat32Ref(), index->xb.data(), stream);
        }
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

void GpuIndexFlat::train(Index::idx_t n, const float* x) {
    // nothing to do
}

void GpuIndexFlat::add(Index::idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    if (n == 0) {
        // nothing to add
        return;
    }

    DeviceScope scope(config_.device);

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

bool GpuIndexFlat::addImplRequiresIDs_() const {
    return false;
}

void GpuIndexFlat::addImpl_(int n, const float* x, const Index::idx_t* ids) {
    FAISS_ASSERT(data_);
    FAISS_ASSERT(n > 0);

    // We do not support add_with_ids
    FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");

    // Due to GPU indexing in int32, we can't store more than this
    // number of vectors on a GPU
    FAISS_THROW_IF_NOT_FMT(
            this->ntotal + n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %zu indices",
            (size_t)std::numeric_limits<int>::max());

    data_->add(x, n, resources_->getDefaultStream(config_.device));
    this->ntotal += n;
}

void GpuIndexFlat::searchImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        Index::idx_t* labels) const {
    auto stream = resources_->getDefaultStream(config_.device);

    // Input and output data are already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int)this->d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<Index::idx_t, 2, true> outLabels(labels, {n, k});

    // FlatIndex only supports int indices
    DeviceTensor<int, 2, true> outIntLabels(
            resources_.get(), makeTempAlloc(AllocType::Other, stream), {n, k});

    data_->query(
            queries,
            k,
            metric_type,
            metric_arg,
            outDistances,
            outIntLabels,
            true);

    // Convert int to idx_t
    convertTensor<int, Index::idx_t, 2>(stream, outIntLabels, outLabels);
}

void GpuIndexFlat::reconstruct(Index::idx_t key, float* out) const {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_MSG(key < this->ntotal, "index out of bounds");
    auto stream = resources_->getDefaultStream(config_.device);

    if (flatConfig_.useFloat16) {
        // FIXME jhj: kernel for copy
        auto vec = data_->getVectorsFloat32Copy(key, 1, stream);
        fromDevice(vec.data(), out, this->d, stream);
    } else {
        auto vec = data_->getVectorsFloat32Ref()[key];
        fromDevice(vec.data(), out, this->d, stream);
    }
}

void GpuIndexFlat::reconstruct_n(Index::idx_t i0, Index::idx_t num, float* out)
        const {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_MSG(i0 < this->ntotal, "index out of bounds");
    FAISS_THROW_IF_NOT_MSG(i0 + num - 1 < this->ntotal, "num out of bounds");
    auto stream = resources_->getDefaultStream(config_.device);

    if (flatConfig_.useFloat16) {
        // FIXME jhj: kernel for copy
        auto vec = data_->getVectorsFloat32Copy(i0, num, stream);
        fromDevice(vec.data(), out, num * this->d, stream);
    } else {
        auto vec = data_->getVectorsFloat32Ref()[i0];
        fromDevice(vec.data(), out, this->d * num, stream);
    }
}

void GpuIndexFlat::compute_residual(
        const float* x,
        float* residual,
        Index::idx_t key) const {
    compute_residual_n(1, x, residual, &key);
}

void GpuIndexFlat::compute_residual_n(
        Index::idx_t n,
        const float* xs,
        float* residuals,
        const Index::idx_t* keys) const {
    FAISS_THROW_IF_NOT_FMT(
            n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %zu indices",
            (size_t)std::numeric_limits<int>::max());

    auto stream = resources_->getDefaultStream(config_.device);

    DeviceScope scope(config_.device);

    auto vecsDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(xs),
            stream,
            {(int)n, (int)this->d});
    auto idsDevice = toDeviceTemporary<Index::idx_t, 1>(
            resources_.get(),
            config_.device,
            const_cast<Index::idx_t*>(keys),
            stream,
            {(int)n});

    auto residualDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            residuals,
            stream,
            {(int)n, (int)this->d});

    // Convert idx_t to int
    auto keysInt = convertTensorTemporary<Index::idx_t, int, 1>(
            resources_.get(), stream, idsDevice);

    FAISS_ASSERT(data_);
    data_->computeResidual(vecsDevice, keysInt, residualDevice);

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
