/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndexBinaryFlat.h>

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/BinaryFlatIndex.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>

namespace faiss {
namespace gpu {

/// Default CPU search size for which we use paged copies
constexpr size_t kMinPageSize = (size_t)256 * 1024 * 1024;

GpuIndexBinaryFlat::GpuIndexBinaryFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexBinaryFlat* index,
        GpuIndexBinaryFlatConfig config)
        : IndexBinary(index->d),
          resources_(provider->getResources()),
          binaryFlatConfig_(config) {
    FAISS_THROW_IF_NOT_FMT(
            this->d % 8 == 0,
            "vector dimension (number of bits) "
            "must be divisible by 8 (passed %d)",
            this->d);

    // Flat index doesn't need training
    this->is_trained = true;

    copyFrom(index);
}

GpuIndexBinaryFlat::GpuIndexBinaryFlat(
        GpuResourcesProvider* provider,
        int dims,
        GpuIndexBinaryFlatConfig config)
        : IndexBinary(dims),
          resources_(provider->getResources()),
          binaryFlatConfig_(std::move(config)) {
    DeviceScope scope(binaryFlatConfig_.device);
    FAISS_THROW_IF_NOT_FMT(
            this->d % 8 == 0,
            "vector dimension (number of bits) "
            "must be divisible by 8 (passed %d)",
            this->d);

    // Flat index doesn't need training
    this->is_trained = true;

    // Construct index
    data_.reset(new BinaryFlatIndex(
            resources_.get(), this->d, binaryFlatConfig_.memorySpace));
}

GpuIndexBinaryFlat::~GpuIndexBinaryFlat() {}

int GpuIndexBinaryFlat::getDevice() const {
    return binaryFlatConfig_.device;
}

std::shared_ptr<GpuResources> GpuIndexBinaryFlat::getResources() {
    return resources_;
}

void GpuIndexBinaryFlat::copyFrom(const faiss::IndexBinaryFlat* index) {
    DeviceScope scope(binaryFlatConfig_.device);

    this->d = index->d;

    this->ntotal = index->ntotal;

    // destroy old first before allocating new
    data_.reset();
    data_.reset(new BinaryFlatIndex(
            resources_.get(), this->d, binaryFlatConfig_.memorySpace));

    // The index could be empty
    if (index->ntotal > 0) {
        data_->add(
                index->xb.data(),
                index->ntotal,
                resources_->getDefaultStream(binaryFlatConfig_.device));
    }
}

void GpuIndexBinaryFlat::copyTo(faiss::IndexBinaryFlat* index) const {
    DeviceScope scope(binaryFlatConfig_.device);

    index->d = this->d;
    index->ntotal = this->ntotal;

    FAISS_ASSERT(data_);
    FAISS_ASSERT(data_->getSize() == this->ntotal);
    index->xb.resize(this->ntotal * (this->d / 8));

    if (this->ntotal > 0) {
        fromDevice(
                data_->getVectorsRef(),
                index->xb.data(),
                resources_->getDefaultStream(binaryFlatConfig_.device));
    }
}

void GpuIndexBinaryFlat::add(idx_t n, const uint8_t* x) {
    DeviceScope scope(binaryFlatConfig_.device);

    // To avoid multiple re-allocations, ensure we have enough storage
    // available
    data_->reserve(n, resources_->getDefaultStream(binaryFlatConfig_.device));

    data_->add(
            (const unsigned char*)x,
            n,
            resources_->getDefaultStream(binaryFlatConfig_.device));
    this->ntotal += n;
}

void GpuIndexBinaryFlat::add(idx_t n, const void* x, NumericType numeric_type) {
    IndexBinary::add(n, x, numeric_type);
}

void GpuIndexBinaryFlat::reset() {
    DeviceScope scope(binaryFlatConfig_.device);

    // Free the underlying memory
    data_->reset();
    this->ntotal = 0;
}

void GpuIndexBinaryFlat::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        faiss::idx_t* labels,
        const SearchParameters* params) const {
    DeviceScope scope(binaryFlatConfig_.device);
    auto stream = resources_->getDefaultStream(binaryFlatConfig_.device);

    if (n == 0) {
        return;
    }

    FAISS_THROW_IF_NOT_MSG(!params, "params not implemented");

    validateKSelect(k, binaryFlatConfig_.use_cuvs);

    // The input vectors may be too large for the GPU, but we still
    // assume that the output distances and labels are not.
    // Go ahead and make space for output distances and labels on the
    // GPU.
    // If we reach a point where all inputs are too big, we can add
    // another level of tiling.
    auto outDistances = toDeviceTemporary<int32_t, 2>(
            resources_.get(),
            binaryFlatConfig_.device,
            distances,
            stream,
            {n, k});

    auto outIndices = toDeviceTemporary<idx_t, 2>(
            resources_.get(), binaryFlatConfig_.device, labels, stream, {n, k});

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
                    n, x, k, outDistances.data(), outIndices.data());
            usePaged = true;
        }
    }

    if (!usePaged) {
        searchNonPaged_(n, x, k, outDistances.data(), outIndices.data());
    }

    // Copy back if necessary
    fromDevice<int32_t, 2>(outDistances, distances, stream);
    fromDevice<idx_t, 2>(outIndices, labels, stream);
}

void GpuIndexBinaryFlat::search(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IndexBinary::search(n, x, numeric_type, k, distances, labels, params);
}

void GpuIndexBinaryFlat::searchNonPaged_(
        idx_t n,
        const uint8_t* x,
        int k,
        int32_t* outDistancesData,
        idx_t* outIndicesData) const {
    Tensor<int32_t, 2, true> outDistances(outDistancesData, {n, k});
    Tensor<idx_t, 2, true> outIndices(outIndicesData, {n, k});

    auto stream = resources_->getDefaultStream(binaryFlatConfig_.device);

    // Make sure arguments are on the device we desire; use temporary
    // memory allocations to move it if necessary
    auto vecs = toDeviceTemporary<uint8_t, 2>(
            resources_.get(),
            binaryFlatConfig_.device,
            const_cast<uint8_t*>(x),
            stream,
            {n, (this->d / 8)});

    data_->query(vecs, k, outDistances, outIndices);
}

void GpuIndexBinaryFlat::searchFromCpuPaged_(
        idx_t n,
        const uint8_t* x,
        int k,
        int32_t* outDistancesData,
        idx_t* outIndicesData) const {
    Tensor<int32_t, 2, true> outDistances(outDistancesData, {n, k});
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
                outIndicesSlice.data());
    }
}

void GpuIndexBinaryFlat::reconstruct(faiss::idx_t key, uint8_t* out) const {
    DeviceScope scope(binaryFlatConfig_.device);

    FAISS_THROW_IF_NOT_MSG(key < this->ntotal, "index out of bounds");
    auto stream = resources_->getDefaultStream(binaryFlatConfig_.device);

    auto& vecs = data_->getVectorsRef();
    auto vec = vecs[key];

    fromDevice(vec.data(), out, vecs.getSize(1), stream);
}

} // namespace gpu
} // namespace faiss
