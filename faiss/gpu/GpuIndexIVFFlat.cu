/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Float16.cuh>

#include <limits>

namespace faiss {
namespace gpu {

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFFlat* index,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(
                  provider,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  index->nlist,
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    copyFrom(index);
}

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    // We haven't trained ourselves, so don't construct the IVFFlat
    // index yet
}

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        int nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(
                  provider,
                  coarseQuantizer,
                  dims,
                  metric,
                  0,
                  nlist,
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    // We could have been passed an already trained coarse quantizer. There is
    // no other quantizer that we need to train, so this is sufficient
    if (this->is_trained) {
        FAISS_ASSERT(this->quantizer);

        index_.reset(new IVFFlat(
                resources_.get(),
                this->d,
                this->nlist,
                this->metric_type,
                this->metric_arg,
                false,   // no residual
                nullptr, // no scalar quantizer
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace));
        baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
        updateQuantizer();
    }
}

GpuIndexIVFFlat::~GpuIndexIVFFlat() {}

void GpuIndexIVFFlat::reserveMemory(size_t numVecs) {
    DeviceScope scope(config_.device);

    reserveMemoryVecs_ = numVecs;
    if (index_) {
        index_->reserveMemory(numVecs);
    }
}

void GpuIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* index) {
    DeviceScope scope(config_.device);

    // This will copy GpuIndexIVF data such as the coarse quantizer
    GpuIndexIVF::copyFrom(index);

    // Clear out our old data
    index_.reset();
    baseIndex_.reset();

    // The other index might not be trained
    if (!index->is_trained) {
        FAISS_ASSERT(!this->is_trained);
        return;
    }

    // Otherwise, we can populate ourselves from the other index
    FAISS_ASSERT(this->is_trained);

    // Copy our lists as well
    index_.reset(new IVFFlat(
            resources_.get(),
            this->d,
            this->nlist,
            index->metric_type,
            index->metric_arg,
            false,   // no residual
            nullptr, // no scalar quantizer
            ivfFlatConfig_.interleavedLayout,
            ivfFlatConfig_.indicesOptions,
            config_.memorySpace));
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
    updateQuantizer();

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);
}

void GpuIndexIVFFlat::copyTo(faiss::IndexIVFFlat* index) const {
    DeviceScope scope(config_.device);

    // We must have the indices in order to copy to ourselves
    FAISS_THROW_IF_NOT_MSG(
            ivfFlatConfig_.indicesOptions != INDICES_IVF,
            "Cannot copy to CPU as GPU index doesn't retain "
            "indices (INDICES_IVF)");

    GpuIndexIVF::copyTo(index);
    index->code_size = this->d * sizeof(float);

    auto ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (index_) {
        // Copy IVF lists
        index_->copyInvertedListsTo(ivf);
    }
}

size_t GpuIndexIVFFlat::reclaimMemory() {
    DeviceScope scope(config_.device);

    if (index_) {
        return index_->reclaimMemory();
    }

    return 0;
}

void GpuIndexIVFFlat::reset() {
    DeviceScope scope(config_.device);

    if (index_) {
        index_->reset();
        this->ntotal = 0;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

void GpuIndexIVFFlat::updateQuantizer() {
    FAISS_THROW_IF_NOT_MSG(
            quantizer, "Calling updateQuantizer without a quantizer instance");

    // Only need to do something if we are already initialized
    if (index_) {
        index_->updateQuantizer(quantizer);
    }
}

void GpuIndexIVFFlat::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    // just in case someone changed our quantizer
    verifyIVFSettings_();

    if (this->is_trained) {
        FAISS_ASSERT(index_);
        return;
    }

    FAISS_ASSERT(!index_);

    // FIXME: GPUize more of this
    // First, make sure that the data is resident on the CPU, if it is not on
    // the CPU, as we depend upon parts of the CPU code
    auto hostData = toHost<float, 2>(
            (float*)x,
            resources_->getDefaultStream(config_.device),
            {(int)n, (int)this->d});

    trainQuantizer_(n, hostData.data());

    // The quantizer is now trained; construct the IVF index
    index_.reset(new IVFFlat(
            resources_.get(),
            this->d,
            this->nlist,
            this->metric_type,
            this->metric_arg,
            false,   // no residual
            nullptr, // no scalar quantizer
            ivfFlatConfig_.interleavedLayout,
            ivfFlatConfig_.indicesOptions,
            config_.memorySpace));
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
    updateQuantizer();

    if (reserveMemoryVecs_) {
        index_->reserveMemory(reserveMemoryVecs_);
    }

    this->is_trained = true;
}

} // namespace gpu
} // namespace faiss
