/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <limits>

namespace faiss {
namespace gpu {

GpuIndexIVFScalarQuantizer::GpuIndexIVFScalarQuantizer(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFScalarQuantizer* index,
        GpuIndexIVFScalarQuantizerConfig config)
        : GpuIndexIVF(
                  provider,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  index->nlist,
                  config),
          sq(index->sq),
          by_residual(index->by_residual),
          ivfSQConfig_(config),
          reserveMemoryVecs_(0) {
    // This will perform SQ settings verification as well
    copyFrom(index);
}

GpuIndexIVFScalarQuantizer::GpuIndexIVFScalarQuantizer(
        GpuResourcesProvider* provider,
        int dims,
        idx_t nlist,
        faiss::ScalarQuantizer::QuantizerType qtype,
        faiss::MetricType metric,
        bool encodeResidual,
        GpuIndexIVFScalarQuantizerConfig config)
        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
          sq(dims, qtype),
          by_residual(encodeResidual),
          ivfSQConfig_(config),
          reserveMemoryVecs_(0) {
    // We haven't trained ourselves, so don't construct the IVFFlat
    // index yet
    verifySQSettings_();
}

GpuIndexIVFScalarQuantizer::GpuIndexIVFScalarQuantizer(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        idx_t nlist,
        faiss::ScalarQuantizer::QuantizerType qtype,
        faiss::MetricType metric,
        bool encodeResidual,
        GpuIndexIVFScalarQuantizerConfig config)
        : GpuIndexIVF(
                  provider,
                  coarseQuantizer,
                  dims,
                  metric,
                  0,
                  nlist,
                  config),
          sq(dims, qtype),
          by_residual(encodeResidual),
          ivfSQConfig_(config),
          reserveMemoryVecs_(0) {
    // While we were passed an existing coarse quantizer instance (possibly
    // trained or not), we have not yet trained our scalar quantizer, so we
    // can't construct our index_ instance yet
    this->is_trained = false;

    verifySQSettings_();
}

GpuIndexIVFScalarQuantizer::~GpuIndexIVFScalarQuantizer() {}

void GpuIndexIVFScalarQuantizer::verifySQSettings_() const {
    FAISS_THROW_IF_NOT_MSG(
            isSQSupported(sq.qtype), "Unsupported scalar QuantizerType on GPU");

    // Check the amount of shared memory per block available based on our type
    // is sufficient
    // This check was previously in IVFFlatScan.cu, moved here to apply upon
    // index construction
    if (sq.qtype == ScalarQuantizer::QuantizerType::QT_8bit ||
        sq.qtype == ScalarQuantizer::QuantizerType::QT_4bit) {
        // There are quantization parameters per each dimension for these SQ
        // types. These parameters are retained in shared memory for access
        int maxDim =
                getMaxSharedMemPerBlock(config_.device) / (sizeof(float) * 2);

        FAISS_THROW_IF_NOT_FMT(
                this->d < maxDim,
                "GpuIndexIVFScalarQuantizer: Insufficient shared memory "
                "available on the GPU for QT_8bit or QT_4bit with %d "
                "dimensions; maximum dimensions possible is %d",
                this->d,
                maxDim);
    }
}

void GpuIndexIVFScalarQuantizer::reserveMemory(size_t numVecs) {
    DeviceScope scope(config_.device);

    reserveMemoryVecs_ = numVecs;
    if (index_) {
        index_->reserveMemory(numVecs);
    }
}

void GpuIndexIVFScalarQuantizer::copyFrom(
        const faiss::IndexIVFScalarQuantizer* index) {
    DeviceScope scope(config_.device);

    // Clear out our old data
    index_.reset();
    baseIndex_.reset();

    // Copy what we need from the CPU index
    GpuIndexIVF::copyFrom(index);
    sq = index->sq;
    by_residual = index->by_residual;

    // The other index might not be trained, in which case we don't need to copy
    // over the lists
    if (!index->is_trained) {
        return;
    }

    // Otherwise, we can populate ourselves from the other index
    this->is_trained = true;

    // Copy our lists as well
    index_.reset(new IVFFlat(
            resources_.get(),
            this->d,
            this->nlist,
            index->metric_type,
            index->metric_arg,
            by_residual,
            &sq,
            ivfSQConfig_.interleavedLayout,
            ivfSQConfig_.indicesOptions,
            config_.memorySpace));
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
    updateQuantizer();

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);

    verifySQSettings_();
}

void GpuIndexIVFScalarQuantizer::copyTo(
        faiss::IndexIVFScalarQuantizer* index) const {
    DeviceScope scope(config_.device);

    // We must have the indices in order to copy to ourselves
    FAISS_THROW_IF_NOT_MSG(
            ivfSQConfig_.indicesOptions != INDICES_IVF,
            "Cannot copy to CPU as GPU index doesn't retain "
            "indices (INDICES_IVF)");

    GpuIndexIVF::copyTo(index);
    index->sq = sq;
    index->code_size = sq.code_size;
    index->by_residual = by_residual;

    auto ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (index_) {
        // Copy IVF lists
        index_->copyInvertedListsTo(ivf);
    }
}

size_t GpuIndexIVFScalarQuantizer::reclaimMemory() {
    DeviceScope scope(config_.device);

    if (index_) {
        return index_->reclaimMemory();
    }

    return 0;
}

void GpuIndexIVFScalarQuantizer::updateQuantizer() {
    FAISS_THROW_IF_NOT_MSG(
            quantizer, "Calling updateQuantizer without a quantizer instance");

    // Only need to do something if we are already initialized
    if (index_) {
        index_->updateQuantizer(quantizer);
    }
}

void GpuIndexIVFScalarQuantizer::reset() {
    DeviceScope scope(config_.device);

    if (index_) {
        index_->reset();
        this->ntotal = 0;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

void GpuIndexIVFScalarQuantizer::trainResiduals_(idx_t n, const float* x) {
    // The input is already guaranteed to be on the CPU
    if (!by_residual) {
        sq.train(n, x);
    } else {
        std::vector<idx_t> assign(n);
        quantizer->assign(n, x, assign.data());

        std::vector<float> residuals(n * d);
        quantizer->compute_residual_n(n, x, residuals.data(), assign.data());

        sq.train(n, residuals.data());
    }
}

void GpuIndexIVFScalarQuantizer::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    // just in case someone changed us
    verifySQSettings_();
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
            {n, this->d});

    trainQuantizer_(n, hostData.data());
    trainResiduals_(n, hostData.data());

    // The quantizer is now trained; construct the IVF index
    index_.reset(new IVFFlat(
            resources_.get(),
            this->d,
            this->nlist,
            this->metric_type,
            this->metric_arg,
            by_residual,
            &sq,
            ivfSQConfig_.interleavedLayout,
            ivfSQConfig_.indicesOptions,
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
