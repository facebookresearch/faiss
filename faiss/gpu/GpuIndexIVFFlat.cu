/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#if defined USE_NVIDIA_CUVS
#include <cuvs/neighbors/ivf_flat.hpp>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsIVFFlat.cuh>
#endif

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
        idx_t nlist,
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
        idx_t nlist,
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
        setIndex_(
                resources_.get(),
                this->d,
                this->nlist,
                this->metric_type,
                this->metric_arg,
                false,   // no residual
                nullptr, // no scalar quantizer
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
        baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
        updateQuantizer();
    }
}

GpuIndexIVFFlat::~GpuIndexIVFFlat() {}

void GpuIndexIVFFlat::reserveMemory(size_t numVecs) {
    DeviceScope scope(config_.device);

    if (should_use_cuvs(config_)) {
        FAISS_THROW_MSG(
                "Pre-allocation of IVF lists is not supported with cuVS enabled.");
    }

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

    // skip base class allocations if cuVS is enabled
    if (!should_use_cuvs(config_)) {
        baseIndex_.reset();
    }

    // The other index might not be trained
    if (!index->is_trained) {
        FAISS_ASSERT(!is_trained);
        return;
    }

    // Otherwise, we can populate ourselves from the other index
    FAISS_ASSERT(is_trained);

    // Copy our lists as well
    setIndex_(
            resources_.get(),
            d,
            nlist,
            index->metric_type,
            index->metric_arg,
            false,   // no residual
            nullptr, // no scalar quantizer
            ivfFlatConfig_.interleavedLayout,
            ivfFlatConfig_.indicesOptions,
            config_.memorySpace);
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

    // just in case someone changed our quantizer
    verifyIVFSettings_();

    if (this->is_trained) {
        FAISS_ASSERT(index_);
        if (should_use_cuvs(config_)) {
            // copy the IVF centroids to the cuVS index
            // in case it has been reset. This is because `reset` clears the
            // cuVS index and its centroids.
            // TODO: change this once the coarse quantizer is separated from
            // cuVS index
            updateQuantizer();
        };
        return;
    }

    FAISS_ASSERT(!index_);

    if (should_use_cuvs(config_)) {
#if defined USE_NVIDIA_CUVS
        setIndex_(
                resources_.get(),
                this->d,
                this->nlist,
                this->metric_type,
                this->metric_arg,
                false,   // no residual
                nullptr, // no scalar quantizer
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
        const raft::device_resources& raft_handle =
                resources_->getRaftHandleCurrentDevice();

        cuvs::neighbors::ivf_flat::index_params cuvs_index_params;
        cuvs_index_params.n_lists = nlist;
        cuvs_index_params.metric = metricFaissToCuvs(metric_type, false);
        cuvs_index_params.add_data_on_build = false;
        cuvs_index_params.kmeans_trainset_fraction =
                static_cast<double>(cp.max_points_per_centroid * nlist) /
                static_cast<double>(n);
        cuvs_index_params.kmeans_n_iters = cp.niter;

        auto cuvsIndex_ =
                std::static_pointer_cast<CuvsIVFFlat, IVFFlat>(index_);

        std::optional<cuvs::neighbors::ivf_flat::index<float, idx_t>>
                cuvs_ivfflat_index;

        if (getDeviceForAddress(x) >= 0) {
            auto dataset_d =
                    raft::make_device_matrix_view<const float, idx_t>(x, n, d);
            cuvs_ivfflat_index = cuvs::neighbors::ivf_flat::build(
                    raft_handle, cuvs_index_params, dataset_d);
        } else {
            auto dataset_h =
                    raft::make_host_matrix_view<const float, idx_t>(x, n, d);
            cuvs_ivfflat_index = cuvs::neighbors::ivf_flat::build(
                    raft_handle, cuvs_index_params, dataset_h);
        }

        quantizer->train(
                nlist, cuvs_ivfflat_index.value().centers().data_handle());
        quantizer->add(
                nlist, cuvs_ivfflat_index.value().centers().data_handle());
        raft_handle.sync_stream();

        cuvsIndex_->setCuvsIndex(std::move(*cuvs_ivfflat_index));
#else
        FAISS_THROW_MSG(
                "cuVS has not been compiled into the current version so it cannot be used.");
#endif
    } else {
        // FIXME: GPUize more of this
        // First, make sure that the data is resident on the CPU, if it is not
        // on the CPU, as we depend upon parts of the CPU code
        auto hostData = toHost<float, 2>(
                (float*)x,
                resources_->getDefaultStream(config_.device),
                {n, this->d});
        trainQuantizer_(n, hostData.data());

        setIndex_(
                resources_.get(),
                this->d,
                this->nlist,
                this->metric_type,
                this->metric_arg,
                false,   // no residual
                nullptr, // no scalar quantizer
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
        updateQuantizer();
    }

    // The quantizer is now trained; construct the IVF index
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);

    if (reserveMemoryVecs_) {
        if (should_use_cuvs(config_)) {
            FAISS_THROW_MSG(
                    "Pre-allocation of IVF lists is not supported with cuVS enabled.");
        } else
            index_->reserveMemory(reserveMemoryVecs_);
    }

    this->is_trained = true;
}

void GpuIndexIVFFlat::setIndex_(
        GpuResources* resources,
        int dim,
        int nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        /// Optional ScalarQuantizer
        faiss::ScalarQuantizer* scalarQ,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space) {
    if (should_use_cuvs(config_)) {
#if defined USE_NVIDIA_CUVS
        FAISS_THROW_IF_NOT_MSG(
                ivfFlatConfig_.indicesOptions == INDICES_64_BIT,
                "cuVS only supports INDICES_64_BIT");
        if (!ivfFlatConfig_.interleavedLayout) {
            fprintf(stderr,
                    "WARN: interleavedLayout is set to False with cuVS enabled. This will be ignored.\n");
        }
        index_.reset(new CuvsIVFFlat(
                resources,
                dim,
                nlist,
                metric,
                metricArg,
                useResidual,
                scalarQ,
                interleavedLayout,
                indicesOptions,
                space));
#else
        FAISS_THROW_MSG(
                "cuVS has not been compiled into the current version so it cannot be used.");
#endif
    } else {
        index_.reset(new IVFFlat(
                resources,
                dim,
                nlist,
                metric,
                metricArg,
                useResidual,
                scalarQ,
                interleavedLayout,
                indicesOptions,
                space));
    }
}

void GpuIndexIVFFlat::reconstruct_n(idx_t i0, idx_t ni, float* out) const {
    FAISS_ASSERT(index_);

    if (ni == 0) {
        // nothing to do
        return;
    }

    FAISS_THROW_IF_NOT_FMT(
            i0 < this->ntotal,
            "start index (%zu) out of bounds (ntotal %zu)",
            i0,
            this->ntotal);
    FAISS_THROW_IF_NOT_FMT(
            i0 + ni - 1 < this->ntotal,
            "max index requested (%zu) out of bounds (ntotal %zu)",
            i0 + ni - 1,
            this->ntotal);

    index_->reconstruct_n(i0, ni, out);
}

} // namespace gpu
} // namespace faiss
