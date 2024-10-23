/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/utils/utils.h>
#include <faiss/gpu/impl/IVFPQ.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>

#if defined USE_NVIDIA_RAFT
#include <faiss/gpu/utils/RaftUtils.h>
#include <faiss/gpu/impl/RaftIVFPQ.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_helpers.cuh>
#endif

#include <limits>

namespace faiss {
namespace gpu {

GpuIndexIVFPQ::GpuIndexIVFPQ(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFPQ* index,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVF(
                  provider,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  index->nlist,
                  config),
          pq(index->pq),
          ivfpqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(0),
          bitsPerCode_(0),
          reserveMemoryVecs_(0) {
    copyFrom(index);
}

GpuIndexIVFPQ::GpuIndexIVFPQ(
        GpuResourcesProvider* provider,
        int dims,
        idx_t nlist,
        idx_t subQuantizers,
        idx_t bitsPerCode,
        faiss::MetricType metric,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
          pq(dims, subQuantizers, bitsPerCode),
          ivfpqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(subQuantizers),
          bitsPerCode_(bitsPerCode),
          reserveMemoryVecs_(0) {
    verifyPQSettings_();
}

GpuIndexIVFPQ::GpuIndexIVFPQ(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        idx_t nlist,
        idx_t subQuantizers,
        idx_t bitsPerCode,
        faiss::MetricType metric,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVF(
                  provider,
                  coarseQuantizer,
                  dims,
                  metric,
                  0,
                  nlist,
                  config),
          pq(dims, subQuantizers, bitsPerCode),
          ivfpqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(subQuantizers),
          bitsPerCode_(bitsPerCode),
          reserveMemoryVecs_(0) {
    // While we were passed an existing coarse quantizer instance (possibly
    // trained or not), we have not yet trained our product quantizer, so we are
    // not ourselves fully trained and we can not yet construct our index_
    // instance
    this->is_trained = false;

    FAISS_THROW_IF_NOT_MSG(
            !config.use_raft,
            "GpuIndexIVFPQ: RAFT does not support separate coarseQuantizer");

    verifyPQSettings_();
}

GpuIndexIVFPQ::~GpuIndexIVFPQ() {}

void GpuIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* index) {
    DeviceScope scope(config_.device);

    // This will copy GpuIndexIVF data such as the coarse quantizer
    GpuIndexIVF::copyFrom(index);

    // Clear out our old data
    index_.reset();

    // skip base class allocations if RAFT is enabled
    if (!should_use_raft(config_)) {
        baseIndex_.reset();
    }

    pq = index->pq;
    subQuantizers_ = index->pq.M;
    bitsPerCode_ = index->pq.nbits;

    // We only support this
    FAISS_THROW_IF_NOT_MSG(
            ivfpqConfig_.interleavedLayout || index->pq.nbits == 8,
            "GPU: only pq.nbits == 8 is supported");
    FAISS_THROW_IF_NOT_MSG(
            index->by_residual, "GPU: only by_residual = true is supported");
    FAISS_THROW_IF_NOT_MSG(
            index->polysemous_ht == 0, "GPU: polysemous codes not supported");

    verifyPQSettings_();

    // The other index might not be trained
    if (!index->is_trained) {
        // copied in GpuIndex::copyFrom
        FAISS_ASSERT(!is_trained);
        return;
    }

    // Copy our lists as well
    // The product quantizer must have data in it
    FAISS_ASSERT(index->pq.centroids.size() > 0);
    setIndex_(
            resources_.get(),
            this->d,
            this->nlist,
            index->metric_type,
            index->metric_arg,
            subQuantizers_,
            bitsPerCode_,
            ivfpqConfig_.useFloat16LookupTables,
            ivfpqConfig_.useMMCodeDistance,
            ivfpqConfig_.interleavedLayout,
            (float*)index->pq.centroids.data(),
            ivfpqConfig_.indicesOptions,
            config_.memorySpace);
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFPQ>(index_);

    // Doesn't make sense to reserve memory here
    FAISS_ASSERT(quantizer);
    updateQuantizer();

    index_->setPrecomputedCodes(quantizer, usePrecomputedTables_);

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);
}

void GpuIndexIVFPQ::copyTo(faiss::IndexIVFPQ* index) const {
    DeviceScope scope(config_.device);

    // We must have the indices in order to copy to ourselves
    FAISS_THROW_IF_NOT_MSG(
            ivfpqConfig_.indicesOptions != INDICES_IVF,
            "Cannot copy to CPU as GPU index doesn't retain "
            "indices (INDICES_IVF)");

    GpuIndexIVF::copyTo(index);

    //
    // IndexIVFPQ information
    //
    index->by_residual = true;
    index->use_precomputed_table = 0;
    index->code_size = utils::divUp(subQuantizers_ * bitsPerCode_, 8);
    index->pq = faiss::ProductQuantizer(this->d, subQuantizers_, bitsPerCode_);

    index->do_polysemous_training = false;
    index->polysemous_training = nullptr;

    index->scan_table_threshold = 0;
    index->max_codes = 0;
    index->polysemous_ht = 0;
    index->precomputed_table.clear();

    auto ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (index_) {
        // Copy IVF lists
        index_->copyInvertedListsTo(ivf);

        // Copy PQ centroids
        auto devPQCentroids = index_->getPQCentroids();
        index->pq.centroids.resize(devPQCentroids.numElements());

        fromDevice<float, 3>(
                devPQCentroids,
                index->pq.centroids.data(),
                resources_->getDefaultStream(config_.device));

        if (usePrecomputedTables_) {
            index->precompute_table();
        }
    }
}

void GpuIndexIVFPQ::reserveMemory(size_t numVecs) {
    DeviceScope scope(config_.device);

    reserveMemoryVecs_ = numVecs;
    if (index_) {
        index_->reserveMemory(numVecs);
    }
}

void GpuIndexIVFPQ::setPrecomputedCodes(bool enable) {
    DeviceScope scope(config_.device);

    usePrecomputedTables_ = enable;
    if (index_) {
        FAISS_ASSERT(quantizer);
        index_->setPrecomputedCodes(quantizer, enable);
    }

    verifyPQSettings_();
}

bool GpuIndexIVFPQ::getPrecomputedCodes() const {
    return usePrecomputedTables_;
}

int GpuIndexIVFPQ::getNumSubQuantizers() const {
    return subQuantizers_;
}

int GpuIndexIVFPQ::getBitsPerCode() const {
    return bitsPerCode_;
}

int GpuIndexIVFPQ::getCentroidsPerSubQuantizer() const {
    return utils::pow2(bitsPerCode_);
}

size_t GpuIndexIVFPQ::reclaimMemory() {
    DeviceScope scope(config_.device);

    if (index_) {
        return index_->reclaimMemory();
    }

    return 0;
}

void GpuIndexIVFPQ::reset() {
    DeviceScope scope(config_.device);

    if (index_) {
        index_->reset();
        this->ntotal = 0;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

void GpuIndexIVFPQ::updateQuantizer() {
    FAISS_THROW_IF_NOT_MSG(
            quantizer, "Calling updateQuantizer without a quantizer instance");

    // Only need to do something if we are already initialized
    if (index_) {
        index_->updateQuantizer(quantizer);
    }
}

void GpuIndexIVFPQ::trainResidualQuantizer_(idx_t n, const float* x) {
    // Code largely copied from faiss::IndexIVFPQ
    auto x_in = x;

    x = fvecs_maybe_subsample(
            d,
            (size_t*)&n,
            pq.cp.max_points_per_centroid * pq.ksub,
            x,
            verbose,
            pq.cp.seed);

    std::unique_ptr<const float[]> del_x(x_in == x ? nullptr : x);

    if (this->verbose) {
        printf("computing residuals\n");
    }

    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data());

    std::vector<float> residuals(n * d);
    quantizer->compute_residual_n(n, x, residuals.data(), assign.data());

    if (this->verbose) {
        printf("training %d x %d product quantizer on %ld vectors in %dD\n",
               subQuantizers_,
               getCentroidsPerSubQuantizer(),
               n,
               this->d);
    }

    // For PQ training purposes, accelerate it by using a GPU clustering index
    // if a clustering index has not already been assigned
    if (!pq.assign_index) {
        try {
            GpuIndexFlatConfig config;
            config.device = ivfpqConfig_.device;
            config.use_raft = false;
            GpuIndexFlatL2 pqIndex(resources_, pq.dsub, config);

            pq.assign_index = &pqIndex;
            pq.train(n, residuals.data());
        } catch (...) {
            pq.assign_index = nullptr;
            throw;
        }

        pq.assign_index = nullptr;
    } else {
        // use the currently assigned clustering index
        pq.train(n, residuals.data());
    }
}

void GpuIndexIVFPQ::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    // just in case someone changed us
    verifyPQSettings_();
    verifyIVFSettings_();

    if (this->is_trained) {
        FAISS_ASSERT(index_);
        if (should_use_raft(config_)) {
            // if RAFT is enabled, copy the IVF centroids to the RAFT index in
            // case it has been reset. This is because reset clears the RAFT
            // index and its centroids.
            // TODO: change this once the coarse quantizer is separated from
            // RAFT index
            updateQuantizer();
        };
        return;
    }

    FAISS_ASSERT(!index_);

    // RAFT does not support using an external index for assignment. Fall back
    // to the classical GPU impl
    if (should_use_raft(config_)) {
#if defined USE_NVIDIA_RAFT
        if (pq.assign_index) {
            fprintf(stderr,
                    "WARN: The Product Quantizer's assign_index will be ignored with RAFT enabled.\n");
        }
        // first initialize the index. The PQ centroids will be updated
        // retroactively.
        setIndex_(
                resources_.get(),
                this->d,
                this->nlist,
                metric_type,
                metric_arg,
                subQuantizers_,
                bitsPerCode_,
                ivfpqConfig_.useFloat16LookupTables,
                ivfpqConfig_.useMMCodeDistance,
                ivfpqConfig_.interleavedLayout,
                pq.centroids.data(),
                ivfpqConfig_.indicesOptions,
                config_.memorySpace);
        // No need to copy the data to host
        const raft::device_resources& raft_handle =
                resources_->getRaftHandleCurrentDevice();

        raft::neighbors::ivf_pq::index_params raft_idx_params;
        raft_idx_params.n_lists = nlist;
        raft_idx_params.metric = metricFaissToRaft(metric_type, false);
        raft_idx_params.kmeans_trainset_fraction =
                static_cast<double>(cp.max_points_per_centroid * nlist) /
                static_cast<double>(n);
        raft_idx_params.kmeans_n_iters = cp.niter;
        raft_idx_params.pq_bits = bitsPerCode_;
        raft_idx_params.pq_dim = subQuantizers_;
        raft_idx_params.conservative_memory_allocation = false;
        raft_idx_params.add_data_on_build = false;

        auto raftIndex_ = std::static_pointer_cast<RaftIVFPQ, IVFPQ>(index_);

        raft::neighbors::ivf_pq::index<idx_t> raft_ivfpq_index =
                raft::neighbors::ivf_pq::build<float, idx_t>(
                        raft_handle, raft_idx_params, x, n, (idx_t)d);

        auto raft_centers = raft::make_device_matrix<float>(
                raft_handle,
                raft_ivfpq_index.n_lists(),
                raft_ivfpq_index.dim());
        raft::neighbors::ivf_pq::helpers::extract_centers(
                raft_handle, raft_ivfpq_index, raft_centers.view());

        quantizer->train(nlist, raft_centers.data_handle());
        quantizer->add(nlist, raft_centers.data_handle());

        raft::copy(
                pq.get_centroids(0, 0),
                raft_ivfpq_index.pq_centers().data_handle(),
                raft_ivfpq_index.pq_centers().size(),
                raft_handle.get_stream());
        raft_handle.sync_stream();
        raftIndex_->setRaftIndex(std::move(raft_ivfpq_index));
#else
        FAISS_THROW_MSG(
                "RAFT has not been compiled into the current version so it cannot be used.");
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
        trainResidualQuantizer_(n, hostData.data());

        setIndex_(
                resources_.get(),
                this->d,
                this->nlist,
                metric_type,
                metric_arg,
                subQuantizers_,
                bitsPerCode_,
                ivfpqConfig_.useFloat16LookupTables,
                ivfpqConfig_.useMMCodeDistance,
                ivfpqConfig_.interleavedLayout,
                pq.centroids.data(),
                ivfpqConfig_.indicesOptions,
                config_.memorySpace);
        updateQuantizer();
    }
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFPQ>(index_);

    if (reserveMemoryVecs_) {
        index_->reserveMemory(reserveMemoryVecs_);
    }

    index_->setPrecomputedCodes(quantizer, usePrecomputedTables_);

    FAISS_ASSERT(index_);

    this->is_trained = true;
}

void GpuIndexIVFPQ::setIndex_(
        GpuResources* resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool useFloat16LookupTables,
        bool useMMCodeDistance,
        bool interleavedLayout,
        float* pqCentroidData,
        IndicesOptions indicesOptions,
        MemorySpace space) {
    if (should_use_raft(config_)) {
#if defined USE_NVIDIA_RAFT
        index_.reset(new RaftIVFPQ(
                resources,
                dim,
                nlist,
                metric,
                metricArg,
                numSubQuantizers,
                bitsPerSubQuantizer,
                useFloat16LookupTables,
                useMMCodeDistance,
                interleavedLayout,
                pqCentroidData,
                indicesOptions,
                space));
#else
        FAISS_THROW_MSG(
                "RAFT has not been compiled into the current version so it cannot be used.");
#endif
    } else {
        index_.reset(new IVFPQ(
                resources,
                dim,
                nlist,
                metric,
                metricArg,
                numSubQuantizers,
                bitsPerSubQuantizer,
                useFloat16LookupTables,
                useMMCodeDistance,
                interleavedLayout,
                pqCentroidData,
                indicesOptions,
                space));
    }
}

void GpuIndexIVFPQ::verifyPQSettings_() const {
    // Our implementation has these restrictions:

    // Must have some number of lists
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be >0");

    // up to a single byte per code
    if (should_use_raft(config_)) {
        if (!ivfpqConfig_.interleavedLayout) {
            fprintf(stderr,
                    "WARN: interleavedLayout is set to False with RAFT enabled. This will be ignored.\n");
        }
        FAISS_THROW_IF_NOT_FMT(
                bitsPerCode_ >= 4 && bitsPerCode_ <= 8,
                "Bits per code must be within closed range [4,8] (passed %d)",
                bitsPerCode_);
        FAISS_THROW_IF_NOT_FMT(
                (bitsPerCode_ * subQuantizers_) % 8 == 0,
                "`Bits per code * number of sub-quantizers must be a multiple of 8, (passed %u * %u = %u).",
                bitsPerCode_,
                subQuantizers_,
                bitsPerCode_ * subQuantizers_);
    } else {
        if (ivfpqConfig_.interleavedLayout) {
            FAISS_THROW_IF_NOT_FMT(
                    bitsPerCode_ == 4 || bitsPerCode_ == 5 ||
                            bitsPerCode_ == 6 || bitsPerCode_ == 8,
                    "Bits per code must be between 4, 5, 6 or 8 (passed %d)",
                    bitsPerCode_);
        } else {
            FAISS_THROW_IF_NOT_FMT(
                    bitsPerCode_ == 8,
                    "Bits per code must be 8 (passed %d)",
                    bitsPerCode_);
        }
    }

    // The number of bytes per encoded vector must be one we support
    FAISS_THROW_IF_NOT_FMT(
            ivfpqConfig_.interleavedLayout ||
                    IVFPQ::isSupportedPQCodeLength(subQuantizers_),
            "Number of bytes per encoded vector / sub-quantizers (%d) "
            "is not supported",
            subQuantizers_);

    if (!should_use_raft(config_)) {
        // Sub-quantizers must evenly divide dimensions available
        FAISS_THROW_IF_NOT_FMT(
                this->d % subQuantizers_ == 0,
                "Number of sub-quantizers (%d) must be an "
                "even divisor of the number of dimensions (%d)",
                subQuantizers_,
                this->d);

        // We must have enough shared memory on the current device to store
        // our lookup distances
        int lookupTableSize = sizeof(float);
        if (ivfpqConfig_.useFloat16LookupTables) {
            lookupTableSize = sizeof(half);
        }

        // 64 bytes per code is only supported with usage of float16, at 2^8
        // codes per subquantizer
        size_t requiredSmemSize =
                lookupTableSize * subQuantizers_ * utils::pow2(bitsPerCode_);
        size_t smemPerBlock = getMaxSharedMemPerBlock(config_.device);

        FAISS_THROW_IF_NOT_FMT(
                requiredSmemSize <= getMaxSharedMemPerBlock(config_.device),
                "Device %d has %zu bytes of shared memory, while "
                "%d bits per code and %d sub-quantizers requires %zu "
                "bytes. Consider useFloat16LookupTables and/or "
                "reduce parameters",
                config_.device,
                smemPerBlock,
                bitsPerCode_,
                subQuantizers_,
                requiredSmemSize);
    }
}

} // namespace gpu
} // namespace faiss
