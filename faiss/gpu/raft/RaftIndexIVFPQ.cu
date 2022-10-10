/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/raft/RaftIndexIVFPQ.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/utils/utils.h>
#include <faiss/gpu/impl/IVFPQ.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <raft/spatial/knn/ivf_pq.cuh>
#include <raft/spatial/knn/ivf_pq_types.hpp>

#include <limits>

namespace faiss {
namespace gpu {
/**
 *     GpuIndexIVFPQ(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            int subQuantizers,
            int bitsPerCode,
            faiss::MetricType metric,
            GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig())
 * @param provider
 * @param index
 * @param config
 */
RaftIndexIVFPQ::RaftIndexIVFPQ(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFPQ* index,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVFPQ(provider, index, config),
          pq(index->pq),
          ivfpqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(0),
          bitsPerCode_(0),
          reserveMemoryVecs_(0) {
    copyFrom(index);
}

RaftIndexIVFPQ::RaftIndexIVFPQ(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        int subQuantizers,
        int bitsPerCode,
        faiss::MetricType metric,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVFPQ(provider, dims, nlist, subQuantizers, bitsPerCode,  metric, config),
          pq(dims, subQuantizers, bitsPerCode),
          ivfpqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(subQuantizers),
          bitsPerCode_(bitsPerCode),
          reserveMemoryVecs_(0) {
    verifySettings_();

    // We haven't trained ourselves, so don't construct the PQ index yet
    this->is_trained = false;
}

RaftIndexIVFPQ::~RaftIndexIVFPQ() {}

void RaftIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* index) {
//    DeviceScope scope(config_.device);
//
//    GpuIndexIVF::copyFrom(index);
//
//    // Clear out our old data
//    index_.reset();
//
//    pq = index->pq;
//    subQuantizers_ = index->pq.M;
//    bitsPerCode_ = index->pq.nbits;
//
//    // We only support this
//    FAISS_THROW_IF_NOT_MSG(
//            ivfpqConfig_.interleavedLayout || index->pq.nbits == 8,
//            "GPU: only pq.nbits == 8 is supported");
//    FAISS_THROW_IF_NOT_MSG(
//            index->by_residual, "GPU: only by_residual = true is supported");
//    FAISS_THROW_IF_NOT_MSG(
//            index->polysemous_ht == 0, "GPU: polysemous codes not supported");
//
//    verifySettings_();
//
//    // The other index might not be trained
//    if (!index->is_trained) {
//        // copied in GpuIndex::copyFrom
//        FAISS_ASSERT(!is_trained);
//        return;
//    }
//
//    // Copy our lists as well
//    // The product quantizer must have data in it
//    FAISS_ASSERT(index->pq.centroids.size() > 0);
//    index_.reset(new IVFPQ(
//            resources_.get(),
//            index->metric_type,
//            index->metric_arg,
//            quantizer->getGpuData(),
//            subQuantizers_,
//            bitsPerCode_,
//            ivfpqConfig_.useFloat16LookupTables,
//            ivfpqConfig_.useMMCodeDistance,
//            ivfpqConfig_.interleavedLayout,
//            (float*)index->pq.centroids.data(),
//            ivfpqConfig_.indicesOptions,
//            config_.memorySpace));
//    // Doesn't make sense to reserve memory here
//    index_->setPrecomputedCodes(usePrecomputedTables_);
//
//    // Copy all of the IVF data
//    index_->copyInvertedListsFrom(index->invlists);
}

void RaftIndexIVFPQ::copyTo(faiss::IndexIVFPQ* index) const {
//    DeviceScope scope(config_.device);
//
//    // We must have the indices in order to copy to ourselves
//    FAISS_THROW_IF_NOT_MSG(
//            ivfpqConfig_.indicesOptions != INDICES_IVF,
//            "Cannot copy to CPU as GPU index doesn't retain "
//            "indices (INDICES_IVF)");
//
//    GpuIndexIVF::copyTo(index);
//
//    //
//    // IndexIVFPQ information
//    //
//    index->by_residual = true;
//    index->use_precomputed_table = 0;
//    index->code_size = subQuantizers_;
//    index->pq = faiss::ProductQuantizer(this->d, subQuantizers_, bitsPerCode_);
//
//    index->do_polysemous_training = false;
//    index->polysemous_training = nullptr;
//
//    index->scan_table_threshold = 0;
//    index->max_codes = 0;
//    index->polysemous_ht = 0;
//    index->precomputed_table.clear();
//
//    auto ivf = new ArrayInvertedLists(nlist, index->code_size);
//    index->replace_invlists(ivf, true);
//
//    if (index_) {
//        // Copy IVF lists
//        index_->copyInvertedListsTo(ivf);
//
//        // Copy PQ centroids
//        auto devPQCentroids = index_->getPQCentroids();
//        index->pq.centroids.resize(devPQCentroids.numElements());
//
//        fromDevice<float, 3>(
//                devPQCentroids,
//                index->pq.centroids.data(),
//                resources_->getDefaultStream(config_.device));
//
//        if (usePrecomputedTables_) {
//            index->precompute_table();
//        }
//    }
}

void RaftIndexIVFPQ::reserveMemory(size_t numVecs) {
    reserveMemoryVecs_ = numVecs;
    if (index_) {
        DeviceScope scope(config_.device);
        index_->reserveMemory(numVecs);
    }
}

void RaftIndexIVFPQ::setPrecomputedCodes(bool enable) {
    usePrecomputedTables_ = enable;
    if (index_) {
        DeviceScope scope(config_.device);
        index_->setPrecomputedCodes(enable);
    }

    verifySettings_();
}

bool RaftIndexIVFPQ::getPrecomputedCodes() const {
    return usePrecomputedTables_;
}

int RaftIndexIVFPQ::getNumSubQuantizers() const {
    return subQuantizers_;
}

int RaftIndexIVFPQ::getBitsPerCode() const {
    return bitsPerCode_;
}

int RaftIndexIVFPQ::getCentroidsPerSubQuantizer() const {
    return utils::pow2(bitsPerCode_);
}

size_t RaftIndexIVFPQ::reclaimMemory() {
    if (index_) {
        DeviceScope scope(config_.device);
        return index_->reclaimMemory();
    }

    return 0;
}

void RaftIndexIVFPQ::reset() {
    if (raft_knn_index.has_value()) {
        raft_knn_index.reset();
        this->ntotal = 0;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

void RaftIndexIVFPQ::train(Index::idx_t n, const float* x) {
    raft::common::nvtx::range<raft::common::nvtx::domain::raft> fun_scope(
            "RaftIndexIVFFlat::train (%ld)", n);

    std::cout << "Calling train() with " << n << " rows" << std::endl;

    uint32_t start = raft::curTimeMillis();
    if (this->is_trained) {
        FAISS_ASSERT(raft_knn_index.has_value());
        return;
    }

    raft::spatial::knn::ivf_pq::index_params raft_idx_params;
    raft_idx_params.n_lists = nlist;
    raft_idx_params.metric = raft::distance::DistanceType::L2Expanded;
    raft_idx_params.add_data_on_build = false;
    raft_idx_params.kmeans_n_iters = 100;

    raft_knn_index.emplace(
            raft::spatial::knn::ivf_pq::build(raft_handle, raft_idx_params,
                                                const_cast<float*>(x),
                                                n, (faiss::Index::idx_t)d));

    raft_handle.sync_stream();
    uint32_t stop = raft::curTimeMillis();

    std::cout << "train took " << (stop - start) << "ms. " << std::endl;
    this->is_trained = true;
}

void RaftIndexIVFPQ::addImpl_(int n, const float* x, const Index::idx_t* xids) {
    // Device is already set in GpuIndex::add
    FAISS_ASSERT(is_trained);
    FAISS_ASSERT(n > 0);

    // but keep the ntotal based on the total number of vectors that we
    // attempted to add
    std::cout << "Calling addImpl_ with " << n << " vectors." << std::endl;

    raft_knn_index.emplace(raft::spatial::knn::ivf_pq::extend(
            raft_handle, raft_knn_index.value(), x, xids, (Index::idx_t)n));
    this->ntotal += n;

    ntotal += n;
}

void RaftIndexIVFPQ::searchImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        Index::idx_t* labels) const {
    // Device is already set in GpuIndex::search
    FAISS_ASSERT(index_);
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= nlist);

    raft::common::nvtx::range<raft::common::nvtx::domain::raft> fun_scope(
            "RaftIndexIVFFlat::searchImpl_ (%ld)", n);

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= nlist);

    raft::spatial::knn::ivf_pq::search_params pams;
    pams.n_probes = nprobe;
    raft::spatial::knn::ivf_pq::search<float, faiss::Index::idx_t>(
            raft_handle,
            pams,
            *raft_knn_index,
            const_cast<float*>(x),
            static_cast<std::uint32_t>(n),
            static_cast<std::uint32_t>(k),
            labels,
            distances);

    raft_handle.sync_stream();
}

int RaftIndexIVFPQ::getListLength(int listId) const {
    FAISS_ASSERT(index_);
    DeviceScope scope(config_.device);

    return index_->getListLength(listId);
}

std::vector<uint8_t> RaftIndexIVFPQ::getListVectorData(
        int listId,
        bool gpuFormat) const {
    FAISS_ASSERT(index_);
    DeviceScope scope(config_.device);

    return index_->getListVectorData(listId, gpuFormat);
}

std::vector<Index::idx_t> RaftIndexIVFPQ::getListIndices(int listId) const {
    FAISS_ASSERT(index_);
    DeviceScope scope(config_.device);

    return index_->getListIndices(listId);
}

void RaftIndexIVFPQ::verifySettings_() const {
    // Our implementation has these restrictions:

    // Must have some number of lists
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be >0");

    // up to a single byte per code
    if (ivfpqConfig_.interleavedLayout) {
        FAISS_THROW_IF_NOT_FMT(
                bitsPerCode_ == 4 || bitsPerCode_ == 5 || bitsPerCode_ == 6 ||
                bitsPerCode_ == 8,
                "Bits per code must be between 4, 5, 6 or 8 (passed %d)",
                bitsPerCode_);

    } else {
        FAISS_THROW_IF_NOT_FMT(
                bitsPerCode_ == 8,
                "Bits per code must be 8 (passed %d)",
                bitsPerCode_);
    }

    // Sub-quantizers must evenly divide dimensions available
    FAISS_THROW_IF_NOT_FMT(
            this->d % subQuantizers_ == 0,
            "Number of sub-quantizers (%d) must be an "
            "even divisor of the number of dimensions (%d)",
            subQuantizers_,
            this->d);

    // The number of bytes per encoded vector must be one we support
    FAISS_THROW_IF_NOT_FMT(
            ivfpqConfig_.interleavedLayout ||
            IVFPQ::isSupportedPQCodeLength(subQuantizers_),
            "Number of bytes per encoded vector / sub-quantizers (%d) "
            "is not supported",
            subQuantizers_);

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

} // namespace gpu
} // namespace faiss
