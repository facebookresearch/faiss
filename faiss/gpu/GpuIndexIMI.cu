/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexPQ.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndexIMI.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/utils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <iostream>
#include <limits>
#include <memory>

namespace faiss {
namespace gpu {

/// Size above which we page copies from the CPU to GPU (non-paged
/// memory usage)
constexpr size_t kNonPinnedPageSize = (size_t)256 * 1024 * 1024;

// Default size for which we page add or search
constexpr size_t kAddPageSize = (size_t)256 * 1024 * 1024;

// Or, maximum number of vectors to consider per page of add or search
constexpr size_t kAddVecSize = (size_t)512 * 1024;

// Use a smaller search size, as precomputed code usage on IVFPQ
// requires substantial amounts of memory
// FIXME: parameterize based on algorithm need
constexpr size_t kSearchVecSize = (size_t)32 * 1024;

GpuIndexIMI::GpuIndexIMI(
        GpuResourcesProvider* provider,
        int dims,
        int coarseCodebookSize,
        GpuIndexIMIConfig config)
        : GpuIndex(
                  provider->getResources(),
                  dims,
                  faiss::MetricType::METRIC_L2,
                  0,
                  config),
          nlist(coarseCodebookSize * coarseCodebookSize),
          nprobe(1),
          imiConfig_(config) {
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be > 0");
    quantizer = new GpuMultiIndex2(
            provider, dims, coarseCodebookSize, imiConfig_.multiIndexConfig);
}

GpuIndexIMI::GpuIndexIMI(
        std::shared_ptr<GpuResources> resources,
        int dims,
        int coarseCodebookSize,
        GpuIndexIMIConfig config)
        : GpuIndex(resources, dims, faiss::MetricType::METRIC_L2, 0, config),
          nlist(coarseCodebookSize * coarseCodebookSize),
          nprobe(1),
          imiConfig_(config) {
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be > 0");
    quantizer = new GpuMultiIndex2(
            resources, dims, coarseCodebookSize, imiConfig_.multiIndexConfig);
}

GpuIndexIMI::~GpuIndexIMI() {
    delete quantizer;
}

size_t GpuIndexIMI::calcMemorySpaceSizeCoarseQuantizer(
        int numVecsTotal,
        int dimPerCodebook,
        bool useFloat16) {
    return GpuMultiIndex2::calcMemorySpaceSize(
            numVecsTotal, dimPerCodebook, useFloat16);
}

GpuMultiIndex2* GpuIndexIMI::getQuantizer() {
    return quantizer;
}

void GpuIndexIMI::copyFrom(const faiss::IndexIVF* index) {
    DeviceScope scope(config_.device);

    GpuIndex::copyFrom(index);

    auto multiIndexCpu =
            dynamic_cast<faiss::MultiIndexQuantizer*>(index->quantizer);

    FAISS_THROW_IF_NOT_MSG(
            multiIndexCpu,
            "Only MultiIndexQuantizer is supported for the coarse quantizer "
            "for copying from an IndexIVF into a GpuIndexIMI");

    delete quantizer;

    GpuMultiIndex2Config config = imiConfig_.multiIndexConfig;
    // FIXME: inherit our same device
    config.device = config_.device;
    quantizer = new GpuMultiIndex2(resources_, multiIndexCpu, config);

    FAISS_ASSERT(index->nlist > 0);
    FAISS_THROW_IF_NOT_FMT(
            index->nlist <= (idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports %zu inverted lists",
            (size_t)std::numeric_limits<int>::max());

    this->nlist = index->nlist;

    FAISS_THROW_IF_NOT_FMT(
            nprobe <= this->nlist, "nprobe must be <= %d", this->nlist);
    FAISS_THROW_IF_NOT_FMT(
            nprobe <= getMaxKSelection() * getMaxKSelection(),
            "nprobe must be <= %d",
            getMaxKSelection() * getMaxKSelection());

    if (index->nprobe > getMaxKSelection()) {
        std::cout
                << "WARNING: nprobe must be <= " << getMaxKSelection()
                << " to ensure the correctness of the multi-sequence algorithm"
                << std::endl;
    }

    this->nprobe = index->nprobe;

    if (!index->is_trained) {
        // copied in GpuIndex::copyFrom
        FAISS_ASSERT(!is_trained && ntotal == 0);
        return;
    }

    // copied in GpuIndex::copyFrom
    // ntotal can exceed max int, but the number of vectors per inverted
    // list cannot exceed this. We check this in the subclasses.
    FAISS_ASSERT(is_trained && (ntotal == index->ntotal));

    // Since we're trained, the quantizer must have data
    FAISS_ASSERT(index->quantizer->ntotal > 0);
}

void GpuIndexIMI::copyTo(faiss::IndexIVF* index) const {
    DeviceScope scope(config_.device);
    //
    // Index information
    //
    GpuIndex::copyTo(index);

    //
    // IndexIVF information
    //
    index->nlist = this->nlist;
    index->nprobe = this->nprobe;

    // Construct and copy the appropriate quantizer
    faiss::MultiIndexQuantizer* q = nullptr;

    if (this->metric_type == faiss::METRIC_L2) {
        q = new faiss::MultiIndexQuantizer(
                this->d,
                this->quantizer->getNumCodebooks(),
                utils::log2(this->quantizer->getCodebookSize()));
    } else {
        // we should have one of the above metrics
        FAISS_ASSERT(false);
    }

    FAISS_ASSERT(quantizer);
    quantizer->copyTo(q);

    if (index->own_fields) {
        delete index->quantizer;
    }

    index->quantizer = q;
    index->quantizer_trains_alone = 1;
    index->own_fields = true;
    index->cp = this->cp;
    index->make_direct_map(false);
}

int GpuIndexIMI::getNumLists() const {
    return nlist;
}

void GpuIndexIMI::setNumProbes(int nprobe) {
    FAISS_THROW_IF_NOT_MSG(
            quantizer->is_trained, "Index must be first trained");
    FAISS_THROW_IF_NOT_MSG(nprobe > 0, "nprobe must be > 0");
    FAISS_THROW_IF_NOT_FMT(
            nprobe <= this->nlist, "nprobe must be <= %d", this->nlist);
    FAISS_THROW_IF_NOT_FMT(
            nprobe <= getMaxKSelection() * getMaxKSelection(),
            "nprobe must be <= %d",
            getMaxKSelection() * getMaxKSelection());

    if (nprobe > getMaxKSelection()) {
        std::cout
                << "WARNING: nprobe must be <= " << getMaxKSelection()
                << " to ensure the correctness of the multi-sequence algorithm"
                << std::endl;
    }

    this->nprobe = nprobe;
}

int GpuIndexIMI::getNumProbes() const {
    return nprobe;
}

bool GpuIndexIMI::addImplRequiresIDs_() const {
    // All IVF indices have storage for IDs
    return true;
}

void GpuIndexIMI::add_with_ids(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    if (n == 0) {
        // nothing to add
        return;
    }

    std::vector<idx_t> generatedIds;

    // Generate IDs if we need them
    if (!ids && addImplRequiresIDs_()) {
        generatedIds = std::vector<idx_t>(n);

        for (idx_t i = 0; i < n; ++i) {
            generatedIds[i] = this->ntotal + i;
        }
    }

    DeviceScope scope(imiConfig_.device);
    addPaged_((int)n, x, ids ? ids : generatedIds.data());
}

void GpuIndexIMI::addPaged_(int n, const float* x, const idx_t* ids) {
    if (n > 0) {
        size_t totalSize = (size_t)n * this->d * sizeof(float);

        if (totalSize > kAddPageSize || n > kAddVecSize) {
            // How many vectors fit into kAddPageSize?
            size_t maxNumVecsForPageSize =
                    kAddPageSize / ((size_t)this->d * sizeof(float));

            // Always add at least 1 vector, if we have huge vectors
            maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);

            size_t tileSize = std::min((size_t)n, maxNumVecsForPageSize);
            tileSize = std::min(tileSize, kSearchVecSize);

            for (size_t i = 0; i < (size_t)n; i += tileSize) {
                size_t curNum = std::min(tileSize, n - i);

                addPage_(
                        curNum,
                        x + i * (size_t)this->d,
                        ids ? ids + i : nullptr);
            }
        } else {
            addPage_(n, x, ids);
        }
    }
}

void GpuIndexIMI::addPage_(int n, const float* x, const idx_t* ids) {
    // At this point, `x` can be resident on CPU or GPU, and `ids` may be
    // resident on CPU, GPU or may be null.
    //
    // Before continuing, we guarantee that all data will be resident on the
    // GPU.
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // FIXME: change location to addPaged_
    std::unique_ptr<float[]> subQueries(new float[n * this->d]);
    fvec_split(
            subQueries.get(),
            quantizer->getNumCodebooks(),
            x,
            (size_t)n,
            quantizer->getSubDim());

    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            imiConfig_.device,
            const_cast<float*>(subQueries.get()),
            stream,
            {quantizer->getNumCodebooks() * (int)n, quantizer->getSubDim()});

    CudaEvent copyEnd(stream);

    if (ids) {
        auto indices = toDeviceTemporary<idx_t, 1>(
                resources_.get(),
                imiConfig_.device,
                const_cast<idx_t*>(ids),
                stream,
                {n});

        addImpl_(n, vecs.data(), ids ? indices.data() : nullptr);
    } else {
        addImpl_(n, vecs.data(), nullptr);
    }

    // synchronizing to ensure that subQueries has not been deleted before copy
    // ends
    copyEnd.cpuWaitOnEvent();
}

void GpuIndexIMI::trainQuantizer_(faiss::idx_t n, const float* x) {
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (this->verbose) {
            printf("IMI quantizer does not need training.\n");
        }

        return;
    }

    if (this->verbose) {
        printf("Training IMI quantizer on %ld vectors in %dD\n", n, d);
    }
    quantizer->train(n, x);
}

void GpuIndexIMI::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    // Maximum k-selection supported is based on the CUDA SDK
    FAISS_THROW_IF_NOT_FMT(
            k <= (idx_t)getMaxKSelection(),
            "GPU index only supports k <= %d (requested %d)",
            getMaxKSelection(),
            (int)k); // select limitation

    if (n == 0 || k == 0) {
        // nothing to search
        return;
    }

    DeviceScope scope(imiConfig_.device);
    auto stream = resources_->getDefaultStream(imiConfig_.device);

    // We guarantee that the searchImpl_ will be called with device-resident
    // pointers.

    // The input vectors may be too large for the GPU, but we still
    // assume that the output distances and labels are not.
    // Go ahead and make space for output distances and labels on the
    // GPU.
    // If we reach a point where all inputs are too big, we can add
    // another level of tiling.
    auto outDistances = toDeviceTemporary<float, 2>(
            resources_.get(),
            imiConfig_.device,
            distances,
            stream,
            {(int)n, (int)k});

    auto outLabels = toDeviceTemporary<idx_t, 2>(
            resources_.get(),
            imiConfig_.device,
            labels,
            stream,
            {(int)n, (int)k});

    bool usePaged = false;

    if (getDeviceForAddress(x) == -1) {
        // It is possible that the user is querying for a vector set size
        // `x` that won't fit on the GPU.
        // In this case, we will have to handle paging of the data from CPU
        // -> GPU.
        // Currently, we don't handle the case where the output data won't
        // fit on the GPU (e.g., n * k is too large for the GPU memory).
        size_t dataSize = (size_t)n * this->d * sizeof(float);

        if (dataSize >= minPagedSize_ || imiConfig_.forcePinnedMemory) {
            searchFromCpuPaged_(
                    n, x, k, outDistances.data(), outLabels.data(), params);
            usePaged = true;
        }
    }

    if (!usePaged) {
        searchNonPaged_(n, x, k, outDistances.data(), outLabels.data(), params);
    }

    // Copy back if necessary
    fromDevice<float, 2>(outDistances, distances, stream);
    fromDevice<faiss::idx_t, 2>(outLabels, labels, stream);
}

void GpuIndexIMI::searchNonPaged_(
        int n,
        const float* x,
        int k,
        float* outDistancesData,
        idx_t* outIndicesData,
        const SearchParameters* params) const {
    auto stream = resources_->getDefaultStream(imiConfig_.device);

    std::unique_ptr<float[]> subQueries(new float[n * this->d]);
    fvec_split(
            subQueries.get(),
            quantizer->getNumCodebooks(),
            x,
            (size_t)n,
            quantizer->getSubDim());

    // Make sure arguments are on the device we desire; use temporary
    // memory allocations to move it if necessary
    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            imiConfig_.device,
            const_cast<float*>(subQueries.get()),
            stream,
            {quantizer->getNumCodebooks() * n, quantizer->getSubDim()});

    searchImpl_(n, vecs.data(), k, outDistancesData, outIndicesData, params);

    CudaEvent searchEnd(stream, true);

    // synchronizing to ensure that subQueries has not been deleted
    // copyEnd.cpuWaitOnEvent();
    searchEnd.cpuWaitOnEvent();
}

void GpuIndexIMI::searchFromCpuPaged_(
        int n,
        const float* x,
        int k,
        float* outDistancesData,
        idx_t* outIndicesData,
        const SearchParameters* params) const {
    Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
    Tensor<idx_t, 2, true> outIndices(outIndicesData, {n, k});

    // Is pinned memory available?
    auto pinnedAlloc = resources_->getPinnedMemory();
    int pageSizeInVecs =
            (int)((pinnedAlloc.second / 2) / (sizeof(float) * this->d));

    if (!pinnedAlloc.first || pageSizeInVecs < 1) {
        // Just page without overlapping copy with compute
        int batchSize = utils::nextHighestPowerOf2(
                (int)((size_t)kNonPinnedPageSize / (sizeof(float) * this->d)));

        for (int cur = 0; cur < n; cur += batchSize) {
            int num = std::min(batchSize, n - cur);

            auto outDistancesSlice = outDistances.narrowOutermost(cur, num);
            auto outIndicesSlice = outIndices.narrowOutermost(cur, num);

            searchNonPaged_(
                    num,
                    x + (size_t)cur * this->d,
                    k,
                    outDistancesSlice.data(),
                    outIndicesSlice.data(),
                    params);
        }
        return;
    }

    //
    // Pinned memory is available, so we can overlap copy with compute.
    // We use two pinned memory buffers, and triple-buffer the
    // procedure:
    //
    // 1 CPU copy -> pinned
    // 2 pinned copy -> GPU
    // 3 GPU compute
    //
    // 1 2 3 1 2 3 ...   (pinned buf A)
    //   1 2 3 1 2 ...   (pinned buf B)
    //     1 2 3 1 ...   (pinned buf A)
    // time ->
    //
    auto defaultStream = resources_->getDefaultStream(imiConfig_.device);
    auto copyStream = resources_->getAsyncCopyStream(imiConfig_.device);

    FAISS_ASSERT(
            (size_t)pageSizeInVecs * this->d <=
            (size_t)std::numeric_limits<int>::max());

    float* bufPinnedA = (float*)pinnedAlloc.first;
    float* bufPinnedB = bufPinnedA + (size_t)pageSizeInVecs * this->d;
    float* bufPinned[2] = {bufPinnedA, bufPinnedB};

    // Reserve space on the GPU for the destination of the pinned buffer
    // copy
    DeviceTensor<float, 2, true> bufGpuA(
            resources_.get(),
            makeTempAlloc(AllocType::Other, defaultStream),
            {quantizer->getNumCodebooks() * pageSizeInVecs,
             quantizer->getSubDim()});
    DeviceTensor<float, 2, true> bufGpuB(
            resources_.get(),
            makeTempAlloc(AllocType::Other, defaultStream),
            {quantizer->getNumCodebooks() * pageSizeInVecs,
             quantizer->getSubDim()});
    DeviceTensor<float, 2, true>* bufGpus[2] = {&bufGpuA, &bufGpuB};

    // Copy completion events for the pinned buffers
    std::unique_ptr<CudaEvent> eventPinnedCopyDone[2];

    // Execute completion events for the GPU buffers
    std::unique_ptr<CudaEvent> eventGpuExecuteDone[2];

    // All offsets are in terms of number of vectors; they remain within
    // int bounds (as this function only handles max in vectors)

    // Current start offset for buffer 1
    int cur1 = 0;
    int cur1BufIndex = 0;

    // Current start offset for buffer 2
    int cur2 = -1;
    int cur2BufIndex = 0;

    // Current start offset for buffer 3
    int cur3 = -1;
    int cur3BufIndex = 0;

    while (cur3 < n) {
        // Start async pinned -> GPU copy first (buf 2)
        if (cur2 != -1 && cur2 < n) {
            // Copy pinned to GPU
            int numToCopy = std::min(pageSizeInVecs, n - cur2);

            // Make sure any previous execution has completed before continuing
            auto& eventPrev = eventGpuExecuteDone[cur2BufIndex];
            if (eventPrev.get()) {
                eventPrev->streamWaitOnEvent(copyStream);
            }

            CUDA_VERIFY(cudaMemcpyAsync(
                    bufGpus[cur2BufIndex]->data(),
                    bufPinned[cur2BufIndex],
                    (size_t)numToCopy * this->d * sizeof(float),
                    cudaMemcpyHostToDevice,
                    copyStream));

            // Mark a completion event in this stream
            eventPinnedCopyDone[cur2BufIndex].reset(new CudaEvent(copyStream));

            // We pick up from here
            cur3 = cur2;
            cur2 += numToCopy;
            cur2BufIndex = (cur2BufIndex == 0) ? 1 : 0;
        }

        if (cur3 != -1 && cur3 < n) {
            // Process on GPU
            int numToProcess = std::min(pageSizeInVecs, n - cur3);

            // Make sure the previous copy has completed before continuing
            auto& eventPrev = eventPinnedCopyDone[cur3BufIndex];
            FAISS_ASSERT(eventPrev.get());

            eventPrev->streamWaitOnEvent(defaultStream);

            // Create tensor wrappers
            // DeviceTensor<float, 2, true> input(bufGpus[cur3BufIndex]->data(),
            //                                    {numToProcess, this->d});
            auto outDistancesSlice =
                    outDistances.narrowOutermost(cur3, numToProcess);
            auto outIndicesSlice =
                    outIndices.narrowOutermost(cur3, numToProcess);

            searchImpl_(
                    numToProcess,
                    bufGpus[cur3BufIndex]->data(),
                    k,
                    outDistancesSlice.data(),
                    outIndicesSlice.data(),
                    params);

            // Create completion event
            eventGpuExecuteDone[cur3BufIndex].reset(
                    new CudaEvent(defaultStream));

            // We pick up from here
            cur3BufIndex = (cur3BufIndex == 0) ? 1 : 0;
            cur3 += numToProcess;
        }

        if (cur1 < n) {
            // Copy CPU mem to CPU pinned
            int numToCopy = std::min(pageSizeInVecs, n - cur1);

            // Make sure any previous copy has completed before continuing
            auto& eventPrev = eventPinnedCopyDone[cur1BufIndex];
            if (eventPrev.get()) {
                eventPrev->cpuWaitOnEvent();
            }

            fvec_split(
                    bufPinned[cur1BufIndex],
                    quantizer->getNumCodebooks(),
                    x + (size_t)cur1 * this->d,
                    (size_t)numToCopy,
                    quantizer->getSubDim());

            // We pick up from here
            cur2 = cur1;
            cur1 += numToCopy;
            cur1BufIndex = (cur1BufIndex == 0) ? 1 : 0;
        }
    }
}

} // namespace gpu
} // namespace faiss
