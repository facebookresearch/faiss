/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <algorithm>
#include <limits>
#include <memory>

namespace faiss {
namespace gpu {

/// Default CPU search size for which we use paged copies
constexpr idx_t kMinPageSize = (idx_t)256 * 1024 * 1024;

/// Size above which we page copies from the CPU to GPU (non-paged
/// memory usage)
constexpr idx_t kNonPinnedPageSize = (idx_t)256 * 1024 * 1024;

// Default size for which we page add or search
constexpr idx_t kAddPageSize = (idx_t)256 * 1024 * 1024;

// Or, maximum number of vectors to consider per page of add or search
constexpr idx_t kAddVecSize = (idx_t)512 * 1024;

// Use a smaller search size, as precomputed code usage on IVFPQ
// requires substantial amounts of memory
// FIXME: parameterize based on algorithm need
constexpr idx_t kSearchVecSize = (idx_t)32 * 1024;

/// Caches device major version
extern int device_major_version;

bool should_use_raft(GpuIndexConfig config_) {
    if (device_major_version < 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, config_.device);
        device_major_version = prop.major;
    }

    if (device_major_version < 7)
        return false;

    return config_.use_raft;
}

GpuIndex::GpuIndex(
        std::shared_ptr<GpuResources> resources,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        GpuIndexConfig config)
        : Index(dims, metric),
          resources_(resources),
          config_(config),
          minPagedSize_(kMinPageSize) {
    FAISS_THROW_IF_NOT_FMT(
            config_.device < getNumDevices(),
            "Invalid GPU device %d",
            config_.device);

    FAISS_THROW_IF_NOT_MSG(dims > 0, "Invalid number of dimensions");

    FAISS_THROW_IF_NOT_FMT(
            config_.memorySpace == MemorySpace::Device ||
                    (config_.memorySpace == MemorySpace::Unified &&
                     getFullUnifiedMemSupport(config_.device)),
            "Device %d does not support full CUDA 8 Unified Memory (CC 6.0+)",
            config_.device);

    metric_arg = metricArg;

    FAISS_ASSERT((bool)resources_);
    resources_->initializeForDevice(config_.device);
}

int GpuIndex::getDevice() const {
    return config_.device;
}

void GpuIndex::copyFrom(const faiss::Index* index) {
    d = index->d;
    metric_type = index->metric_type;
    metric_arg = index->metric_arg;
    ntotal = index->ntotal;
    is_trained = index->is_trained;
}

void GpuIndex::copyTo(faiss::Index* index) const {
    index->d = d;
    index->metric_type = metric_type;
    index->metric_arg = metric_arg;
    index->ntotal = ntotal;
    index->is_trained = is_trained;
}

void GpuIndex::setMinPagingSize(size_t size) {
    minPagedSize_ = size;
}

size_t GpuIndex::getMinPagingSize() const {
    return minPagedSize_;
}

void GpuIndex::add(idx_t n, const float* x) {
    // Pass to add_with_ids
    add_with_ids(n, x, nullptr);
}

void GpuIndex::add_with_ids(idx_t n, const float* x, const idx_t* ids) {
    DeviceScope scope(config_.device);
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

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

    addPaged_(n, x, ids ? ids : generatedIds.data());
}

void GpuIndex::addPaged_(idx_t n, const float* x, const idx_t* ids) {
    if (n > 0) {
        idx_t totalSize = n * this->d * sizeof(float);

        if (!should_use_raft(config_) &&
            (totalSize > kAddPageSize || n > kAddVecSize)) {
            // How many vectors fit into kAddPageSize?
            idx_t maxNumVecsForPageSize =
                    kAddPageSize / (this->d * sizeof(float));

            // Always add at least 1 vector, if we have huge vectors
            maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, idx_t(1));

            auto tileSize = std::min(n, maxNumVecsForPageSize);
            tileSize = std::min(tileSize, kSearchVecSize);

            for (idx_t i = 0; i < n; i += tileSize) {
                auto curNum = std::min(tileSize, n - i);

                addPage_(curNum, x + i * this->d, ids ? ids + i : nullptr);
            }
        } else {
            addPage_(n, x, ids);
        }
    }
}

void GpuIndex::addPage_(idx_t n, const float* x, const idx_t* ids) {
    // At this point, `x` can be resident on CPU or GPU, and `ids` may be
    // resident on CPU, GPU or may be null.
    //
    // Before continuing, we guarantee that all data will be resident on the
    // GPU.
    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(x),
            stream,
            {n, this->d});

    if (ids) {
        auto indices = toDeviceTemporary<idx_t, 1>(
                resources_.get(),
                config_.device,
                const_cast<idx_t*>(ids),
                stream,
                {n});

        addImpl_(n, vecs.data(), ids ? indices.data() : nullptr);
    } else {
        addImpl_(n, vecs.data(), nullptr);
    }
}

void GpuIndex::assign(idx_t n, const float* x, idx_t* labels, idx_t k) const {
    DeviceScope scope(config_.device);
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    validateKSelect(k);

    auto stream = resources_->getDefaultStream(config_.device);

    // We need to create a throw-away buffer for distances, which we don't use
    // but which we do need for the search call
    DeviceTensor<float, 2, true> distances(
            resources_.get(), makeTempAlloc(AllocType::Other, stream), {n, k});

    // Forward to search
    search(n, x, k, distances.data(), labels);
}

void GpuIndex::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    DeviceScope scope(config_.device);
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    validateKSelect(k);

    if (n == 0 || k == 0) {
        // nothing to search
        return;
    }

    auto stream = resources_->getDefaultStream(config_.device);

    // We guarantee that the searchImpl_ will be called with device-resident
    // pointers.

    // The input vectors may be too large for the GPU, but we still
    // assume that the output distances and labels are not.
    // Go ahead and make space for output distances and labels on the
    // GPU.
    // If we reach a point where all inputs are too big, we can add
    // another level of tiling.
    auto outDistances = toDeviceTemporary<float, 2>(
            resources_.get(), config_.device, distances, stream, {n, k});

    auto outLabels = toDeviceTemporary<idx_t, 2>(
            resources_.get(), config_.device, labels, stream, {n, k});

    bool usePaged = false;

    if (getDeviceForAddress(x) == -1) {
        // It is possible that the user is querying for a vector set size
        // `x` that won't fit on the GPU.
        // In this case, we will have to handle paging of the data from CPU
        // -> GPU.
        // Currently, we don't handle the case where the output data won't
        // fit on the GPU (e.g., n * k is too large for the GPU memory).
        size_t dataSize = (size_t)n * this->d * sizeof(float);

        if (dataSize >= minPagedSize_) {
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
    fromDevice<idx_t, 2>(outLabels, labels, stream);
}

void GpuIndex::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params) const {
    search(n, x, k, distances, labels, params);
    reconstruct_batch(n * k, labels, recons);
}

void GpuIndex::searchNonPaged_(
        idx_t n,
        const float* x,
        int k,
        float* outDistancesData,
        idx_t* outIndicesData,
        const SearchParameters* params) const {
    auto stream = resources_->getDefaultStream(config_.device);

    // Make sure arguments are on the device we desire; use temporary
    // memory allocations to move it if necessary
    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(x),
            stream,
            {n, this->d});

    searchImpl_(n, vecs.data(), k, outDistancesData, outIndicesData, params);
}

void GpuIndex::searchFromCpuPaged_(
        idx_t n,
        const float* x,
        int k,
        float* outDistancesData,
        idx_t* outIndicesData,
        const SearchParameters* params) const {
    Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
    Tensor<idx_t, 2, true> outIndices(outIndicesData, {n, k});

    // Is pinned memory available?
    auto pinnedAlloc = resources_->getPinnedMemory();
    idx_t pageSizeInVecs =
            ((pinnedAlloc.second / 2) / (sizeof(float) * this->d));

    if (!pinnedAlloc.first || pageSizeInVecs < 1) {
        // Just page without overlapping copy with compute
        idx_t batchSize = utils::nextHighestPowerOf2(
                (kNonPinnedPageSize / (sizeof(float) * this->d)));

        for (idx_t cur = 0; cur < n; cur += batchSize) {
            auto num = std::min(batchSize, n - cur);

            auto outDistancesSlice = outDistances.narrowOutermost(cur, num);
            auto outIndicesSlice = outIndices.narrowOutermost(cur, num);

            searchNonPaged_(
                    num,
                    x + cur * this->d,
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
    auto defaultStream = resources_->getDefaultStream(config_.device);
    auto copyStream = resources_->getAsyncCopyStream(config_.device);

    float* bufPinnedA = (float*)pinnedAlloc.first;
    float* bufPinnedB = bufPinnedA + (size_t)pageSizeInVecs * this->d;
    float* bufPinned[2] = {bufPinnedA, bufPinnedB};

    // Reserve space on the GPU for the destination of the pinned buffer
    // copy
    DeviceTensor<float, 2, true> bufGpuA(
            resources_.get(),
            makeTempAlloc(AllocType::Other, defaultStream),
            {pageSizeInVecs, this->d});
    DeviceTensor<float, 2, true> bufGpuB(
            resources_.get(),
            makeTempAlloc(AllocType::Other, defaultStream),
            {pageSizeInVecs, this->d});
    DeviceTensor<float, 2, true>* bufGpus[2] = {&bufGpuA, &bufGpuB};

    // Copy completion events for the pinned buffers
    std::unique_ptr<CudaEvent> eventPinnedCopyDone[2];

    // Execute completion events for the GPU buffers
    std::unique_ptr<CudaEvent> eventGpuExecuteDone[2];

    // All offsets are in terms of number of vectors

    // Current start offset for buffer 1
    idx_t cur1 = 0;
    idx_t cur1BufIndex = 0;

    // Current start offset for buffer 2
    idx_t cur2 = -1;
    idx_t cur2BufIndex = 0;

    // Current start offset for buffer 3
    idx_t cur3 = -1;
    idx_t cur3BufIndex = 0;

    while (cur3 < n) {
        // Start async pinned -> GPU copy first (buf 2)
        if (cur2 != -1 && cur2 < n) {
            // Copy pinned to GPU
            auto numToCopy = std::min(pageSizeInVecs, n - cur2);

            // Make sure any previous execution has completed before continuing
            auto& eventPrev = eventGpuExecuteDone[cur2BufIndex];
            if (eventPrev.get()) {
                eventPrev->streamWaitOnEvent(copyStream);
            }

            CUDA_VERIFY(cudaMemcpyAsync(
                    bufGpus[cur2BufIndex]->data(),
                    bufPinned[cur2BufIndex],
                    numToCopy * this->d * sizeof(float),
                    cudaMemcpyHostToDevice,
                    copyStream));

            // Mark a completion event in this stream
            eventPinnedCopyDone[cur2BufIndex].reset(new CudaEvent(copyStream));

            // We pick up from here
            cur3 = cur2;
            cur2 += numToCopy;
            cur2BufIndex = (cur2BufIndex == 0) ? 1 : 0;
        }

        if (cur3 != idx_t(-1) && cur3 < n) {
            // Process on GPU
            auto numToProcess = std::min(pageSizeInVecs, n - cur3);

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
            auto numToCopy = std::min(pageSizeInVecs, n - cur1);

            // Make sure any previous copy has completed before continuing
            auto& eventPrev = eventPinnedCopyDone[cur1BufIndex];
            if (eventPrev.get()) {
                eventPrev->cpuWaitOnEvent();
            }

            memcpy(bufPinned[cur1BufIndex],
                   x + cur1 * this->d,
                   numToCopy * this->d * sizeof(float));

            // We pick up from here
            cur2 = cur1;
            cur1 += numToCopy;
            cur1BufIndex = (cur1BufIndex == 0) ? 1 : 0;
        }
    }
}

void GpuIndex::compute_residual(const float* x, float* residual, idx_t key)
        const {
    FAISS_THROW_MSG("compute_residual not implemented for this type of index");
}

void GpuIndex::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
    FAISS_THROW_MSG(
            "compute_residual_n not implemented for this type of index");
}

std::shared_ptr<GpuResources> GpuIndex::getResources() {
    return resources_;
}

GpuIndex* tryCastGpuIndex(faiss::Index* index) {
    return dynamic_cast<GpuIndex*>(index);
}

bool isGpuIndex(faiss::Index* index) {
    return tryCastGpuIndex(index) != nullptr;
}

bool isGpuIndexImplemented(faiss::Index* index) {
#define CHECK_INDEX(TYPE)                 \
    do {                                  \
        if (dynamic_cast<TYPE*>(index)) { \
            return true;                  \
        }                                 \
    } while (false)

    CHECK_INDEX(faiss::IndexFlat);
    // FIXME: do we want recursive checking of the IVF quantizer?
    CHECK_INDEX(faiss::IndexIVFFlat);
    CHECK_INDEX(faiss::IndexIVFPQ);
    CHECK_INDEX(faiss::IndexIVFScalarQuantizer);

    return false;
}

} // namespace gpu

// This is the one defined in utils.cpp
// Crossing fingers that the InitGpuCompileOptions_instance will
// be instanciated after this global variable
extern std::string gpu_compile_options;

struct InitGpuCompileOptions {
    InitGpuCompileOptions() {
        gpu_compile_options = "GPU ";
#ifdef USE_NVIDIA_RAFT
        gpu_compile_options += "NVIDIA_RAFT ";
#endif

#ifdef USE_AMD_ROCM
        gpu_compile_options += "AMD_ROCM ";
#endif
    }
};

InitGpuCompileOptions InitGpuCompileOptions_instance;

} // namespace faiss
