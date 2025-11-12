/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Clustering.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuMultiIndex2.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/utils.h>
#include <cstdio>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/MultiIndex2.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace faiss {
namespace gpu {

/// Size above which we page copies from the CPU to GPU (non-paged
/// memory usage)
constexpr size_t kNonPinnedPageSize = (size_t)256 * 1024 * 1024;

const int GpuMultiIndex2::NUM_CODEBOOKS = 2;

GpuMultiIndex2::GpuMultiIndex2(
        GpuResourcesProvider* provider,
        const faiss::MultiIndexQuantizer* index,
        GpuMultiIndex2Config config)
        : GpuIndex(
                  provider->getResources(),
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  config),
          numVecsPerCodebook_(0),
          subDim_(index->d / GpuMultiIndex2::NUM_CODEBOOKS),
          config_(config) {
    init_();
    copyFrom(index);
}

GpuMultiIndex2::GpuMultiIndex2(
        std::shared_ptr<GpuResources> resources,
        const faiss::MultiIndexQuantizer* index,
        GpuMultiIndex2Config config)
        : GpuIndex(
                  resources,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  config),
          numVecsPerCodebook_(0),
          subDim_(index->d / GpuMultiIndex2::NUM_CODEBOOKS),
          config_(config) {
    init_();
    copyFrom(index);
}

GpuMultiIndex2::GpuMultiIndex2(
        GpuResourcesProvider* provider,
        int dims,
        int numVecsPerCodebook_,
        GpuMultiIndex2Config config)
        : GpuIndex(
                  provider->getResources(),
                  dims,
                  faiss::MetricType::METRIC_L2,
                  0,
                  config),
          numVecsPerCodebook_(numVecsPerCodebook_),
          subDim_(dims / GpuMultiIndex2::NUM_CODEBOOKS),
          config_(config) {
    init_();
    this->is_trained = false;
    DeviceScope scope(config_.device);
    data_.reset(new MultiIndex2(resources_.get(), dims, config_.memorySpace));
}

GpuMultiIndex2::GpuMultiIndex2(
        std::shared_ptr<GpuResources> resources,
        int dims,
        int numVecsPerCodebook_,
        GpuMultiIndex2Config config)
        : GpuIndex(resources, dims, faiss::MetricType::METRIC_L2, 0, config),
          numVecsPerCodebook_(numVecsPerCodebook_),
          subDim_(dims / GpuMultiIndex2::NUM_CODEBOOKS),
          config_(config) {
    init_();
    this->is_trained = false;
    DeviceScope scope(config_.device);
    data_.reset(new MultiIndex2(resources_.get(), dims, config_.memorySpace));
}

void GpuMultiIndex2::init_() {
    FAISS_ASSERT(this->d % GpuMultiIndex2::NUM_CODEBOOKS == 0);

    // here we set a low # iterations because this is typically used
    // for large clusterings
    cp.niter = 10;
    cp.verbose = verbose;
}

GpuMultiIndex2::~GpuMultiIndex2() {}

size_t GpuMultiIndex2::calcMemorySpaceSize(
        int numVecsTotal,
        int dimPerCodebook,
        bool useFloat16) {
    return MultiIndex2::calcMemorySpaceSize(
            numVecsTotal, dimPerCodebook, useFloat16);
}

void GpuMultiIndex2::copyFrom(const faiss::MultiIndexQuantizer* index) {
    DeviceScope scope(config_.device);

    FAISS_ASSERT(index->pq.M == GpuMultiIndex2::NUM_CODEBOOKS);
    FAISS_ASSERT(index->metric_type == faiss::MetricType::METRIC_L2);
    FAISS_ASSERT(index->metric_arg == 0);

    GpuIndex::copyFrom(index);

    // GPU code has 32 bit indices
    FAISS_THROW_IF_NOT_FMT(
            index->ntotal <= (idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %zu indices; "
            "attempting to copy CPU index with %zu parameters",
            (size_t)std::numeric_limits<int>::max(),
            (size_t)index->ntotal);

    data_.reset(
            new MultiIndex2(resources_.get(), this->d, config_.memorySpace));

    FAISS_ASSERT(subDim_ == index->pq.dsub);

    subDim_ = index->pq.dsub;
    numVecsPerCodebook_ = index->pq.centroids.size() / this->d;

    FAISS_ASSERT(this->ntotal == numVecsPerCodebook_ * numVecsPerCodebook_);

    // The other index might not be trained
    if (!index->is_trained) {
        // copied in GpuIndex::copyFrom
        FAISS_ASSERT(!this->is_trained);
        return;
    }

    if (this->is_trained) {
        data_->reset();
    }

    FAISS_ASSERT(index->pq.centroids.size() == numVecsPerCodebook_ * this->d);

    auto stream = resources_->getDefaultStream(config_.device);

    data_->add(
            index->pq.centroids.data(),
            GpuMultiIndex2::NUM_CODEBOOKS * numVecsPerCodebook_,
            stream);

    FAISS_ASSERT(this->is_trained);
}

void GpuMultiIndex2::copyTo(faiss::MultiIndexQuantizer* index) const {
    DeviceScope scope(config_.device);

    GpuIndex::copyTo(index);

    FAISS_ASSERT(data_);
    FAISS_ASSERT(data_->getSize() == this->ntotal);

    index->pq = faiss::ProductQuantizer();
    index->pq.d = this->d;
    index->pq.M = GpuMultiIndex2::NUM_CODEBOOKS;
    index->pq.nbits = utils::log2(data_->getCodebookSize());
    index->pq.dsub = subDim_;

    size_t nbitsCode = utils::isPowerOf2(data_->getCodebookSize())
            ? index->pq.nbits
            : index->pq.nbits + 1;

    index->pq.code_size = (nbitsCode * index->pq.M + 7) / 8;
    index->pq.ksub = data_->getCodebookSize();
    index->pq.centroids.resize(index->pq.d * index->pq.ksub);
    index->pq.verbose = false;
    index->pq.train_type = faiss::ProductQuantizer::train_type_t::Train_default;

    auto stream = resources_->getDefaultStream(config_.device);

    if (this->ntotal > 0) {
        fromDevice(
                data_->getVectorsFloat32Ref(),
                index->pq.centroids.data(),
                stream);
    }
}

int GpuMultiIndex2::toMultiIndex(std::pair<ushort, ushort> indexPair) const {
    ushort2* indexPairUshort2 = (ushort2*)&indexPair;
    return this->data_->toMultiIndex(*indexPairUshort2);
}

int GpuMultiIndex2::getCodebookSize() {
    return this->data_->getCodebookSize();
}

int GpuMultiIndex2::getNumCodebooks() {
    return this->data_->getNumCodebooks();
}

int GpuMultiIndex2::getNumVecs() {
    return this->data_->getSize();
}

int GpuMultiIndex2::getSubDim() {
    return this->subDim_;
}

std::vector<float> GpuMultiIndex2::getCentroids() {
    auto stream = resources_->getDefaultStream(config_.device);
    auto centroidsDevice = data_->getVectorsFloat32Ref();
    std::vector<float> centroids(centroidsDevice.numElements());
    fromDevice(centroidsDevice, centroids.data(), stream);
    return centroids;
}

void GpuMultiIndex2::load(int codebookSize, const float* centroids) {
    FAISS_ASSERT(data_);
    FAISS_ASSERT(codebookSize > 0);

    DeviceScope scope(config_.device);
    // If it is trained, just resets
    if (this->is_trained) {
        data_->reset();
    }

    data_->add(
            centroids,
            GpuMultiIndex2::NUM_CODEBOOKS * codebookSize,
            resources_->getDefaultStream(config_.device));

    this->ntotal = 1;
    for (int i = 0; i < GpuMultiIndex2::NUM_CODEBOOKS; i++) {
        this->ntotal *= numVecsPerCodebook_;
    }
    this->is_trained = true;
}

void GpuMultiIndex2::reset() {
    FAISS_THROW_MSG(
            "This index has virtual elements, "
            "it does not support reset");
}

void GpuMultiIndex2::train(idx_t n, const float* x) {
    FAISS_ASSERT(data_);
    FAISS_ASSERT(n > 0);

    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    // If it is trained, just resets and re-trains it
    if (this->is_trained) {
        data_->reset();
    }

    std::unique_ptr<float[]> subQueries(
            new float[GpuMultiIndex2::NUM_CODEBOOKS * n * subDim_]);
    fvec_split(
            subQueries.get(),
            GpuMultiIndex2::NUM_CODEBOOKS,
            x,
            (size_t)n,
            subDim_);

    int numSubCentroids = GpuMultiIndex2::NUM_CODEBOOKS * numVecsPerCodebook_;

    DeviceTensor<float, 1, true> subCentroids(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {numSubCentroids * subDim_});

    GpuIndexFlatConfig flatConfig;
    flatConfig.device = config_.device;

    std::vector<std::unique_ptr<GpuIndexFlatL2>> codebookList(
            GpuMultiIndex2::NUM_CODEBOOKS);

    for (int i = 0; i < GpuMultiIndex2::NUM_CODEBOOKS; i++) {
        codebookList[i].reset(
                new GpuIndexFlatL2(resources_, subDim_, flatConfig));

        Clustering clus(subDim_, numVecsPerCodebook_, this->cp);
        clus.verbose = verbose;
        clus.train(n, subQueries.get() + (i * n * subDim_), *codebookList[i]);
        codebookList[i]->is_trained = true;

        auto codebookVecs =
                codebookList[i]->getGpuData()->getVectorsFloat32Ref();

        FAISS_ASSERT(
                codebookVecs.numElements() == numVecsPerCodebook_ * subDim_);

        fromDevice<float>(
                codebookVecs.data(),
                subCentroids.data() + i * numVecsPerCodebook_ * subDim_,
                codebookVecs.numElements(),
                stream);
    }

    data_->add(subCentroids.data(), numSubCentroids, stream);

    CudaEvent addEnd(stream);

    this->ntotal = 1;
    for (int i = 0; i < GpuMultiIndex2::NUM_CODEBOOKS; i++) {
        this->ntotal *= numVecsPerCodebook_;
    }
    this->is_trained = true;

    // synchronizing to ensure that subQueries has not been deleted before copy
    // ends
    addEnd.cpuWaitOnEvent();
}

void GpuMultiIndex2::add(faiss::idx_t n, const float* x) {
    FAISS_THROW_MSG(
            "This index has virtual elements, "
            "it does not support add");
}

void GpuMultiIndex2::add_with_ids(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_MSG(
            "This index has virtual elements, "
            "it does not support add_with_ids");
}

void GpuMultiIndex2::assign(idx_t n, const float* x, idx_t* labels, idx_t k)
        const {
    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    // We need to create a throw-away buffer for distances, which we don't use
    // but which we do need for the search call
    DeviceTensor<float, 2, true> distances(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {(int)n, (int)k});

    // Forward to search
    search(n, x, k, distances.data(), labels);
}

void GpuMultiIndex2::assign_pair(
        idx_t n,
        const float* x,
        std::pair<ushort, ushort>* labels,
        idx_t k) const {
    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    // We need to create a throw-away buffer for distances, which we don't use
    // but which we do need for the search call
    DeviceTensor<float, 2, true> distances(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {(int)n, (int)k});
    search_pair(n, x, k, distances.data(), labels);
}

void GpuMultiIndex2::search(
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

    FAISS_THROW_IF_NOT_FMT(
            k <= (idx_t)this->ntotal,
            "GPU index only supports k <= %d (requested %d)",
            (int)this->ntotal,
            (int)k); // select limitation

    FAISS_THROW_IF_NOT_FMT(
            k <= (idx_t)getMaxKSelection() * getMaxKSelection(),
            "GPU index only supports k <= %d (requested %d)",
            getMaxKSelection() * getMaxKSelection(),
            (int)k); // select limitation

    if (k > (idx_t)getMaxKSelection()) {
        std::cout
                << "WARNING: k on multi-index must be <= " << getMaxKSelection()
                << " to ensure the correctness of the multi-sequence algorithm"
                << std::endl;
    }

    if (n == 0 || k == 0) {
        // nothing to search
        return;
    }

    DeviceScope scope(config_.device);
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
            resources_.get(),
            config_.device,
            distances,
            stream,
            {(int)n, (int)k});

    auto outLabels = toDeviceTemporary<faiss::idx_t, 2>(
            resources_.get(), config_.device, labels, stream, {(int)n, (int)k});

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
            searchFromCpuPaged_(n, x, k, outDistances.data(), outLabels.data());
            usePaged = true;
        }
    }

    if (!usePaged) {
        searchNonPaged_(n, x, k, outDistances.data(), outLabels.data());
    }

    // Copy back if necessary
    fromDevice<float, 2>(outDistances, distances, stream);
    fromDevice<faiss::idx_t, 2>(outLabels, labels, stream);
}

void GpuMultiIndex2::searchNonPaged_(
        int n,
        const float* x,
        int k,
        float* outDistancesData,
        idx_t* outIndicesData) const {
    auto stream = resources_->getDefaultStream(config_.device);

    // FIXME: Change location to searchFromCpuPaged_
    std::unique_ptr<float[]> subQueries(new float[n * this->d]);
    fvec_split(
            subQueries.get(),
            GpuMultiIndex2::NUM_CODEBOOKS,
            x,
            (size_t)n,
            subDim_);

    // Make sure arguments are on the device we desire; use temporary
    // memory allocations to move it if necessary
    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(subQueries.get()),
            stream,
            {GpuMultiIndex2::NUM_CODEBOOKS * n, (int)this->subDim_});

    CudaEvent copyEnd(stream);

    searchImpl_(n, vecs.data(), k, outDistancesData, outIndicesData);

    // synchronizing to ensure that subQueries has not been deleted before copy
    // ends
    copyEnd.cpuWaitOnEvent();
}

void GpuMultiIndex2::searchFromCpuPaged_(
        int n,
        const float* x,
        int k,
        float* outDistancesData,
        idx_t* outIndicesData) const {
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
                    outIndicesSlice.data());
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
            {(int)pageSizeInVecs, (int)this->d});
    DeviceTensor<float, 2, true> bufGpuB(
            resources_.get(),
            makeTempAlloc(AllocType::Other, defaultStream),
            {(int)pageSizeInVecs, (int)this->d});
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
                    outIndicesSlice.data());

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
                    GpuMultiIndex2::NUM_CODEBOOKS,
                    x + (size_t)cur1 * this->d,
                    (size_t)numToCopy,
                    subDim_);

            // We pick up from here
            cur2 = cur1;
            cur1 += numToCopy;
            cur1BufIndex = (cur1BufIndex == 0) ? 1 : 0;
        }
    }
}

void GpuMultiIndex2::search_pair(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        std::pair<ushort, ushort>* labels) const {
    static_assert(sizeof(std::pair<ushort, ushort>) == sizeof(ushort2));
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    FAISS_THROW_IF_NOT_FMT(
            k <= (idx_t)this->ntotal,
            "GPU index only supports k <= %d (requested %d)",
            (int)this->ntotal,
            (int)k); // select limitation

    FAISS_THROW_IF_NOT_FMT(
            k <= (idx_t)getMaxKSelection() * getMaxKSelection(),
            "GPU index only supports k <= %d (requested %d)",
            getMaxKSelection() * getMaxKSelection(),
            (int)k); // select limitation

    if (k > (idx_t)getMaxKSelection()) {
        std::cout
                << "WARNING: k on multi-index must be <= " << getMaxKSelection()
                << " to ensure the correctness of the multi-sequence algorithm"
                << std::endl;
    }

    if (n == 0 || k == 0) {
        // nothing to search
        return;
    }

    DeviceScope scope(config_.device);
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
            resources_.get(),
            config_.device,
            distances,
            stream,
            {(int)n, (int)k});

    auto outLabels = toDeviceTemporary<ushort2, 2>(
            resources_.get(),
            config_.device,
            (ushort2*)labels,
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

        if (dataSize >= minPagedSize_) {
            searchFromCpuPaged_(
                    n,
                    x,
                    k,
                    outDistances.data(),
                    (std::pair<ushort, ushort>*)outLabels.data());
            usePaged = true;
        }
    }

    if (!usePaged) {
        searchNonPaged_(
                n,
                x,
                k,
                outDistances.data(),
                (std::pair<ushort, ushort>*)outLabels.data());
    }

    // Copy back if necessary
    fromDevice<float, 2>(outDistances, distances, stream);
    fromDevice<ushort2, 2>(outLabels, (ushort2*)labels, stream);
}

void GpuMultiIndex2::searchNonPaged_(
        int n,
        const float* x,
        int k,
        float* outDistancesData,
        std::pair<ushort, ushort>* outIndicesData) const {
    auto stream = resources_->getDefaultStream(config_.device);

    // FIXME: Change location to searchFromCpuPaged_
    std::unique_ptr<float[]> subQueries(new float[n * this->d]);
    fvec_split(
            subQueries.get(),
            GpuMultiIndex2::NUM_CODEBOOKS,
            x,
            (size_t)n,
            subDim_);

    // Make sure arguments are on the device we desire; use temporary
    // memory allocations to move it if necessary
    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(subQueries.get()),
            stream,
            {GpuMultiIndex2::NUM_CODEBOOKS * n, (int)this->subDim_});

    CudaEvent copyEnd(stream);

    searchPairImpl_(n, vecs.data(), k, outDistancesData, outIndicesData);

    // synchronizing to ensure that subQueries has not been deleted before copy
    // ends
    copyEnd.cpuWaitOnEvent();
}

void GpuMultiIndex2::searchFromCpuPaged_(
        int n,
        const float* x,
        int k,
        float* outDistancesData,
        std::pair<ushort, ushort>* outIndicesData) const {
    static_assert(sizeof(std::pair<ushort, ushort>) == sizeof(ushort2));
    Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
    Tensor<ushort2, 2, true> outIndices((ushort2*)outIndicesData, {n, k});

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
                    (std::pair<ushort, ushort>*)outIndicesSlice.data());
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
            {(int)pageSizeInVecs, (int)this->d});
    DeviceTensor<float, 2, true> bufGpuB(
            resources_.get(),
            makeTempAlloc(AllocType::Other, defaultStream),
            {(int)pageSizeInVecs, (int)this->d});
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

            searchPairImpl_(
                    numToProcess,
                    bufGpus[cur3BufIndex]->data(),
                    k,
                    outDistancesSlice.data(),
                    (std::pair<ushort, ushort>*)outIndicesSlice.data());

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
                    GpuMultiIndex2::NUM_CODEBOOKS,
                    x + (size_t)cur1 * this->d,
                    (size_t)numToCopy,
                    subDim_);

            // We pick up from here
            cur2 = cur1;
            cur1 += numToCopy;
            cur1BufIndex = (cur1BufIndex == 0) ? 1 : 0;
        }
    }
}

bool GpuMultiIndex2::addImplRequiresIDs_() const {
    FAISS_THROW_MSG(
            "This index has virtual elements, "
            "it does not support add");
}

void GpuMultiIndex2::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_MSG(
            "This index has virtual elements, "
            "it does not support add");
}

void GpuMultiIndex2::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    auto stream = resources_->getDefaultStream(config_.device);

    // Input and output data are already resident on the GPU
    Tensor<float, 2, true> queries(
            const_cast<float*>(x),
            {GpuMultiIndex2::NUM_CODEBOOKS * n, this->subDim_});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(labels, {n, k});

    data_->query(queries, k, outDistances, outLabels, true);
}

void GpuMultiIndex2::searchPairImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        std::pair<ushort, ushort>* labels) const {
    static_assert(sizeof(std::pair<ushort, ushort>) == sizeof(ushort2));
    auto stream = resources_->getDefaultStream(config_.device);

    // Input and output data are already resident on the GPU
    Tensor<float, 2, true> queries(
            const_cast<float*>(x),
            {GpuMultiIndex2::NUM_CODEBOOKS * n, this->subDim_});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<ushort2, 2, true> outLabels((ushort2*)labels, {n, k});

    data_->query(queries, k, outDistances, outLabels, true);
}

void GpuMultiIndex2::compute_residual_pair(
        const float* x,
        float* residual,
        std::pair<ushort, ushort> key) const {
    compute_residual_n_pair(1, x, residual, &key);
}

void GpuMultiIndex2::compute_residual_n_pair(
        faiss::idx_t n,
        const float* xs,
        float* residuals,
        const std::pair<ushort, ushort>* keys) const {
    FAISS_THROW_IF_NOT_FMT(
            n <= (faiss::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %zu indices",
            (size_t)std::numeric_limits<int>::max());

    auto stream = resources_->getDefaultStream(config_.device);

    DeviceScope scope(config_.device);

    std::unique_ptr<float[]> subQueries(new float[n * this->d]);
    fvec_split(
            subQueries.get(),
            GpuMultiIndex2::NUM_CODEBOOKS,
            xs,
            (size_t)n,
            subDim_);

    auto vecsDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(subQueries.get()),
            stream,
            {GpuMultiIndex2::NUM_CODEBOOKS * (int)n, (int)this->subDim_});

    CudaEvent copyEnd(stream);

    auto idsDevice = toDeviceTemporary<ushort2, 1>(
            resources_.get(),
            config_.device,
            (ushort2*)(keys),
            stream,
            {(int)n});

    DeviceTensor<float, 2, true> residualDevice(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {(int)n, (int)this->d});

    FAISS_ASSERT(data_);
    data_->computeResidual(vecsDevice, idsDevice, residualDevice);

    fromDevice<float, 2>(residualDevice, residuals, stream);

    // synchronizing to ensure that subQueries has not been deleted before copy
    // ends
    copyEnd.cpuWaitOnEvent();
}

void GpuMultiIndex2::compute_nearest_residual_n(
        faiss::idx_t n,
        const float* x,
        float* residuals) const {
    std::vector<std::pair<ushort, ushort>> keys(n);
    assign_pair(n, x, keys.data());

    // FIXME jhj convert to _n version
    for (idx_t i = 0; i < n; i++) {
        compute_residual_pair(x + i * d, &residuals[i * d], keys[i]);
    }
}

} // namespace gpu
} // namespace faiss
