
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "GpuIndexFlat.h"
#include "../IndexFlat.h"
#include "GpuResources.h"
#include "impl/FlatIndex.cuh"
#include "utils/CopyUtils.cuh"
#include "utils/DeviceUtils.h"
#include "utils/StaticUtils.h"

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <limits>

namespace faiss { namespace gpu {

/// Default CPU search size for which we use paged copies
constexpr size_t kMinPageSize = (size_t) 256 * 1024 * 1024;

/// Size above which we page copies from the CPU to GPU (non-paged
/// memory usage)
constexpr size_t kNonPinnedPageSize = (size_t) 256 * 1024 * 1024;

GpuIndexFlat::GpuIndexFlat(GpuResources* resources,
                           const faiss::IndexFlat* index,
                           GpuIndexFlatConfig config) :
    GpuIndex(resources, config.device, index->d, index->metric_type),
    minPagedSize_(kMinPageSize),
    config_(config),
    data_(nullptr) {
  // Flat index doesn't need training
  this->is_trained = true;
  copyFrom(index);
}

GpuIndexFlat::GpuIndexFlat(GpuResources* resources,
                           int dims,
                           faiss::MetricType metric,
                           GpuIndexFlatConfig config) :
    GpuIndex(resources, config.device, dims, metric),
    minPagedSize_(kMinPageSize),
    config_(config),
    data_(nullptr) {
  // Flat index doesn't need training
  this->is_trained = true;
  DeviceScope scope(device_);

  data_ = new FlatIndex(resources,
                        dims,
                        metric == faiss::METRIC_L2,
                        config_.useFloat16,
                        config_.storeTransposed);
}

GpuIndexFlat::~GpuIndexFlat() {
  delete data_;
}

void
GpuIndexFlat::setMinPagingSize(size_t size) {
  minPagedSize_ = size;
}

size_t
GpuIndexFlat::getMinPagingSize() const {
  return minPagedSize_;
}

bool
GpuIndexFlat::getUseFloat16() const {
  return config_.useFloat16;
}

void
GpuIndexFlat::copyFrom(const faiss::IndexFlat* index) {
  DeviceScope scope(device_);

  this->d = index->d;
  this->metric_type = index->metric_type;

  // GPU code has 32 bit indices
  FAISS_ASSERT(index->ntotal <=
               (faiss::Index::idx_t) std::numeric_limits<int>::max());
  this->ntotal = index->ntotal;

  delete data_;
  data_ = new FlatIndex(resources_,
                        this->d,
                        index->metric_type == faiss::METRIC_L2,
                        config_.useFloat16,
                        config_.storeTransposed);

  // The index could be empty
  if (index->ntotal > 0) {
    data_->add(index->xb.data(),
               index->ntotal,
               resources_->getDefaultStream(device_));
  }
}

void
GpuIndexFlat::copyTo(faiss::IndexFlat* index) const {
  DeviceScope scope(device_);

  index->d = this->d;
  index->ntotal = this->ntotal;
  index->metric_type = this->metric_type;

  FAISS_ASSERT(data_->getSize() == this->ntotal);
  index->xb.resize(this->ntotal * this->d);

  auto stream = resources_->getDefaultStream(device_);

  if (this->ntotal > 0) {
    if (config_.useFloat16) {
      auto vecFloat32 = data_->getVectorsFloat32Copy(stream);
      fromDevice(vecFloat32, index->xb.data(), stream);
    } else {
      fromDevice(data_->getVectorsFloat32Ref(), index->xb.data(), stream);
    }
  }
}

size_t
GpuIndexFlat::getNumVecs() const {
  return this->ntotal;
}

void
GpuIndexFlat::reset() {
  DeviceScope scope(device_);

  // Free the underlying memory
  data_->reset();
  this->ntotal = 0;
}

void
GpuIndexFlat::train(Index::idx_t n, const float* x) {
  // nothing to do
}

void
GpuIndexFlat::addImpl_(Index::idx_t n,
                       const float* x,
                       const Index::idx_t* ids) {
  // Device is already set in GpuIndex::addInternal_

  // We do not support add_with_ids
  FAISS_ASSERT(!ids);
  FAISS_ASSERT(n > 0);

  // Due to GPU indexing in int32, we can't store more than this
  // number of vectors on a GPU
  FAISS_ASSERT(this->ntotal + n <=
               (faiss::Index::idx_t) std::numeric_limits<int>::max());

  data_->add(x, n, resources_->getDefaultStream(device_));
  this->ntotal += n;
}

struct IntToLong {
  __device__ long operator()(int v) const { return (long) v; }
};

void
GpuIndexFlat::search(faiss::Index::idx_t n,
                     const float* x,
                     faiss::Index::idx_t k,
                     float* distances,
                     faiss::Index::idx_t* labels) const {
  if (n == 0) {
    return;
  }

  // For now, only support <= max int results
  // TODO: handle tiling over arbitrary n to keep within 32 bit bounds
  FAISS_ASSERT(n <= (faiss::Index::idx_t) std::numeric_limits<int>::max());
  FAISS_ASSERT(k <= 1024); // select limitation

  DeviceScope scope(device_);
  auto stream = resources_->getDefaultStream(device_);

  // The input vectors may be too large for the GPU, but we still
  // assume that the output distances and labels are not.
  // Go ahead and make space for output distances and labels on the
  // GPU.
  // If we reach a point where all inputs are too big, we can add
  // another level of tiling.
  auto outDistances = toDevice<float, 2>(resources_,
                                         device_,
                                         distances,
                                         stream,
                                         {(int) n, (int) k});

  // FlatIndex only supports an interface returning int indices
  DeviceTensor<int, 2, true> outIntIndices(
    resources_->getMemoryManagerCurrentDevice(),
    {(int) n, (int) k}, stream);

  bool usePaged = false;

  if (getDeviceForAddress(x) == -1) {
    // It is possible that the user is querying for a vector set size
    // `x` that won't fit on the GPU.
    // In this case, we will have to handle paging of the data from CPU
    // -> GPU.
    // Currently, we don't handle the case where the output data won't
    // fit on the GPU (e.g., n * k is too large for the GPU memory).
    size_t dataSize = (size_t) n * this->d * sizeof(float);

    if (dataSize >= minPagedSize_) {
      searchFromCpuPaged_(n, x, k,
                          outDistances.data(),
                          outIntIndices.data());
      usePaged = true;
    }
  }

  if (!usePaged) {
    searchNonPaged_(n, x, k,
                    outDistances.data(),
                    outIntIndices.data());
  }

  // Convert and copy int indices out
  auto outIndices = toDevice<faiss::Index::idx_t, 2>(resources_,
                                                     device_,
                                                     labels,
                                                     stream,
                                                     {(int) n, (int) k});

  // Convert int to long
  thrust::transform(thrust::cuda::par.on(stream),
                    outIntIndices.data(),
                    outIntIndices.end(),
                    outIndices.data(),
                    IntToLong());

  // Copy back if necessary
  fromDevice<float, 2>(outDistances, distances, stream);
  fromDevice<faiss::Index::idx_t, 2>(outIndices, labels, stream);
}

void
GpuIndexFlat::searchImpl_(faiss::Index::idx_t n,
                          const float* x,
                          faiss::Index::idx_t k,
                          float* distances,
                          faiss::Index::idx_t* labels) const {
  FAISS_ASSERT(!"Should not be called");
}

void
GpuIndexFlat::searchNonPaged_(int n,
                              const float* x,
                              int k,
                              float* outDistancesData,
                              int* outIndicesData) const {
  Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
  Tensor<int, 2, true> outIndices(outIndicesData, {n, k});

  auto stream = resources_->getDefaultStream(device_);

  // Make sure arguments are on the device we desire; use temporary
  // memory allocations to move it if necessary
  auto vecs = toDevice<float, 2>(resources_,
                                 device_,
                                 const_cast<float*>(x),
                                 stream,
                                 {n, (int) this->d});

  data_->query(vecs, k, outDistances, outIndices, true);
}

void
GpuIndexFlat::searchFromCpuPaged_(int n,
                                  const float* x,
                                  int k,
                                  float* outDistancesData,
                                  int* outIndicesData) const {
  Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
  Tensor<int, 2, true> outIndices(outIndicesData, {n, k});

  // Is pinned memory available?
  auto pinnedAlloc = resources_->getPinnedMemory();
  int pageSizeInVecs =
    (int) ((pinnedAlloc.second / 2) / (sizeof(float) * this->d));

  if (!pinnedAlloc.first || pageSizeInVecs < 1) {
    // Just page without overlapping copy with compute
    int batchSize = utils::nextHighestPowerOf2(
      (int) ((size_t) kNonPinnedPageSize /
             (sizeof(float) * this->d)));

    for (int cur = 0; cur < n; cur += batchSize) {
      int num = std::min(batchSize, n - cur);

      auto outDistancesSlice = outDistances.narrowOutermost(cur, num);
      auto outIndicesSlice = outIndices.narrowOutermost(cur, num);

      searchNonPaged_(num,
                      x + (size_t) cur * this->d,
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
  auto defaultStream = resources_->getDefaultStream(device_);
  auto copyStream = resources_->getAsyncCopyStream(device_);

  FAISS_ASSERT((size_t) pageSizeInVecs * this->d <=
               (size_t) std::numeric_limits<int>::max());

  float* bufPinnedA = (float*) pinnedAlloc.first;
  float* bufPinnedB = bufPinnedA + (size_t) pageSizeInVecs * this->d;
  float* bufPinned[2] = {bufPinnedA, bufPinnedB};

  // Reserve space on the GPU for the destination of the pinned buffer
  // copy
  DeviceTensor<float, 2, true> bufGpuA(
    resources_->getMemoryManagerCurrentDevice(),
    {(int) pageSizeInVecs, (int) this->d},
    defaultStream);
  DeviceTensor<float, 2, true> bufGpuB(
    resources_->getMemoryManagerCurrentDevice(),
    {(int) pageSizeInVecs, (int) this->d},
    defaultStream);
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

      CUDA_VERIFY(cudaMemcpyAsync(bufGpus[cur2BufIndex]->data(),
                                  bufPinned[cur2BufIndex],
                                  (size_t) numToCopy * this->d * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  copyStream));

      // Mark a completion event in this stream
      eventPinnedCopyDone[cur2BufIndex] =
        std::move(std::unique_ptr<CudaEvent>(new CudaEvent(copyStream)));

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
      DeviceTensor<float, 2, true> input(bufGpus[cur3BufIndex]->data(),
                                         {numToProcess, this->d});
      auto outDistancesSlice = outDistances.narrowOutermost(cur3, numToProcess);
      auto outIndicesSlice = outIndices.narrowOutermost(cur3, numToProcess);

      data_->query(input, k,
                   outDistancesSlice,
                   outIndicesSlice, true);

      // Create completion event
      eventGpuExecuteDone[cur3BufIndex] =
        std::move(std::unique_ptr<CudaEvent>(new CudaEvent(defaultStream)));

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

      memcpy(bufPinned[cur1BufIndex],
             x + (size_t) cur1 * this->d,
             (size_t) numToCopy * this->d * sizeof(float));

      // We pick up from here
      cur2 = cur1;
      cur1 += numToCopy;
      cur1BufIndex = (cur1BufIndex == 0) ? 1 : 0;
    }
  }
}

void
GpuIndexFlat::reconstruct(faiss::Index::idx_t key,
                          float* out) const {
  DeviceScope scope(device_);

  FAISS_ASSERT(key < this->ntotal);
  auto stream = resources_->getDefaultStream(device_);

  if (config_.useFloat16) {
    auto vec = data_->getVectorsFloat32Copy(key, 1, stream);
    fromDevice(vec.data(), out, this->d, stream);
  } else {
    auto vec = data_->getVectorsFloat32Ref()[key];
    fromDevice(vec.data(), out, this->d, stream);
  }
}

void
GpuIndexFlat::reconstruct_n(faiss::Index::idx_t i0,
                            faiss::Index::idx_t num,
                            float* out) const {
  DeviceScope scope(device_);

  FAISS_ASSERT(i0 < this->ntotal);
  FAISS_ASSERT(i0 + num - 1 < this->ntotal);
  auto stream = resources_->getDefaultStream(device_);

  if (config_.useFloat16) {
    auto vec = data_->getVectorsFloat32Copy(i0, num, stream);
    fromDevice(vec.data(), out, num * this->d, stream);
  } else {
    auto vec = data_->getVectorsFloat32Ref()[i0];
    fromDevice(vec.data(), out, this->d * num, stream);
  }
}

void
GpuIndexFlat::set_typename() {
  if (this->metric_type == faiss::METRIC_L2) {
    this->index_typename = "GpuL2";
  } else {
    this->index_typename = "GpuIP";
  }
}

//
// GpuIndexFlatL2
//

GpuIndexFlatL2::GpuIndexFlatL2(GpuResources* resources,
                               faiss::IndexFlatL2* index,
                               GpuIndexFlatConfig config) :
    GpuIndexFlat(resources, index, config) {
}

GpuIndexFlatL2::GpuIndexFlatL2(GpuResources* resources,
                               int dims,
                               GpuIndexFlatConfig config) :
    GpuIndexFlat(resources, dims, faiss::METRIC_L2, config) {
}

void
GpuIndexFlatL2::copyFrom(faiss::IndexFlatL2* index) {
  GpuIndexFlat::copyFrom(index);
}

void
GpuIndexFlatL2::copyTo(faiss::IndexFlatL2* index) {
  GpuIndexFlat::copyTo(index);
}

//
// GpuIndexFlatIP
//

GpuIndexFlatIP::GpuIndexFlatIP(GpuResources* resources,
                               faiss::IndexFlatIP* index,
                               GpuIndexFlatConfig config) :
    GpuIndexFlat(resources, index, config) {
}

GpuIndexFlatIP::GpuIndexFlatIP(GpuResources* resources,
                               int dims,
                               GpuIndexFlatConfig config) :
    GpuIndexFlat(resources, dims, faiss::METRIC_INNER_PRODUCT, config) {
}

void
GpuIndexFlatIP::copyFrom(faiss::IndexFlatIP* index) {
  GpuIndexFlat::copyFrom(index);
}

void
GpuIndexFlatIP::copyTo(faiss::IndexFlatIP* index) {
  GpuIndexFlat::copyTo(index);
}

} } // namespace
