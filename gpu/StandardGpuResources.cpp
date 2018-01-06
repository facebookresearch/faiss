/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "StandardGpuResources.h"
#include "../FaissAssert.h"

namespace faiss { namespace gpu {

namespace {

constexpr int kNumStreams = 2;

/// Use 18% of GPU memory for temporary space by default
constexpr float kDefaultTempMemFraction = 0.18f;

/// Default pinned memory allocation size
constexpr size_t kDefaultPinnedMemoryAllocation = (size_t) 256 * 1024 * 1024;

}

StandardGpuResources::StandardGpuResources() :
    pinnedMemAlloc_(nullptr),
    pinnedMemAllocSize_(0),
    tempMemFraction_(kDefaultTempMemFraction),
    tempMemSize_(0),
    useFraction_(true),
    pinnedMemSize_(kDefaultPinnedMemoryAllocation) {
}

StandardGpuResources::~StandardGpuResources() {
  for (auto& entry : defaultStreams_) {
    DeviceScope scope(entry.first);

    auto it = userDefaultStreams_.find(entry.first);
    if (it == userDefaultStreams_.end()) {
      // The user did not specify this stream, thus we are the ones
      // who have created it
      CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }
  }

  for (auto& entry : alternateStreams_) {
    DeviceScope scope(entry.first);

    for (auto stream : entry.second) {
      CUDA_VERIFY(cudaStreamDestroy(stream));
    }
  }

  for (auto& entry : asyncCopyStreams_) {
    DeviceScope scope(entry.first);

    CUDA_VERIFY(cudaStreamDestroy(entry.second));
  }

  for (auto& entry : blasHandles_) {
    DeviceScope scope(entry.first);

    auto blasStatus = cublasDestroy(entry.second);
    FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
  }

  if (pinnedMemAlloc_) {
    CUDA_VERIFY(cudaFreeHost(pinnedMemAlloc_));
  }
}

void
StandardGpuResources::noTempMemory() {
  setTempMemory(0);
}

void
StandardGpuResources::setTempMemory(size_t size) {
  useFraction_ = false;
  tempMemSize_ = size;
}

void
StandardGpuResources::setTempMemoryFraction(float fraction) {
  FAISS_ASSERT(fraction >= 0.0f && fraction <= 0.5f);
  useFraction_ = true;
  tempMemFraction_ = fraction;
}

void
StandardGpuResources::setPinnedMemory(size_t size) {
  // Should not call this after devices have been initialized
  FAISS_ASSERT(defaultStreams_.size() == 0);
  FAISS_ASSERT(!pinnedMemAlloc_);

  pinnedMemSize_ = size;
}

void
StandardGpuResources::setDefaultStream(int device, cudaStream_t stream) {
  auto it = defaultStreams_.find(device);
  if (it != defaultStreams_.end()) {
    // Replace this stream with the user stream
    CUDA_VERIFY(cudaStreamDestroy(it->second));
    it->second = stream;
  }

  userDefaultStreams_[device] = stream;
}

void
StandardGpuResources::setDefaultNullStreamAllDevices() {
  for (int dev = 0; dev < getNumDevices(); ++dev) {
    setDefaultStream(dev, nullptr);
  }
}

void
StandardGpuResources::initializeForDevice(int device) {
  // Use default streams as a marker for whether or not a certain
  // device has been initialized
  if (defaultStreams_.count(device) != 0) {
    return;
  }

  // If this is the first device that we're initializing, create our
  // pinned memory allocation
  if (defaultStreams_.empty() && pinnedMemSize_ > 0) {
    CUDA_VERIFY(cudaHostAlloc(&pinnedMemAlloc_,
                              pinnedMemSize_,
                              cudaHostAllocDefault));
    pinnedMemAllocSize_ = pinnedMemSize_;
  }

  FAISS_ASSERT(device < getNumDevices());
  DeviceScope scope(device);

  // Make sure that device properties for all devices are cached
  auto& prop = getDeviceProperties(device);

  // Also check to make sure we meet our minimum compute capability (3.0)
  FAISS_ASSERT_FMT(prop.major >= 3,
                   "Device id %d with CC %d.%d not supported, "
                   "need 3.0+ compute capability",
                   device, prop.major, prop.minor);

  // Create streams
  cudaStream_t defaultStream = 0;
  auto it = userDefaultStreams_.find(device);
  if (it != userDefaultStreams_.end()) {
    // We already have a stream provided by the user
    defaultStream = it->second;
  } else {
    CUDA_VERIFY(cudaStreamCreateWithFlags(&defaultStream,
                                          cudaStreamNonBlocking));
  }

  defaultStreams_[device] = defaultStream;

  cudaStream_t asyncCopyStream = 0;
  CUDA_VERIFY(cudaStreamCreateWithFlags(&asyncCopyStream,
                                        cudaStreamNonBlocking));

  asyncCopyStreams_[device] = asyncCopyStream;

  std::vector<cudaStream_t> deviceStreams;
  for (int j = 0; j < kNumStreams; ++j) {
    cudaStream_t stream = 0;
    CUDA_VERIFY(cudaStreamCreateWithFlags(&stream,
                                          cudaStreamNonBlocking));

    deviceStreams.push_back(stream);
  }

  alternateStreams_[device] = std::move(deviceStreams);

  // Create cuBLAS handle
  cublasHandle_t blasHandle = 0;
  auto blasStatus = cublasCreate(&blasHandle);
  FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
  blasHandles_[device] = blasHandle;

  size_t toAlloc = 0;
  if (useFraction_) {
    size_t devFree = 0;
    size_t devTotal = 0;

    CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));

    toAlloc = (size_t) (tempMemFraction_ * devTotal);
  } else {
    toAlloc = tempMemSize_;
  }

  FAISS_ASSERT(memory_.count(device) == 0);
  memory_.emplace(device,
                  std::unique_ptr<StackDeviceMemory>(
                    new StackDeviceMemory(device, toAlloc)));
}

cublasHandle_t
StandardGpuResources::getBlasHandle(int device) {
  initializeForDevice(device);
  return blasHandles_[device];
}

cudaStream_t
StandardGpuResources::getDefaultStream(int device) {
  initializeForDevice(device);
  return defaultStreams_[device];
}

std::vector<cudaStream_t>
StandardGpuResources::getAlternateStreams(int device) {
  initializeForDevice(device);
  return alternateStreams_[device];
}

DeviceMemory& StandardGpuResources::getMemoryManager(int device) {
  initializeForDevice(device);
  return *memory_[device];
}

std::pair<void*, size_t>
StandardGpuResources::getPinnedMemory() {
  return std::make_pair(pinnedMemAlloc_, pinnedMemAllocSize_);
}

cudaStream_t
StandardGpuResources::getAsyncCopyStream(int device) {
  initializeForDevice(device);
  return asyncCopyStreams_[device];
}

} } // namespace
