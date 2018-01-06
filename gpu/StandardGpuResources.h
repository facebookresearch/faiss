/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "GpuResources.h"
#include "utils/StackDeviceMemory.h"
#include "utils/DeviceUtils.h"
#include <unordered_map>
#include <vector>

namespace faiss { namespace gpu {

/// Default implementation of GpuResources that allocates a cuBLAS
/// stream and 2 streams for use, as well as temporary memory
class StandardGpuResources : public GpuResources {
 public:
  StandardGpuResources();

  ~StandardGpuResources() override;

  /// Disable allocation of temporary memory; all temporary memory
  /// requests will call cudaMalloc / cudaFree at the point of use
  void noTempMemory();

  /// Specify that we wish to use a certain fixed size of memory on
  /// all devices as temporary memory
  void setTempMemory(size_t size);

  /// Specify that we wish to use a certain fraction of memory on
  /// all devices as temporary memory
  void setTempMemoryFraction(float fraction);

  /// Set amount of pinned memory to allocate, for async GPU <-> CPU
  /// transfers
  void setPinnedMemory(size_t size);

  /// Called to change the stream for work ordering
  void setDefaultStream(int device, cudaStream_t stream);

  /// Called to change the work ordering streams to the null stream
  /// for all devices
  void setDefaultNullStreamAllDevices();

 public:
  /// Internal system calls
  void initializeForDevice(int device) override;

  cublasHandle_t getBlasHandle(int device) override;

  cudaStream_t getDefaultStream(int device) override;

  std::vector<cudaStream_t> getAlternateStreams(int device) override;

  DeviceMemory& getMemoryManager(int device) override;

  std::pair<void*, size_t> getPinnedMemory() override;

  cudaStream_t getAsyncCopyStream(int device) override;

 private:
  /// Our default stream that work is ordered on, one per each device
  std::unordered_map<int, cudaStream_t> defaultStreams_;

  /// This contains particular streams as set by the user for
  /// ordering, if any
  std::unordered_map<int, cudaStream_t> userDefaultStreams_;

  /// Other streams we can use, per each device
  std::unordered_map<int, std::vector<cudaStream_t> > alternateStreams_;

  /// Async copy stream to use for GPU <-> CPU pinned memory copies
  std::unordered_map<int, cudaStream_t> asyncCopyStreams_;

  /// cuBLAS handle for each device
  std::unordered_map<int, cublasHandle_t> blasHandles_;

  /// Temporary memory provider, per each device
  std::unordered_map<int, std::unique_ptr<StackDeviceMemory> > memory_;

  /// Pinned memory allocation for use with this GPU
  void* pinnedMemAlloc_;
  size_t pinnedMemAllocSize_;

  /// By default, we reserve this fraction of memory on all devices
  float tempMemFraction_;

  /// Another option is to use a specified amount of memory on all
  /// devices
  size_t tempMemSize_;

  /// Whether we look at tempMemFraction_ or tempMemSize_
  bool useFraction_;

  /// Amount of pinned memory we should allocate
  size_t pinnedMemSize_;
};

} } // namespace
