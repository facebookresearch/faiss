
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "utils/DeviceMemory.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <utility>
#include <vector>

namespace faiss { namespace gpu {

/// Base class of GPU-side resource provider; hides provision of
/// cuBLAS handles, CUDA streams and a temporary memory manager
class GpuResources {
 public:
  virtual ~GpuResources();

  /// Call to pre-allocate resources for a particular device. If this is
  /// not called, then resources will be allocated at the first time
  /// of demand
  virtual void initializeForDevice(int device) = 0;

  virtual cublasHandle_t getBlasHandle(int device) = 0;

  virtual cudaStream_t getDefaultStream(int device) = 0;

  virtual std::vector<cudaStream_t> getAlternateStreams(int device) = 0;

  virtual DeviceMemory& getMemoryManager(int device) = 0;

  virtual std::pair<void*, size_t> getPinnedMemory() = 0;

  virtual cudaStream_t getAsyncCopyStream(int device) = 0;

  cublasHandle_t getBlasHandleCurrentDevice();

  cudaStream_t getDefaultStreamCurrentDevice();

  std::vector<cudaStream_t> getAlternateStreamsCurrentDevice();

  DeviceMemory& getMemoryManagerCurrentDevice();

  cudaStream_t getAsyncCopyStreamCurrentDevice();
};

} } // namespace
