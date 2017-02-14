
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "GpuResources.h"
#include "utils/DeviceUtils.h"

namespace faiss { namespace gpu {

GpuResources::~GpuResources() {
}

cublasHandle_t
GpuResources::getBlasHandleCurrentDevice() {
  return getBlasHandle(getCurrentDevice());
}

cudaStream_t
GpuResources::getDefaultStreamCurrentDevice() {
  return getDefaultStream(getCurrentDevice());
}

std::vector<cudaStream_t>
GpuResources::getAlternateStreamsCurrentDevice() {
  return getAlternateStreams(getCurrentDevice());
}

DeviceMemory&
GpuResources::getMemoryManagerCurrentDevice() {
  return getMemoryManager(getCurrentDevice());
}

cudaStream_t
GpuResources::getAsyncCopyStreamCurrentDevice() {
  return getAsyncCopyStream(getCurrentDevice());
}

} } // namespace
