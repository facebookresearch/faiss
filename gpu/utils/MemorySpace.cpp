/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "MemorySpace.h"
#include <cuda_runtime.h>

namespace faiss { namespace gpu {

/// Allocates CUDA memory for a given memory space
void allocMemorySpace(MemorySpace space, void** p, size_t size) {
  if (space == MemorySpace::Device) {
    FAISS_ASSERT_FMT(cudaMalloc(p, size) == cudaSuccess,
                     "Failed to cudaMalloc %zu bytes", size);
  }
#ifdef FAISS_UNIFIED_MEM
  else if (space == MemorySpace::Unified) {
    FAISS_ASSERT_FMT(cudaMallocManaged(p, size) == cudaSuccess,
                     "Failed to cudaMallocManaged %zu bytes", size);
  }
#endif
  else {
    FAISS_ASSERT_FMT(false, "Unknown MemorySpace %d", (int) space);
  }
}

} }
