/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../../FaissAssert.h"
#include <cuda.h>

#if CUDA_VERSION >= 8000
// Whether or not we enable usage of CUDA Unified Memory
#define FAISS_UNIFIED_MEM 1
#endif

namespace faiss { namespace gpu {

enum MemorySpace {
  /// Managed using cudaMalloc/cudaFree
  Device = 1,
  /// Managed using cudaMallocManaged/cudaFree
  Unified = 2,
};

/// Allocates CUDA memory for a given memory space
void allocMemorySpace(MemorySpace space, void** p, size_t size);

} }
