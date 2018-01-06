/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

namespace faiss { namespace gpu {

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ <= 700
constexpr int kWarpSize = 32;
#else
#error Unknown __CUDA_ARCH__; please define parameters for compute capability
#endif // __CUDA_ARCH__ types
#endif // __CUDA_ARCH__

#ifndef __CUDA_ARCH__
// dummy value for host compiler
constexpr int kWarpSize = 32;
#endif // !__CUDA_ARCH__

// This is a memory barrier for intra-warp writes to shared memory.
__forceinline__ __device__ void warpFence() {

#if __CUDA_ARCH__ >= 700
  __syncwarp();
#else
  // For the time being, assume synchronicity.
  //  __threadfence_block();
#endif
}

} } // namespace
