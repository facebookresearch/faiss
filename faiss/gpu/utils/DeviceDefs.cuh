/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

namespace faiss {
namespace gpu {

// We require at least CUDA 8.0 for compilation
#if CUDA_VERSION < 8000
#error "CUDA >= 8.0 is required"
#endif

// We validate this against the actual architecture in device initialization
constexpr int kWarpSize = 32;

// This is a memory barrier for intra-warp writes to shared memory.
__forceinline__ __device__ void warpFence() {
#if CUDA_VERSION >= 9000
    __syncwarp();
#else
    // For the time being, assume synchronicity.
    //  __threadfence_block();
#endif
}

#if CUDA_VERSION > 9000
// Based on the CUDA version (we assume what version of nvcc/ptxas we were
// compiled with), the register allocation algorithm is much better, so only
// enable the 2048 selection code if we are above 9.0 (9.2 seems to be ok)
#define GPU_MAX_SELECTION_K 2048
#else
#define GPU_MAX_SELECTION_K 1024
#endif

} // namespace gpu
} // namespace faiss
