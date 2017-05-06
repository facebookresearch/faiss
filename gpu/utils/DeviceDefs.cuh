
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

namespace faiss { namespace gpu {

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ <= 610
constexpr int kWarpSize = 32;
#else
#error Unknown __CUDA_ARCH__; please define parameters for compute capability
#endif // __CUDA_ARCH__ types
#endif // __CUDA_ARCH__

#ifndef __CUDA_ARCH__
// dummy value for host compiler
constexpr int kWarpSize = 32;
#endif // !__CUDA_ARCH__

__forceinline__ __device__ void warpFence() {
  // Technically, memory barriers are required via the CUDA
  // programming model, since warp synchronous programming no longer
  // is guaranteed.
  //
  // There are two components to it:
  // -a barrier known to the compiler such that the compiler will not
  // schedule loads and stores across the barrier;
  // -a HW-level barrier that guarantees that writes are seen in the
  // proper order
  //
  // However, __threadfence_block() is a stronger constraint than what
  // we really want out of the hardware: a warp-wide barrier.
  //
  // In current hardware, it appears that warp synchronous programming
  // is a reality; by all tests it appears safe and race-free.
  //
  // However, understandably it may not be in the future (based on
  // what Nvidia says in the Kepler guide, it may change depending
  // upon compiler/toolchain issues or future hardware).
  //
  // Removing the fence results in 10%+ faster performance.
  // However, we are judicious as to where we insert the fence, so if
  // this reality ever changes, uncommenting this will result in CUDA
  // programming model-safe ordering again.
  //
  // FIXME: we should probably qualify as volatile as well, since the
  // compiler could technically preserve values across loops? This
  // seems very impractical for the compiler to do, however.

  //  __threadfence_block();
}

} } // namespace
