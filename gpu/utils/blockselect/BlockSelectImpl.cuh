
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
#include "../BlockSelectKernel.cuh"
#include "../Limits.cuh"

#define BLOCK_SELECT_DECL(TYPE, DIR, WARP_Q)                            \
  extern void runBlockSelect_ ## TYPE ## _ ## DIR ## _ ## WARP_Q ## _(  \
    Tensor<TYPE, 2, true>& in,                                          \
    Tensor<TYPE, 2, true>& outK,                                        \
    Tensor<int, 2, true>& outV,                                         \
    bool dir,                                                           \
    int k,                                                              \
    cudaStream_t stream)

#define BLOCK_SELECT_IMPL(TYPE, DIR, WARP_Q, THREAD_Q)                  \
  void runBlockSelect_ ## TYPE ## _ ## DIR ## _ ## WARP_Q ## _(         \
    Tensor<TYPE, 2, true>& in,                                          \
    Tensor<TYPE, 2, true>& outK,                                        \
    Tensor<int, 2, true>& outV,                                         \
    bool dir,                                                           \
    int k,                                                              \
    cudaStream_t stream) {                                              \
    auto grid = dim3(in.getSize(0));                                    \
                                                                        \
    constexpr int kBlockSelectNumThreads = 128;                         \
    auto block = dim3(kBlockSelectNumThreads);                          \
                                                                        \
    FAISS_ASSERT(k <= WARP_Q);                                          \
    FAISS_ASSERT(dir == DIR);                                           \
                                                                        \
    auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax(); \
    auto vInit = -1;                                                    \
                                                                        \
    blockSelect<TYPE, int, DIR, WARP_Q, THREAD_Q, kBlockSelectNumThreads> \
      <<<grid, block, 0, stream>>>(in, outK, outV, kInit, vInit, k);    \
  }

#define BLOCK_SELECT_CALL(TYPE, DIR, WARP_Q)                    \
  runBlockSelect_ ## TYPE ## _ ## DIR ## _ ## WARP_Q ## _(      \
    in, outK, outV, dir, k, stream)
