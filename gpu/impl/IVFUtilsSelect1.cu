/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "IVFUtils.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/Limits.cuh"
#include "../utils/Select.cuh"
#include "../utils/StaticUtils.h"
#include "../utils/Tensor.cuh"

//
// This kernel is split into a separate compilation unit to cut down
// on compile time
//

namespace faiss { namespace gpu {

template <int ThreadsPerBlock, int NumWarpQ, int NumThreadQ, bool Dir>
__global__ void
pass1SelectLists(Tensor<int, 2, true> prefixSumOffsets,
                 Tensor<float, 1, true> distance,
                 int nprobe,
                 int k,
                 Tensor<float, 3, true> heapDistances,
                 Tensor<int, 3, true> heapIndices) {
  constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

  __shared__ float smemK[kNumWarps * NumWarpQ];
  __shared__ int smemV[kNumWarps * NumWarpQ];

  constexpr auto kInit = Dir ? kFloatMin : kFloatMax;
  BlockSelect<float, int, Dir, Comparator<float>,
              NumWarpQ, NumThreadQ, ThreadsPerBlock>
    heap(kInit, -1, smemK, smemV, k);

  auto queryId = blockIdx.y;
  auto sliceId = blockIdx.x;
  auto numSlices = gridDim.x;

  int sliceSize = (nprobe / numSlices);
  int sliceStart = sliceSize * sliceId;
  int sliceEnd = sliceId == (numSlices - 1) ? nprobe :
    sliceStart + sliceSize;
  auto offsets = prefixSumOffsets[queryId].data();

  // We ensure that before the array (at offset -1), there is a 0 value
  int start = *(&offsets[sliceStart] - 1);
  int end = offsets[sliceEnd - 1];

  int num = end - start;
  int limit = utils::roundDown(num, kWarpSize);

  int i = threadIdx.x;
  auto distanceStart = distance[start].data();

  // BlockSelect add cannot be used in a warp divergent circumstance; we
  // handle the remainder warp below
  for (; i < limit; i += blockDim.x) {
    heap.add(distanceStart[i], start + i);
  }

  // Handle warp divergence separately
  if (i < num) {
    heap.addThreadQ(distanceStart[i], start + i);
  }

  // Merge all final results
  heap.reduce();

  // Write out the final k-selected values; they should be all
  // together
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    heapDistances[queryId][sliceId][i] = smemK[i];
    heapIndices[queryId][sliceId][i] = smemV[i];
  }
}

void
runPass1SelectLists(Tensor<int, 2, true>& prefixSumOffsets,
                    Tensor<float, 1, true>& distance,
                    int nprobe,
                    int k,
                    bool chooseLargest,
                    Tensor<float, 3, true>& heapDistances,
                    Tensor<int, 3, true>& heapIndices,
                    cudaStream_t stream) {
  constexpr auto kThreadsPerBlock = 128;

  auto grid = dim3(heapDistances.getSize(1), prefixSumOffsets.getSize(0));
  auto block = dim3(kThreadsPerBlock);

#define RUN_PASS(NUM_WARP_Q, NUM_THREAD_Q, DIR)                         \
  do {                                                                  \
    pass1SelectLists<kThreadsPerBlock, NUM_WARP_Q, NUM_THREAD_Q, DIR>   \
      <<<grid, block, 0, stream>>>(prefixSumOffsets,                    \
                                   distance,                            \
                                   nprobe,                              \
                                   k,                                   \
                                   heapDistances,                       \
                                   heapIndices);                        \
    CUDA_TEST_ERROR();                                                  \
    return; /* success */                                               \
  } while (0)

#define RUN_PASS_DIR(DIR)                            \
  do {                                               \
    if (k == 1) {                                    \
      RUN_PASS(1, 1, DIR);                           \
    } else if (k <= 32) {                            \
      RUN_PASS(32, 2, DIR);                          \
    } else if (k <= 64) {                            \
      RUN_PASS(64, 3, DIR);                          \
    } else if (k <= 128) {                           \
      RUN_PASS(128, 3, DIR);                         \
    } else if (k <= 256) {                           \
      RUN_PASS(256, 4, DIR);                         \
    } else if (k <= 512) {                           \
      RUN_PASS(512, 8, DIR);                         \
    } else if (k <= 1024) {                          \
      RUN_PASS(1024, 8, DIR);                        \
    }                                                \
  } while (0)

  if (chooseLargest) {
    RUN_PASS_DIR(true);
  } else {
    RUN_PASS_DIR(false);
  }

  // unimplemented / too many resources
  FAISS_ASSERT_FMT(false, "unimplemented k value (%d)", k);

#undef RUN_PASS_DIR
#undef RUN_PASS
}

} } // namespace
