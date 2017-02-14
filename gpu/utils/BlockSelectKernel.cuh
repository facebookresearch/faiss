
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "Float16.cuh"
#include "Select.cuh"

namespace faiss { namespace gpu {

template <typename K,
          typename IndexType,
          bool Dir,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
__global__ void blockSelect(Tensor<K, 2, true> in,
                            Tensor<K, 2, true> outK,
                            Tensor<IndexType, 2, true> outV,
                            K initK,
                            IndexType initV,
                            int k) {
  constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

  __shared__ K smemK[kNumWarps * NumWarpQ];
  __shared__ IndexType smemV[kNumWarps * NumWarpQ];

  BlockSelect<K, IndexType, Dir, Comparator<K>,
            NumWarpQ, NumThreadQ, ThreadsPerBlock>
    heap(initK, initV, smemK, smemV, k);

  int row = blockIdx.x;

  // Whole warps must participate in the selection
  int limit = utils::roundDown(in.getSize(1), kWarpSize);
  int i = threadIdx.x;

  for (; i < limit; i += blockDim.x) {
    heap.add(in[row][i], (IndexType) i);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < in.getSize(1)) {
    heap.addThreadQ(in[row][i], (IndexType) i);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    outK[row][i] = smemK[i];
    outV[row][i] = smemV[i];
  }
}

void runBlockSelect(Tensor<float, 2, true>& in,
                  Tensor<float, 2, true>& outKeys,
                  Tensor<int, 2, true>& outIndices,
                  bool dir, int k, cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runBlockSelect(Tensor<half, 2, true>& in,
                  Tensor<half, 2, true>& outKeys,
                  Tensor<int, 2, true>& outIndices,
                  bool dir, int k, cudaStream_t stream);
#endif

} } // namespace
