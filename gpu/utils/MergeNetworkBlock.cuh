
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "DeviceDefs.cuh"
#include "PtxUtils.cuh"
#include "StaticUtils.h"
#include "WarpShuffles.cuh"
#include "../../FaissAssert.h"
#include <cuda.h>

namespace faiss { namespace gpu {

// Merge pairs of lists smaller than blockDim.x (NumThreads)
template <int NumThreads, typename K, typename V, int L,
          bool Dir, typename Comp>
inline __device__ void blockMergeSmall(K* listK, V* listV) {
  static_assert(utils::isPowerOf2(L), "L must be a power-of-2");
  static_assert(utils::isPowerOf2(NumThreads),
                "NumThreads must be a power-of-2");
  static_assert(L <= NumThreads, "merge list size must be <= NumThreads");

  // Which pair of lists we are merging
  int mergeId = threadIdx.x / L;

  // Which thread we are within the merge
  int tid = threadIdx.x % L;

  // listK points to a region of size N * 2 * L
  listK += 2 * L * mergeId;
  listV += 2 * L * mergeId;

  // It's not a bitonic merge, both lists are in the same direction,
  // so handle the first swap assuming the second list is reversed
  int pos = L - 1 - tid;
  int stride = 2 * tid + 1;

  K ka = listK[pos];
  K kb = listK[pos + stride];

  bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
  listK[pos] = swap ? kb : ka;
  listK[pos + stride] = swap ? ka : kb;

  V va = listV[pos];
  V vb = listV[pos + stride];
  listV[pos] = swap ? vb : va;
  listV[pos + stride] = swap ? va : vb;

  __syncthreads();

#pragma unroll
  for (int stride = L / 2; stride > 0; stride /= 2) {
    int pos = 2 * tid - (tid & (stride - 1));

    K ka = listK[pos];
    K kb = listK[pos + stride];

    bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
    listK[pos] = swap ? kb : ka;
    listK[pos + stride] = swap ? ka : kb;

    V va = listV[pos];
    V vb = listV[pos + stride];
    listV[pos] = swap ? vb : va;
    listV[pos + stride] = swap ? va : vb;

    __syncthreads();
  }
}

// Merge pairs of sorted lists larger than blockDim.x (NumThreads)
template <int NumThreads, typename K, typename V, int L,
          bool Dir, typename Comp>
inline __device__ void blockMergeLarge(K* listK, V* listV) {
  static_assert(utils::isPowerOf2(L), "L must be a power-of-2");
  static_assert(L >= kWarpSize, "merge list size must be >= 32");
  static_assert(utils::isPowerOf2(NumThreads),
                "NumThreads must be a power-of-2");
  static_assert(L >= NumThreads, "merge list size must be >= NumThreads");

  // For L > NumThreads, each thread has to perform more work
  // per each stride.
  constexpr int kLoopPerThread = L / NumThreads;

  // It's not a bitonic merge, both lists are in the same direction,
  // so handle the first swap assuming the second list is reversed
#pragma unroll
  for (int loop = 0; loop < kLoopPerThread; ++loop) {
    int tid = loop * NumThreads + threadIdx.x;
    int pos = L - 1 - tid;
    int stride = 2 * tid + 1;

    K ka = listK[pos];
    K kb = listK[pos + stride];

    bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
    listK[pos] = swap ? kb : ka;
    listK[pos + stride] = swap ? ka : kb;

    V va = listV[pos];
    V vb = listV[pos + stride];
    listV[pos] = swap ? vb : va;
    listV[pos + stride] = swap ? va : vb;
  }

  __syncthreads();

#pragma unroll
  for (int stride = L / 2; stride > 0; stride /= 2) {
#pragma unroll
    for (int loop = 0; loop < kLoopPerThread; ++loop) {
      int tid = loop * NumThreads + threadIdx.x;
      int pos = 2 * tid - (tid & (stride - 1));

      K ka = listK[pos];
      K kb = listK[pos + stride];

      bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
      listK[pos] = swap ? kb : ka;
      listK[pos + stride] = swap ? ka : kb;

      V va = listV[pos];
      V vb = listV[pos + stride];
      listV[pos] = swap ? vb : va;
      listV[pos + stride] = swap ? va : vb;
    }

    __syncthreads();
  }
}

/// Class template to prevent static_assert from firing for
/// mixing smaller/larger than block cases
template <int NumThreads,
          typename K, typename V, int N, int L,
          bool Dir, typename Comp, bool SmallerThanBlock>
struct BlockMerge {
};

/// Merging lists smaller than a block
template <int NumThreads, typename K, typename V, int N, int L,
          bool Dir, typename Comp>
struct BlockMerge<NumThreads, K, V, N, L, Dir, Comp, true> {
  static inline __device__ void merge(K* listK, V* listV) {
    constexpr int kNumParallelMerges = NumThreads / L;
    constexpr int kNumIterations = N / kNumParallelMerges;

    static_assert(L <= NumThreads, "list must be <= NumThreads");
    static_assert((N < kNumParallelMerges) ||
                  (kNumIterations * kNumParallelMerges == N),
                  "improper selection of N and L");

    if (N < kNumParallelMerges) {
      // We only need L threads per each list to perform the merge
      if (threadIdx.x < N * L) {
        blockMergeSmall<NumThreads, K, V, L, Dir, Comp>(listK, listV);
      }
    } else {
      // All threads participate
#pragma unroll
      for (int i = 0; i < kNumIterations; ++i) {
        int start = i * kNumParallelMerges * 2 * L;

        blockMergeSmall<NumThreads, K, V, L, Dir, Comp>(listK + start,
                                                        listV + start);
      }
    }
  }
};

/// Merging lists larger than a block
template <int NumThreads, typename K, typename V, int N, int L,
          bool Dir, typename Comp>
struct BlockMerge<NumThreads, K, V, N, L, Dir, Comp, false> {
  static inline __device__ void merge(K* listK, V* listV) {
    // Each pair of lists is merged sequentially
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int start = i * 2 * L;

      blockMergeLarge<NumThreads, K, V, L, Dir, Comp>(listK + start,
                                                      listV + start);
    }
  }
};

template <int NumThreads, typename K, typename V, int N, int L,
          bool Dir, typename Comp>
inline __device__ void blockMerge(K* listK, V* listV) {
  constexpr bool kSmallerThanBlock = (L <= NumThreads);

  BlockMerge<NumThreads, K, V, N, L, Dir, Comp, kSmallerThanBlock>::
    merge(listK, listV);
}

} } // namespace
