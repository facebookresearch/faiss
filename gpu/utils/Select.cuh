
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "Comparators.cuh"
#include "DeviceDefs.cuh"
#include "MergeNetworkBlock.cuh"
#include "MergeNetworkWarp.cuh"
#include "PtxUtils.cuh"
#include "Reductions.cuh"
#include "ReductionOperators.cuh"
#include "Tensor.cuh"

namespace faiss { namespace gpu {

// Specialization for block-wide monotonic merges producing a merge sort
// since what we really want is a constexpr loop expansion
template <int NumWarps,
          int NumThreads, typename K, typename V, int NumWarpQ,
          bool Dir, typename Comp>
struct FinalBlockMerge {
};

template <int NumThreads, typename K, typename V, int NumWarpQ,
          bool Dir, typename Comp>
struct FinalBlockMerge<1, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* sharedK, V* sharedV) {
    // no merge required; single warp
  }
};

template <int NumThreads, typename K, typename V, int NumWarpQ,
          bool Dir, typename Comp>
struct FinalBlockMerge<2, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* sharedK, V* sharedV) {
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 2),
               NumWarpQ, !Dir, Comp>(sharedK, sharedV);
  }
};

template <int NumThreads, typename K, typename V, int NumWarpQ,
          bool Dir, typename Comp>
struct FinalBlockMerge<4, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* sharedK, V* sharedV) {
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 2),
               NumWarpQ, !Dir, Comp>(sharedK, sharedV);
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 4),
               NumWarpQ * 2, !Dir, Comp>(sharedK, sharedV);
  }
};

template <int NumThreads, typename K, typename V, int NumWarpQ,
          bool Dir, typename Comp>
struct FinalBlockMerge<8, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* sharedK, V* sharedV) {
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 2),
               NumWarpQ, !Dir, Comp>(sharedK, sharedV);
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 4),
               NumWarpQ * 2, !Dir, Comp>(sharedK, sharedV);
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 8),
               NumWarpQ * 4, !Dir, Comp>(sharedK, sharedV);
  }
};

// `Dir` true, produce largest values.
// `Dir` false, produce smallest values.
template <typename K,
          typename V,
          bool Dir,
          typename Comp,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
struct BlockSelect {
  static constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;
  static constexpr int kTotalWarpSortSize = NumWarpQ;

  __device__ inline BlockSelect(K initK, V initV, K* smemK, V* smemV, int k) :
      sharedK(smemK),
      sharedV(smemV),
      warpKTop(initK),
      kMinus1(k - 1) {
    static_assert(utils::isPowerOf2(ThreadsPerBlock),
                  "threads must be a power-of-2");
    static_assert(utils::isPowerOf2(NumWarpQ),
                  "warp queue must be power-of-2");

    // Fill the per-thread queue keys with the default value; values
    // can remain uninitialized
#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      threadK[i] = initK;
      threadV[i] = initV;
    }

    int laneId = getLaneId();
    int warpId = threadIdx.x / kWarpSize;
    warpK = sharedK + warpId * kTotalWarpSortSize;
    warpV = sharedV + warpId * kTotalWarpSortSize;

    // Fill warp queue (only the actual queue space is fine, not where
    // we write the per-thread queues for merging)
    for (int i = laneId; i < NumWarpQ; i += kWarpSize) {
      warpK[i] = initK;
      warpV[i] = initV;
    }

    warpFence();
  }

  __device__ inline void addThreadQ(K k, V v) {
    // If we're greater or equal to the highest element, then we don't
    // need to add. In the equal to case, the element we have is
    // sufficient.
    if (Dir ? Comp::gt(k, threadK[0]) : Comp::lt(k, threadK[0])) {
      threadK[0] = k;
      threadV[0] = v;

      // Perform in-register bubble sort of this new element
#pragma unroll
      for (int i = 1; i < NumThreadQ; ++i) {
        bool swap = Dir ? Comp::lt(threadK[i], threadK[i - 1]) :
          Comp::gt(threadK[i], threadK[i - 1]);

        K tmpK = threadK[i];
        threadK[i] = swap ? threadK[i - 1] : tmpK;
        threadK[i - 1] = swap ? tmpK : threadK[i - 1];

        V tmpV = threadV[i];
        threadV[i] = swap ? threadV[i - 1] : tmpV;
        threadV[i - 1] = swap ? tmpV : threadV[i - 1];
      }
    }
  }

  __device__ inline void checkThreadQ() {
    // There is no need to merge queues if no thread-local queue is
    // better than the warp-wide queue
    bool needSort = (Dir ?
                     Comp::gt(threadK[0], warpKTop) :
                     Comp::lt(threadK[0], warpKTop));
    if (!__any(needSort)) {
      return;
    }

    // This has a trailing warpFence
    mergeWarpQ();

    // We have to beat at least this element
    warpKTop = warpK[kMinus1];

    warpFence();
  }

  /// This function handles sorting and merging together the
  /// per-thread queues with the warp-wide queue, creating a sorted
  /// list across both
  __device__ inline void mergeWarpQ() {
    int laneId = getLaneId();

    // Sort all of the per-thread queues
    warpSortAnyRegisters<K, V, NumThreadQ, !Dir, Comp>(threadK, threadV);

    constexpr int kNumWarpQRegisters = NumWarpQ / kWarpSize;
    K warpKRegisters[kNumWarpQRegisters];
    V warpVRegisters[kNumWarpQRegisters];

#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warpKRegisters[i] = warpK[i * kWarpSize + laneId];
      warpVRegisters[i] = warpV[i * kWarpSize + laneId];
    }

    warpFence();

    // The warp queue is already sorted, and now that we've sorted the
    // per-thread queue, merge both sorted lists together, producing
    // one sorted list
    warpMergeAnyRegisters<K, V, kNumWarpQRegisters, NumThreadQ, !Dir, Comp>(
      warpKRegisters, warpVRegisters, threadK, threadV);

    // Write back out the warp queue
#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warpK[i * kWarpSize + laneId] = warpKRegisters[i];
      warpV[i * kWarpSize + laneId] = warpVRegisters[i];
    }

    warpFence();

    // Re-map the registers for our per-thread queues
    K tmpThreadK[NumThreadQ];
    V tmpThreadV[NumThreadQ];

#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      tmpThreadK[i] = threadK[i];
      tmpThreadV[i] = threadV[i];
    }

#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      // After merging, the data is in the order small -> large.
      // We wish to reload data in our thread queues, which have the
      // order large -> small by indexing, so it will be reverse order.
      // However, we also wish to give every thread an equal shot at the
      // largest elements, so we interleave the loading.
      threadK[i] = tmpThreadK[NumThreadQ - i - 1];
      threadV[i] = tmpThreadV[NumThreadQ - i - 1];
    }
  }

  /// WARNING: all threads in a warp must participate in this.
  /// Otherwise, you must call the constituent parts separately.
  __device__ inline void add(K k, V v) {
    addThreadQ(k, v);
    checkThreadQ();
  }

  __device__ inline void reduce() {
    // Have all warps dump and merge their queues; this will produce
    // the final per-warp results
    mergeWarpQ();

    // block-wide dep; thus far, all warps have been completely
    // independent
    __syncthreads();

    // All warp queues are contiguous in smem.
    // Now, we have kNumWarps lists of NumWarpQ elements.
    // This is a power of 2.
    FinalBlockMerge<kNumWarps, ThreadsPerBlock, K, V, NumWarpQ, Dir, Comp>::
      merge(sharedK, sharedV);

    // The block-wide merge has a trailing syncthreads
  }

  // threadK[0] is lowest (Dir) or highest (!Dir)
  K threadK[NumThreadQ];
  V threadV[NumThreadQ];

  // Queues for all warps
  K* sharedK;
  V* sharedV;

  // Our warp's queue (points into sharedK/sharedV)
  // warpK[0] is highest (Dir) or lowest (!Dir)
  K* warpK;
  V* warpV;

  // Warp queue head value cached in a register, so we needn't read
  // shared memory as frequently
  K warpKTop;

  // This is a cached k-1 value
  int kMinus1;
};

/// Specialization for k == 1 (NumWarpQ == 1)
template <typename K,
          typename V,
          bool Dir,
          typename Comp,
          int NumThreadQ,
          int ThreadsPerBlock>
struct BlockSelect<K, V, Dir, Comp, 1, NumThreadQ, ThreadsPerBlock> {
  static constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

  __device__ inline BlockSelect(K initK, V initV, K* smemK, V* smemV, int k) :
      sharedK(smemK),
      sharedV(smemV),
      threadK(initK),
      threadV(initV) {
  }

  __device__ inline void addThreadQ(K k, V v) {
    bool swap = Dir ? Comp::gt(k, threadK) : Comp::lt(k, threadK);
    threadK = swap ? k : threadK;
    threadV = swap ? v : threadV;
  }

  __device__ inline void checkThreadQ() {
    // We don't need to do anything here, since the warp doesn't
    // cooperate until the end
  }

  __device__ inline void add(K k, V v) {
    addThreadQ(k, v);
  }

  __device__ inline void reduce() {
    // Reduce within the warp
    Pair<K, V> pair(threadK, threadV);

    if (Dir) {
      pair =
        warpReduceAll<Pair<K, V>, Max<Pair<K, V>>>(pair, Max<Pair<K, V>>());
    } else {
      pair =
        warpReduceAll<Pair<K, V>, Min<Pair<K, V>>>(pair, Min<Pair<K, V>>());
    }

    // Each warp writes out a single value
    int laneId = getLaneId();
    int warpId = threadIdx.x / kWarpSize;

    if (laneId == 0) {
      sharedK[warpId] = pair.k;
      sharedV[warpId] = pair.v;
    }

    __syncthreads();

    // We typically use this for small blocks (<= 128), just having the first
    // thread in the block perform the reduction across warps is
    // faster
    if (threadIdx.x == 0) {
      threadK = sharedK[0];
      threadV = sharedV[0];

#pragma unroll
      for (int i = 1; i < kNumWarps; ++i) {
        K k = sharedK[i];
        V v = sharedV[i];

        bool swap = Dir ? Comp::gt(k, threadK) : Comp::lt(k, threadK);
        threadK = swap ? k : threadK;
        threadV = swap ? v : threadV;
      }

      // Hopefully a thread's smem reads/writes are ordered wrt
      // itself, so no barrier needed :)
      sharedK[0] = threadK;
      sharedV[0] = threadV;
    }

    // In case other threads wish to read this value
    __syncthreads();
  }

  // threadK is lowest (Dir) or highest (!Dir)
  K threadK;
  V threadV;

  // Where we reduce in smem
  K* sharedK;
  V* sharedV;
};

//
// per-warp WarpSelect
//

// `Dir` true, produce largest values.
// `Dir` false, produce smallest values.
template <typename K,
          typename V,
          bool Dir,
          typename Comp,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
struct WarpSelect {
  static constexpr int kNumWarpQRegisters = NumWarpQ / kWarpSize;

  __device__ inline WarpSelect(K initK, V initV, int k) :
      warpKTop(initK),
      kLane((k - 1) % kWarpSize) {
    static_assert(utils::isPowerOf2(ThreadsPerBlock),
                  "threads must be a power-of-2");
    static_assert(utils::isPowerOf2(NumWarpQ),
                  "warp queue must be power-of-2");

    // Fill the per-thread queue keys with the default value; values
    // can remain uninitialized
#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      threadK[i] = initK;
      threadV[i] = initV;
    }

    // Fill the warp queue with the default value
#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warpK[i] = initK;
      warpV[i] = initV;
    }
  }

  __device__ inline void addThreadQ(K k, V v) {
    // If we're greater or equal to the highest element, then we don't
    // need to add. In the equal to case, the element we have is
    // sufficient.
    if (Dir ? Comp::gt(k, threadK[0]) : Comp::lt(k, threadK[0])) {
      threadK[0] = k;
      threadV[0] = v;

      // Perform in-register bubble sort of this new element
#pragma unroll
      for (int i = 1; i < NumThreadQ; ++i) {
        bool swap = Dir ? Comp::lt(threadK[i], threadK[i - 1]) :
          Comp::gt(threadK[i], threadK[i - 1]);

        K tmpK = threadK[i];
        threadK[i] = swap ? threadK[i - 1] : tmpK;
        threadK[i - 1] = swap ? tmpK : threadK[i - 1];

        V tmpV = threadV[i];
        threadV[i] = swap ? threadV[i - 1] : tmpV;
        threadV[i - 1] = swap ? tmpV : threadV[i - 1];
      }
    }
  }

  __device__ inline void checkThreadQ() {
    // There is no need to merge queues if no thread-local queue is
    // better than the warp-wide queue
    bool needSort = (Dir ?
                     Comp::gt(threadK[0], warpKTop) :
                     Comp::lt(threadK[0], warpKTop));
    if (!__any(needSort)) {
      return;
    }

    mergeWarpQ();

    // We have to beat at least this element
    warpKTop = shfl(warpK[kNumWarpQRegisters - 1], kLane);
  }

  /// This function handles sorting and merging together the
  /// per-thread queues with the warp-wide queue, creating a sorted
  /// list across both
  __device__ inline void mergeWarpQ() {
    // Sort all of the per-thread queues
    warpSortAnyRegisters<K, V, NumThreadQ, !Dir, Comp>(threadK, threadV);

    // The warp queue is already sorted, and now that we've sorted the
    // per-thread queue, merge both sorted lists together, producing
    // one sorted list
    warpMergeAnyRegisters<K, V, kNumWarpQRegisters, NumThreadQ, !Dir, Comp>(
      warpK, warpV, threadK, threadV);

    // Re-map the registers for our per-thread queues
    K tmpThreadK[NumThreadQ];
    V tmpThreadV[NumThreadQ];

#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      tmpThreadK[i] = threadK[i];
      tmpThreadV[i] = threadV[i];
    }

#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      // After merging, the data is in the order small -> large.
      // We wish to reload data in our thread queues, which have the
      // order large -> small by indexing, so it will be reverse order.
      // However, we also wish to give every thread an equal shot at the
      // largest elements, so we interleave the loading.
      threadK[i] = tmpThreadK[NumThreadQ - i - 1];
      threadV[i] = tmpThreadV[NumThreadQ - i - 1];
    }
  }

  /// WARNING: all threads in a warp must participate in this.
  /// Otherwise, you must call the constituent parts separately.
  __device__ inline void add(K k, V v) {
    addThreadQ(k, v);
    checkThreadQ();
  }

  __device__ inline void reduce() {
    // Have all warps dump and merge their queues; this will produce
    // the final per-warp results
    mergeWarpQ();
  }

  /// Dump final k selected values for this warp out
  __device__ inline void writeOut(K* outK, V* outV, int k) {
    int laneId = getLaneId();

#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      int idx = i * kWarpSize + laneId;

      if (idx < k) {
        outK[idx] = warpK[i];
        outV[idx] = warpV[i];
      }
    }
  }

  // threadK[0] is lowest (Dir) or highest (!Dir)
  K threadK[NumThreadQ];
  V threadV[NumThreadQ];

  // warpK[0] is highest (Dir) or lowest (!Dir)
  K warpK[kNumWarpQRegisters];
  V warpV[kNumWarpQRegisters];

  // Warp queue head value cached in a register, so we needn't read
  // shared memory as frequently
  K warpKTop;

  // This is what lane we should load an approximation (>=k) to the
  // kth element from the last register in the warp queue (i.e.,
  // warpK[kNumWarpQRegisters - 1]).
  int kLane;
};

/// Specialization for k == 1 (NumWarpQ == 1)
template <typename K,
          typename V,
          bool Dir,
          typename Comp,
          int NumThreadQ,
          int ThreadsPerBlock>
struct WarpSelect<K, V, Dir, Comp, 1, NumThreadQ, ThreadsPerBlock> {
  static constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

  __device__ inline WarpSelect(K initK, V initV, int k) :
      threadK(initK),
      threadV(initV) {
  }

  __device__ inline void addThreadQ(K k, V v) {
    bool swap = Dir ? Comp::gt(k, threadK) : Comp::lt(k, threadK);
    threadK = swap ? k : threadK;
    threadV = swap ? v : threadV;
  }

  __device__ inline void checkThreadQ() {
    // We don't need to do anything here, since the warp doesn't
    // cooperate until the end
  }

  __device__ inline void add(K k, V v) {
    addThreadQ(k, v);
  }

  __device__ inline void reduce() {
    // Reduce within the warp
    Pair<K, V> pair(threadK, threadV);

    if (Dir) {
      pair =
        warpReduceAll<Pair<K, V>, Max<Pair<K, V>>>(pair, Max<Pair<K, V>>());
    } else {
      pair =
        warpReduceAll<Pair<K, V>, Min<Pair<K, V>>>(pair, Min<Pair<K, V>>());
    }

    threadK = pair.k;
    threadV = pair.v;
  }

  /// Dump final k selected values for this warp out
  __device__ inline void writeOut(K* outK, V* outV, int k) {
    if (getLaneId() == 0) {
      *outK = threadK;
      *outV = threadV;
    }
  }

  // threadK is lowest (Dir) or highest (!Dir)
  K threadK;
  V threadV;
};

} } // namespace
