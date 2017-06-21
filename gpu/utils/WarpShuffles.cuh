/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <cuda.h>
#include "DeviceDefs.cuh"
#include "Float16.cuh"

namespace faiss { namespace gpu {

template <typename T>
inline __device__ T shfl(const T val,
                         int srcLane, int width = kWarpSize) {
  return __shfl(val, srcLane, width);
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl(T* const val,
                         int srcLane, int width = kWarpSize) {
  static_assert(sizeof(T*) == sizeof(long long), "pointer size");
  long long v = (long long) val;
  return (T*) __shfl(v, srcLane, width);
}

template <typename T>
inline __device__ T shfl_up(const T val,
                            unsigned int delta, int width = kWarpSize) {
  return __shfl_up(val, delta, width);
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_up(T* const val,
                             unsigned int delta, int width = kWarpSize) {
  static_assert(sizeof(T*) == sizeof(long long), "pointer size");
  long long v = (long long) val;
  return (T*) __shfl_up(v, delta, width);
}

template <typename T>
inline __device__ T shfl_down(const T val,
                              unsigned int delta, int width = kWarpSize) {
  return __shfl_down(val, delta, width);
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_down(T* const val,
                              unsigned int delta, int width = kWarpSize) {
  static_assert(sizeof(T*) == sizeof(long long), "pointer size");
  long long v = (long long) val;
  return (T*) __shfl_down(v, delta, width);
}

template <typename T>
inline __device__ T shfl_xor(const T val,
                             int laneMask, int width = kWarpSize) {
  return __shfl_xor(val, laneMask, width);
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_xor(T* const val,
                              int laneMask, int width = kWarpSize) {
  static_assert(sizeof(T*) == sizeof(long long), "pointer size");
  long long v = (long long) val;
  return (T*) __shfl_xor(v, laneMask, width);
}

#ifdef FAISS_USE_FLOAT16
inline __device__ half shfl(half v,
                            int srcLane, int width = kWarpSize) {
  unsigned int vu = v.x;
  vu = __shfl(vu, srcLane, width);

  half h;
  h.x = (unsigned short) vu;
  return h;
}

inline __device__ half shfl_xor(half v,
                                int laneMask, int width = kWarpSize) {
  unsigned int vu = v.x;
  vu = __shfl_xor(vu, laneMask, width);

  half h;
  h.x = (unsigned short) vu;
  return h;
}
#endif

} } // namespace
