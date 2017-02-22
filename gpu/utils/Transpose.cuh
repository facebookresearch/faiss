
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../../FaissAssert.h"
#include "Tensor.cuh"
#include "DeviceUtils.h"
#include <cuda.h>

#include <stdio.h>

namespace faiss { namespace gpu {

template <typename T>
struct TensorInfo {
  static constexpr int kMaxDims = 8;

  T* data;
  int sizes[kMaxDims];
  int strides[kMaxDims];
  int dims;
};

template <typename T, int Dim>
struct TensorInfoOffset {
  __device__ inline static unsigned int get(const TensorInfo<T>& info,
                                            unsigned int linearId) {
    unsigned int offset = 0;

#pragma unroll
    for (int i = Dim - 1; i >= 0; --i) {
      unsigned int curDimIndex = linearId % info.sizes[i];
      unsigned int curDimOffset = curDimIndex * info.strides[i];

      offset += curDimOffset;

      if (i > 0) {
        linearId /= info.sizes[i];
      }
    }

    return offset;
  }
};

template <typename T>
struct TensorInfoOffset<T, -1> {
  __device__ inline static unsigned int get(const TensorInfo<T>& info,
                                            unsigned int linearId) {
    return linearId;
  }
};

template <typename T, int Dim>
TensorInfo<T> getTensorInfo(const Tensor<T, Dim, true>& t) {
  TensorInfo<T> info;

  for (int i = 0; i < Dim; ++i) {
    info.sizes[i] = t.getSize(i);
    info.strides[i] = t.getStride(i);
  }

  info.data = t.data();
  info.dims = Dim;

  return info;
}

template <typename T, int DimInput, int DimOutput>
__global__ void transposeAny(TensorInfo<T> input,
                             TensorInfo<T> output,
                             unsigned int totalSize) {
  auto linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearThreadId >= totalSize) {
    return;
  }

  auto inputOffset =
    TensorInfoOffset<T, DimInput>::get(input, linearThreadId);
  auto outputOffset =
    TensorInfoOffset<T, DimOutput>::get(output, linearThreadId);

  output.data[outputOffset] = __ldg(&input.data[inputOffset]);
}

/// Performs an out-of-place transposition between any two dimensions.
/// Best performance is if the transposed dimensions are not
/// innermost, since the reads and writes will be coalesced.
/// Could include a shared memory transposition if the dimensions
/// being transposed are innermost, but would require support for
/// arbitrary rectangular matrices.
/// This linearized implementation seems to perform well enough,
/// especially for cases that we care about (outer dimension
/// transpositions).
template <typename T, int Dim>
void runTransposeAny(Tensor<T, Dim, true>& in,
                     int dim1, int dim2,
                     Tensor<T, Dim, true>& out,
                     cudaStream_t stream) {
  static_assert(Dim <= TensorInfo<T>::kMaxDims, "too many dimensions");

  FAISS_ASSERT(dim1 != dim2);
  FAISS_ASSERT(dim1 < Dim && dim2 < Dim);

  int outSize[Dim];

  for (int i = 0; i < Dim; ++i) {
    outSize[i] = in.getSize(i);
  }

  std::swap(outSize[dim1], outSize[dim2]);

  for (int i = 0; i < Dim; ++i) {
    FAISS_ASSERT(out.getSize(i) == outSize[i]);
  }

  auto inInfo = getTensorInfo<T, Dim>(in);
  auto outInfo = getTensorInfo<T, Dim>(out);

  std::swap(inInfo.sizes[dim1], inInfo.sizes[dim2]);
  std::swap(inInfo.strides[dim1], inInfo.strides[dim2]);

  int totalSize = in.numElements();

  int numThreads = std::min(getMaxThreadsCurrentDevice(), totalSize);
  auto grid = dim3(utils::divUp(totalSize, numThreads));
  auto block = dim3(numThreads);

  transposeAny<T, Dim, -1><<<grid, block, 0, stream>>>(inInfo, outInfo, totalSize);
  CUDA_VERIFY(cudaGetLastError());
}

} } // namespace
