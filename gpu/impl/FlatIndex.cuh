
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../utils/DeviceTensor.cuh"
#include "../utils/DeviceVector.cuh"
#include "../utils/Float16.cuh"

namespace faiss { namespace gpu {

class GpuResources;

/// Holder of GPU resources for a particular flat index
class FlatIndex {
 public:
  FlatIndex(GpuResources* res,
            int dim,
            bool l2Distance,
            bool useFloat16);

  bool getUseFloat16() const;

  /// Returns the number of vectors we contain
  int getSize() const;

  int getDim() const;

  /// Returns a reference to our vectors currently in use
  Tensor<float, 2, true>& getVectorsFloat32Ref();

#ifdef FAISS_USE_FLOAT16
  /// Returns a reference to our vectors currently in use (useFloat16 mode)
  Tensor<half, 2, true>& getVectorsFloat16Ref();
#endif

  /// Performs a copy of the vectors on the given device, converting
  /// as needed from float16
  DeviceTensor<float, 2, true> getVectorsFloat32Copy(cudaStream_t stream);

  /// Returns only a subset of the vectors
  DeviceTensor<float, 2, true> getVectorsFloat32Copy(int from,
                                                     int num,
                                                     cudaStream_t stream);

  void query(Tensor<float, 2, true>& vecs,
             int k,
             Tensor<float, 2, true>& outDistances,
             Tensor<int, 2, true>& outIndices,
             bool exactDistance,
             int tileSize = -1);

#ifdef FAISS_USE_FLOAT16
  void query(Tensor<half, 2, true>& vecs,
             int k,
             Tensor<half, 2, true>& outDistances,
             Tensor<int, 2, true>& outIndices,
             bool exactDistance,
             int tileSize = -1);
#endif

  /// Add vectors to ourselves; the pointer passed can be on the host
  /// or the device
  void add(const float* data, int numVecs, cudaStream_t stream);

  /// Free all storage
  void reset();

 private:
  /// Collection of GPU resources that we use
  GpuResources* resources_;

  /// Dimensionality of our vectors
  const int dim_;

  /// Float16 data format
  const bool useFloat16_;

  /// L2 or inner product distance?
  bool l2Distance_;

  /// How many vectors we have
  int num_;

  /// The underlying expandable storage
  DeviceVector<char> rawData_;

  /// Vectors currently in rawData_
  DeviceTensor<float, 2, true> vectors_;

#ifdef FAISS_USE_FLOAT16
  /// Vectors currently in rawData_, float16 form
  DeviceTensor<half, 2, true> vectorsHalf_;
#endif

  /// Precomputed L2 norms
  DeviceTensor<float, 1, true> norms_;

#ifdef FAISS_USE_FLOAT16
  /// Precomputed L2 norms, float16 form
  DeviceTensor<half, 1, true> normsHalf_;
#endif
};

} } // namespace
