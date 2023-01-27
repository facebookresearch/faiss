/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

/// Holder of GPU resources for a particular flat index
/// Can be in either float16 or float32 mode. If float32, we only store
/// the vectors in float32.
/// If float16, we store the vectors in both float16 and float32, where float32
/// data is possibly needed for certain residual operations
class FlatIndex {
   public:
    FlatIndex(GpuResources* res, int dim, bool useFloat16, MemorySpace space);

    /// Whether or not this flat index primarily stores data in float16
    bool getUseFloat16() const;

    /// Returns the number of vectors we contain
    int getSize() const;

    /// Returns the dimensionality of the vectors
    int getDim() const;

    /// Reserve storage that can contain at least this many vectors
    void reserve(size_t numVecs, cudaStream_t stream);

    /// Returns a reference to our vectors currently in use
    Tensor<float, 2, true>& getVectorsFloat32Ref();

    /// Returns a reference to our vectors currently in use (if useFloat16 mode)
    Tensor<half, 2, true>& getVectorsFloat16Ref();

    void query(
            Tensor<float, 2, true>& vecs,
            int k,
            faiss::MetricType metric,
            float metricArg,
            Tensor<float, 2, true>& outDistances,
            Tensor<int, 2, true>& outIndices,
            bool exactDistance);

    void query(
            Tensor<half, 2, true>& vecs,
            int k,
            faiss::MetricType metric,
            float metricArg,
            Tensor<float, 2, true>& outDistances,
            Tensor<int, 2, true>& outIndices,
            bool exactDistance);

    /// Compute residual for set of vectors
    void computeResidual(
            Tensor<float, 2, true>& vecs,
            Tensor<idx_t, 1, true>& ids,
            Tensor<float, 2, true>& residuals);

    /// Gather vectors given the set of IDs
    void reconstruct(Tensor<idx_t, 1, true>& ids, Tensor<float, 2, true>& vecs);

    /// Gather vectors given a range of IDs
    void reconstruct(idx_t start, idx_t num, Tensor<float, 2, true>& vecs);

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

    /// Memory space for our allocations
    MemorySpace space_;

    /// How many vectors we have
    int num_;

    /// The underlying expandable storage for float32 data
    DeviceVector<char> rawData32_;

    /// The underlying expandable storage for float16 data
    DeviceVector<char> rawData16_;

    /// Vectors currently in rawData32_
    DeviceTensor<float, 2, true> vectors_;

    /// Vectors currently in rawData16_, float16 form
    DeviceTensor<half, 2, true> vectorsHalf_;

    /// Precomputed L2 norms
    DeviceTensor<float, 1, true> norms_;
};

} // namespace gpu
} // namespace faiss
