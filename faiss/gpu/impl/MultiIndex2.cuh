/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <unordered_map>

namespace faiss {
namespace gpu {

class GpuResources;

class MultiIndex2 {
   public:
    MultiIndex2(GpuResources* res, int dim, MemorySpace space);

    // numVecsTotal: NUM_CODEBOOKS * numCentroidsPerCodebook_
    static size_t calcMemorySpaceSize(
            int numVecsTotal,
            int dimPerCodebook,
            bool useFloat16);

    static std::unordered_map<AllocType, size_t> getAllocSizePerTypeInfo(
            int numVecsTotal,
            int dimPerCodebook,
            bool useFloat16);

    /// Whether or not this flat index primarily stores data in float16
    bool getUseFloat16() const;

    /// Returns the number of indexed vectors
    int getSize() const;

    /// Returns the number of centroids per codebook
    int getCodebookSize() const;

    /// Returns the multi-index of index pair
    int toMultiIndex(ushort2 indexPair) const;

    /// Returns the total dimension of vectors
    int getDim() const;

    /// Returns the total dimension of vectors
    int getSubDim() const;

    /// Returns the total number of codebooks
    int getNumCodebooks() const;

    /// Reserve storage that can contain this many vectors
    void reserve(int numVecsTotal, cudaStream_t stream);

    /// Returns a reference to our vectors currently in use
    Tensor<float, 2, true>& getVectorsFloat32Ref();

    void query(
            Tensor<float, 2, true>& subQueries,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<int2, 2, true>& outIndices,
            bool exactDistance);

    void query(
            Tensor<float, 2, true>& subQueries,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<ushort2, 2, true>& outIndices,
            bool exactDistance);

    void query(
            Tensor<float, 2, true>& subQueries,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            bool exactDistance);

    /// Compute residuals (NUM_CODEBOOKS * num of vecs)(dimPerCodebook_)
    void computeResidual(
            Tensor<float, 2, true>& vecs,
            Tensor<ushort2, 1, true>& listIds,
            Tensor<float, 2, true>& residuals);

    /// Compute residuals (NUM_CODEBOOKS * num of vecs)(dimPerCodebook_)
    void computeResidual(
            Tensor<float, 2, true>& vecs,
            Tensor<int2, 1, true>& listIds,
            Tensor<float, 2, true>& residuals);

    /** Add vectors to ourselves; the pointer passed can be on the host
     * or the device
     * @param data (numVecsTotal * dimPerCodebook_)
     * @param numVecsTotal NUM_CODEBOOKS * numCentroidsPerCodebook_
     * @param stream executing stream
     */
    void add(const float* data, int numVecsTotal, cudaStream_t stream);

    void reset();

   private:
    template <typename IndexT, typename IndexTVect2>
    void queryImpl(
            Tensor<float, 2, true>& subQueries,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<IndexTVect2, 2, true>& outIndices,
            bool exactDistance);

    template <typename IndexT>
    void queryImpl(
            Tensor<float, 2, true>& subQueries,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            bool exactDistance);

    GpuResources* resources_;

    MemorySpace space_;

    const int numCodebooks_;

    int dimPerCodebook_, numCentroidsPerCodebook_;

    /// The underlying expandable storage
    DeviceVector<char> rawData_;

    /// Vectors currently in rawData_
    /// (codebook c * subVec v)(subDim d)
    DeviceTensor<float, 2, true> vectors_;

    /// Precomputed L2 norms
    /// (codebook c * norm n)
    DeviceTensor<float, 1, true> norms_;
};

} // namespace gpu
} // namespace faiss
