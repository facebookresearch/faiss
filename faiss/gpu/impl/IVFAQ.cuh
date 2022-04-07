/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/gpu/utils/Float16.cuh>

namespace faiss {
namespace gpu {

/// Implementing class for IVFAQ on the GPU
class IVFAQ : public IVFBase {
   public:
    IVFAQ(GpuResources* resources,
          faiss::MetricType metric,
          float metricArg,
          /// We do not own this reference
          FlatIndex* quantizer,
          int numSubQuantizers,
          int bitsPerSubQuantizer,
          bool useFloat16LookupTables, // wxx
          bool useMMCodeDistance,
          bool interleavedLayout,
          float* aqCentroidData,
          IndicesOptions indicesOptions, // wxx
          MemorySpace space);            // wxx
    /// Returns true if we support AQ in this size
    static bool isSupportedAQCodeLength(int size);

    ~IVFAQ() override;

    /// Enable or disable pre-computed codes
    /// void setPrecomputedCodes(bool enable);

    /// Find the approximate k nearest neigbors for `queries` against
    /// our database
    void query(
            Tensor<float, 2, true>& queries,
            int nprobe,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<Index::idx_t, 2, true>& outIndices);

    /// Returns our set of sub-quantizers of the form
    /// (sub q)(code id)(sub dim)
    Tensor<float, 3, true> getAQCentroids();

   protected:
    /// Returns
    size_t getGpuVectorsEncodingSize_(int numVecs) const override;
    size_t getCpuVectorsEncodingSize_(int numVecs) const override;

    /// Translate to our preferred GPU encoding
    std::vector<uint8_t> translateCodesToGpu_(
            std::vector<uint8_t> codes,
            size_t numVecs) const override;

    /// Translate from our preferred GPU encoding
    std::vector<uint8_t> translateCodesFromGpu_(
            std::vector<uint8_t> codes,
            size_t numVecs) const override;

    /// Encode the vectors that we're adding and append to our IVF lists
    void appendVectors_(
            Tensor<float, 2, true>& vecs,
            Tensor<Index::idx_t, 1, true>& indices,
            Tensor<int, 1, true>& uniqueLists,
            Tensor<int, 1, true>& vectorsByUniqueList,
            Tensor<int, 1, true>& uniqueListVectorStart,
            Tensor<int, 1, true>& uniqueListStartOffset,
            Tensor<int, 1, true>& listIds,
            Tensor<int, 1, true>& listOffset,
            cudaStream_t stream) override;

    /// Calculate precomputed residual distance information
    void precomputeCodes_();

    /// Runs kernels for scanning inverted lists with precomputed codes
    void runPQPrecomputedCodes_(
            Tensor<float, 2, true>& queries,
            DeviceTensor<float, 2, true>& coarseDistances,
            DeviceTensor<int, 2, true>& coarseIndices,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<Index::idx_t, 2, true>& outIndices);

    /// Runs kernels for scanning inverted lists without precomputed codes
    void runPQNoPrecomputedCodes_(
            Tensor<float, 2, true>& queries,
            DeviceTensor<float, 2, true>& coarseDistances,
            DeviceTensor<int, 2, true>& coarseIndices,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<Index::idx_t, 2, true>& outIndices);

    /// Runs kernels for scanning inverted lists without precomputed codes (for
    /// different coarse centroid type)
    template <typename CentroidT>
    void runPQNoPrecomputedCodesT_(
            Tensor<float, 2, true>& queries,
            DeviceTensor<float, 2, true>& coarseDistances,
            DeviceTensor<int, 2, true>& coarseIndices,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<Index::idx_t, 2, true>& outIndices);

   private:
    ///
    const bool useFloat16LookupTables_;

    /// For usage without precomputed codes,
    /// do we use general-purpose MM code distance
    const bool useMMCodeDistance_;

    /// On the GPU, we prefer different AQ centroid data layouts for
    /// different purposes.
    /// (sub q)(sub dim)(code id)

    /// Are precomputed codes enabled? (additional factoring and
    /// precomputation of the residual distance, to reduce query-time work)
    bool precomputedCodes_;

    /// Precomputed term 2 in float form
    /// (centroid id)(sub q)(code id)
    DeviceTensor<float, 3, true> precomputedCode_;

    /// Precomputed term 2 in half form
    DeviceTensor<half, 3, true> precomputedCodeHalf_;
};

} // namespace gpu
} // namespace faiss