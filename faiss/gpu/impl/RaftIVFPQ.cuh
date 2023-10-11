// /**
//  * Copyright (c) Facebook, Inc. and its affiliates.
//  *
//  * This source code is licensed under the MIT license found in the
//  * LICENSE file in the root directory of this source tree.
//  */

// #pragma once

#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/gpu/impl/IVFPQ.cuh>

#include <faiss/impl/CodePacker.h>

#include <raft/neighbors/ivf_pq.cuh>

#include <optional>

namespace faiss {
namespace gpu {
/// Implementing class for IVFPQ on the GPU
class RaftIVFPQ : public IVFPQ {
   public:
    RaftIVFPQ(
            GpuResources* resources,
            int dim,
            idx_t nlist,
            faiss::MetricType metric,
            float metricArg,
            int numSubQuantizers,
            int bitsPerSubQuantizer,
            bool useFloat16LookupTables,
            bool useMMCodeDistance,
            bool interleavedLayout,
            float* pqCentroidData,
            IndicesOptions indicesOptions,
            MemorySpace space);

    /// Clear out all inverted lists, but retain the coarse quantizer
    /// and the product quantizer info
    void reset() override;

    /// Returns true if we support PQ in this size
    static bool isSupportedPQCodeLength(int size);

    ~RaftIVFPQ() override;

    /// Enable or disable pre-computed codes. The quantizer is needed to gather
    /// the IVF centroids for use
    void setPrecomputedCodes(Index* coarseQuantizer, bool enable);

    /// Returns our set of sub-quantizers of the form
    /// (sub q)(code id)(sub dim)
    Tensor<float, 3, true> getPQCentroids();

    /// Find the approximate k nearest neigbors for `queries` against
    /// our database
    void search(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& queries,
            int nprobe,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices) override;

    /// Performs search when we are already given the IVF cells to look at
    /// (GpuIndexIVF::search_preassigned implementation)
    void searchPreassigned(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& vecs,
            Tensor<float, 2, true>& ivfDistances,
            Tensor<idx_t, 2, true>& ivfAssignments,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            bool storePairs) override;

    /// Return the encoded vectors of a particular list back to the CPU
    std::vector<uint8_t> getListVectorData(idx_t listId, bool gpuFormat)
            const override;

    void updateQuantizer(Index* quantizer) override;

    /// Copy all inverted lists from a CPU representation to ourselves
    void copyInvertedListsFrom(const InvertedLists* ivf) override;

    void setRaftIndex(std::optional<raft::neighbors::ivf_pq::index<idx_t>>& idx);

    /// Classify and encode/add vectors to our IVF lists.
    /// The input data must be on our current device.
    /// Returns the number of vectors successfully added. Vectors may
    /// not be able to be added because they contain NaNs.
    idx_t addVectors(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& vecs,
            Tensor<idx_t, 1, true>& indices) override;

   protected:
    /// Returns the encoding size for a PQ-encoded IVF list
    size_t getGpuVectorsEncodingSize_(idx_t numVecs) const override;
    size_t getCpuVectorsEncodingSize_(idx_t numVecs) const override;

    /// Encode the vectors that we're adding and append to our IVF lists
    void appendVectors_(
            Tensor<float, 2, true>& vecs,
            Tensor<float, 2, true>& ivfCentroidResiduals,
            Tensor<idx_t, 1, true>& indices,
            Tensor<idx_t, 1, true>& uniqueLists,
            Tensor<idx_t, 1, true>& vectorsByUniqueList,
            Tensor<idx_t, 1, true>& uniqueListVectorStart,
            Tensor<idx_t, 1, true>& uniqueListStartOffset,
            Tensor<idx_t, 1, true>& listIds,
            Tensor<idx_t, 1, true>& listOffset,
            cudaStream_t stream) override;

    /// Sets the current product quantizer centroids; the data can be
    /// resident on either the host or the device. It will be transposed
    /// into our preferred data layout
    /// Data must be a row-major, 3-d array of size
    /// (numSubQuantizers, numSubQuantizerCodes, dim / numSubQuantizers)
    void setPQCentroids_(float* data);

    /// Calculate precomputed residual distance information
    void precomputeCodes_(Index* quantizer);

    /// Runs kernels for scanning inverted lists with precomputed codes
    void runPQPrecomputedCodes_(
            Tensor<float, 2, true>& queries,
            Tensor<float, 2, true>& coarseDistances,
            Tensor<idx_t, 2, true>& coarseIndices,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices);

    /// Runs kernels for scanning inverted lists without precomputed codes
    void runPQNoPrecomputedCodes_(
            Tensor<float, 2, true>& queries,
            Tensor<float, 2, true>& coarseDistances,
            Tensor<idx_t, 2, true>& coarseIndices,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices);

   private:
    /// On the GPU, we prefer different PQ centroid data layouts for
    /// different purposes.
    ///
    /// (sub q)(sub dim)(code id)
    DeviceTensor<float, 3, true> pqCentroidsInnermostCode_;

    /// (sub q)(code id)(sub dim)
    DeviceTensor<float, 3, true> pqCentroidsMiddleCode_;

    /// Are precomputed codes enabled? (additional factoring and
    /// precomputation of the residual distance, to reduce query-time work)
    bool precomputedCodes_;

    /// Precomputed term 2 in float form
    /// (centroid id)(sub q)(code id)
    DeviceTensor<float, 3, true> precomputedCode_;

    /// Precomputed term 2 in half form
    DeviceTensor<half, 3, true> precomputedCodeHalf_;

    std::optional<raft::neighbors::ivf_pq::index<idx_t>>
        raft_knn_index{std::nullopt};
};

struct RaftIVFPQCodePackerInterleaved : CodePacker {
    RaftIVFPQCodePackerInterleaved(
            size_t list_size,
            int numSubQuantizers,
            int bitsPerSubQuantizer);
    void pack_1(const uint8_t* flat_code, size_t offset, uint8_t* block)
            const final;
    void unpack_1(const uint8_t* block, size_t offset, uint8_t* flat_code)
            const final;

   protected:
    int bitsPerSubQuantizer_;
    int numSubQuantizers_;
};

} // namespace gpu
} // namespace faiss
