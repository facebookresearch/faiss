/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/impl/IVFBase.cuh>

namespace faiss {
namespace gpu {

class IVFFlat : public IVFBase {
   public:
    /// Construct from a quantizer that has elemen
    IVFFlat(GpuResources* resources,
            /// We do not own this reference
            FlatIndex* quantizer,
            faiss::MetricType metric,
            float metricArg,
            bool useResidual,
            /// Optional ScalarQuantizer
            faiss::ScalarQuantizer* scalarQ,
            bool interleavedLayout,
            IndicesOptions indicesOptions,
            MemorySpace space);

    ~IVFFlat() override;

    /// Find the approximate k nearest neigbors for `queries` against
    /// our database
    void query(
            Tensor<float, 2, true>& queries,
            int nprobe,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<Index::idx_t, 2, true>& outIndices);

   protected:
    /// Returns the number of bytes in which an IVF list containing numVecs
    /// vectors is encoded on the device. Note that due to padding this is not
    /// the same as the encoding size for a subset of vectors in an IVF list;
    /// this is the size for an entire IVF list
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

   protected:
    /// Do we encode the residual from a coarse quantizer or not?
    bool useResidual_;

    /// Scalar quantizer for encoded vectors, if any
    std::unique_ptr<GpuScalarQuantizer> scalarQ_;
};

} // namespace gpu
} // namespace faiss
