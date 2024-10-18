// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <faiss/impl/CodePacker.h>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>

#include <raft/neighbors/ivf_flat.cuh>

#include <optional>

#pragma GCC visibility push(default)
namespace faiss {
namespace gpu {

class RaftIVFFlat : public IVFFlat {
   public:
    RaftIVFFlat(
            GpuResources* resources,
            int dim,
            int nlist,
            faiss::MetricType metric,
            float metricArg,
            bool useResidual,
            /// Optional ScalarQuantizer
            faiss::ScalarQuantizer* scalarQ,
            bool interleavedLayout,
            IndicesOptions indicesOptions,
            MemorySpace space);

    ~RaftIVFFlat() override;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(idx_t numVecs) override;

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

    /// Classify and encode/add vectors to our IVF lists.
    /// The input data must be on our current device.
    /// Returns the number of vectors successfully added. Vectors may
    /// not be able to be added because they contain NaNs.
    idx_t addVectors(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& vecs,
            Tensor<idx_t, 1, true>& indices) override;

    /// Clear out the Raft index
    void reset() override;

    /// For debugging purposes, return the list length of a particular
    /// list
    idx_t getListLength(idx_t listId) const override;

    /// Return the list indices of a particular list back to the CPU
    std::vector<idx_t> getListIndices(idx_t listId) const override;

    /// Return the encoded vectors of a particular list back to the CPU
    std::vector<uint8_t> getListVectorData(idx_t listId, bool gpuFormat)
            const override;

    /// Update our Raft index with this quantizer instance; may be a CPU
    /// or GPU quantizer
    void updateQuantizer(Index* quantizer) override;

    /// Copy all inverted lists from a CPU representation to ourselves
    void copyInvertedListsFrom(const InvertedLists* ivf) override;

    /// Replace the Raft index
    void setRaftIndex(raft::neighbors::ivf_flat::index<float, idx_t>&& idx);

   private:
    /// Adds a set of codes and indices to a list, with the representation
    /// coming from the CPU equivalent
    void addEncodedVectorsToList_(
            idx_t listId,
            // resident on the host
            const void* codes,
            // resident on the host
            const idx_t* indices,
            idx_t numVecs) override;

    /// Returns the number of bytes in which an IVF list containing numVecs
    /// vectors is encoded on the device. Note that due to padding this is not
    /// the same as the encoding size for a subset of vectors in an IVF list;
    /// this is the size for an entire IVF list
    size_t getGpuVectorsEncodingSize_(idx_t numVecs) const override;

    std::optional<raft::neighbors::ivf_flat::index<float, idx_t>>
            raft_knn_index{std::nullopt};
};

struct RaftIVFFlatCodePackerInterleaved : CodePacker {
    RaftIVFFlatCodePackerInterleaved(
            size_t list_size,
            uint32_t dim,
            uint32_t chuk_size);
    void pack_1(const uint8_t* flat_code, size_t offset, uint8_t* block)
            const final;
    void unpack_1(const uint8_t* block, size_t offset, uint8_t* flat_code)
            const final;

   protected:
    uint32_t chunk_size;
    uint32_t dim;
};

} // namespace gpu
} // namespace faiss
#pragma GCC visibility pop
