// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <faiss/impl/IDSelector.h>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>

#include <cuvs/neighbors/ivf_sq.hpp>

#include <memory>
#include <vector>

#pragma GCC visibility push(default)
namespace faiss {
namespace gpu {

class CuvsIVFSQ : public IVFFlat {
   public:
    CuvsIVFSQ(
            GpuResources* resources,
            int dim,
            int nlist,
            faiss::MetricType metric,
            float metricArg,
            bool useResidual,
            faiss::ScalarQuantizer* scalarQ,
            bool interleavedLayout,
            IndicesOptions indicesOptions,
            MemorySpace space);

    ~CuvsIVFSQ() override;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(idx_t numVecs) override;

    /// Clear out the cuVS index
    void reset() override;

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    /// Find the approximate k nearest neighbors for `queries` against
    /// our database
    void search(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& queries,
            int nprobe,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            const IDSelector* sel = nullptr) override;

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

    /// For debugging purposes, return the list length of a particular
    /// list
    idx_t getListLength(idx_t listId) const override;

    /// Return the list indices of a particular list back to the CPU
    std::vector<idx_t> getListIndices(idx_t listId) const override;

    /// Return the encoded vectors of a particular list back to the CPU
    std::vector<uint8_t> getListVectorData(idx_t listId, bool gpuFormat)
            const override;

    /// Update our cuVS index with this quantizer instance; may be a CPU
    /// or GPU quantizer
    void updateQuantizer(Index* quantizer) override;

    /// Copy all inverted lists from a CPU representation to ourselves
    void copyInvertedListsFrom(const InvertedLists* ivf) override;

    /// Replace the cuVS index
    void setCuvsIndex(cuvs::neighbors::ivf_sq::index<uint8_t>&& idx);

    /// Copy the cuVS SQ range state back to a FAISS scalar quantizer
    void setFaissSQFromCuvs(faiss::ScalarQuantizer* sq) const;

    /// Reconstruct vectors by id from the cuVS list state
    void reconstruct_n(idx_t i0, idx_t n, float* out) override;

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

    void copyFaissSQToCuvs_();
    void copyCuvsSQToFaiss_(faiss::ScalarQuantizer* sq) const;
    void computeCenterNorms_();
    void recomputeListState_(const std::vector<uint32_t>& listSizes);
    idx_t getBitsetSizeForFiltering_() const;
    std::vector<float> getCentersHost_() const;

    /// Scalar quantizer owned by the public GpuIndexIVFScalarQuantizer wrapper
    faiss::ScalarQuantizer* faissSQ_;

    std::shared_ptr<cuvs::neighbors::ivf_sq::index<uint8_t>> cuvs_index{
            nullptr};
};

struct CuvsIVFSQCodePackerInterleaved : CodePacker {
    CuvsIVFSQCodePackerInterleaved(size_t list_size, uint32_t dim);
    CodePacker* clone() const final;
    void pack_1(const uint8_t* flat_code, size_t offset, uint8_t* block)
            const final;
    void unpack_1(const uint8_t* block, size_t offset, uint8_t* flat_code)
            const final;

   protected:
    uint32_t dim;
    uint32_t padded_dim;
};

} // namespace gpu
} // namespace faiss
#pragma GCC visibility pop
