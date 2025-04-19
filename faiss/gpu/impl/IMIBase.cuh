/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <thrust/device_vector.h>
#include <faiss/gpu/impl/MultiIndex2.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <memory>
#include <unordered_map>
#include <vector>

namespace faiss {
struct InvertedLists;
}

namespace faiss {
namespace gpu {

class GpuResources;

class IMIBase {
   public:
    IMIBase(GpuResources* resources,
            /// We do not own this reference
            MultiIndex2* quantizer,
            bool interleavedLayout,
            IndicesOptions indicesOptions,
            MemorySpace space);

    virtual ~IMIBase();

    /// Reserve GPU memory in our inverted lists for the exepected number of
    /// adds per each inverted list
    void reserveMemory(
            const std::unordered_map<int, int>* expectedNumAddsPerList,
            int numVecs);

    void reset();

    /// Clear out all inverted lists, but retain the coarse quantizer
    /// and the product quantizer info
    void reset(int numVecs);

    int getMaxListLength() const;

    /// Return the number of dimensions we are indexing
    int getDim() const;

    /// Returns the number of inverted lists
    size_t getNumLists() const;

    /// For debugging purposes, return the list length of a particular
    /// list
    int getListLength(int listId);

    /// For debugging purposes, return the list offset of a particular
    /// list
    unsigned int getListOffset(int listId);

    /// For debugging purposes, return the length of all lists
    int getAllListsLength();

    /// Return the list indices of a particular list back to the CPU
    std::vector<idx_t> getListIndices(int listId);

    /// Return the encoded vectors of a particular list back to the CPU
    std::vector<uint8_t> getListVectorData(int listId, bool gpuFormat);

    /// Copy all inverted lists from a CPU representation to ourselves
    void copyInvertedListsFrom(InvertedLists* ivf);

    /// Copy all inverted lists from ourselves to a CPU representation
    void copyInvertedListsTo(InvertedLists* ivf);

    /// Classify and encode/add vectors to our IVF lists.
    /// The input data must be on our current device.
    /// Returns the number of vectors successfully added. Vectors may
    /// not be able to be added because they contain NaNs.
    int addVectors(
            Tensor<float, 2, true>& vecs,
            Tensor<idx_t, 1, true>& indices);

   protected:
    /// Returns the number of bytes in which an IVF list containing numVecs
    /// vectors is encoded on the device. Note that due to padding this is not
    /// the same as the encoding size for a subset of vectors in an IVF list;
    /// this is the size for an entire IVF list
    virtual size_t getGpuVectorsEncodingSize_(int numVecs) const = 0;
    virtual size_t getCpuVectorsEncodingSize_(int numVecs) const = 0;

    /// Translate to our preferred GPU encoding
    virtual std::vector<uint8_t> translateCodesToGpu_(
            std::vector<uint8_t> codes,
            size_t numVecs) const = 0;

    /// Translate from our preferred GPU encoding
    virtual std::vector<uint8_t> translateCodesFromGpu_(
            std::vector<uint8_t> codes,
            size_t numVecs) const = 0;

    /// Append vectors to our on-device lists
    virtual void appendVectors_(
            Tensor<float, 2, true>& vecs,
            Tensor<idx_t, 1, true>& indices,
            Tensor<int, 1, true>& uniqueLists,
            Tensor<int, 1, true>& vectorsByUniqueList,
            Tensor<int, 1, true>& uniqueListVectorStart,
            Tensor<int, 1, true>& uniqueListStartOffset,
            Tensor<ushort2, 1, true>& listIds,
            Tensor<int, 1, true>& listOffset,
            cudaStream_t stream) = 0;

    /// Reclaim memory consumed on the device for our inverted lists
    /// `exact` means we trim exactly to the memory needed
    size_t reclaimMemory_(bool exact);

   protected:
    /// Collection of GPU resources that we use
    GpuResources* resources_;

    /// Quantizer object
    MultiIndex2* quantizer_;

    /// Expected dimensionality of the vectors
    const int dim_;

    /// Number of inverted lists we maintain
    const int numLists_;

    /// Whether or not our index uses an interleaved by 32 layout:
    /// The default memory layout is [vector][PQ/SQ component]:
    /// (v0 d0) (v0 d1) ... (v0 dD-1) (v1 d0) (v1 d1) ...
    ///
    /// The interleaved by 32 memory layout is:
    /// [vector / 32][PQ/SQ component][vector % 32] with padding:
    /// (v0 d0) (v1 d0) ... (v31 d0) (v0 d1) (v1 d1) ... (v31 dD-1) (v32 d0)
    /// (v33 d0) ... so the list length is always a multiple of num quantizers *
    /// 32
    bool interleavedLayout_;

    /// How are user indices stored on the GPU?
    const IndicesOptions indicesOptions_;

    /// What memory space our inverted list storage is in
    const MemorySpace space_;

    DeviceVector<uint8_t> deviceListData_;
    DeviceVector<uint8_t> deviceListIndices_;
    DeviceVector<unsigned int> deviceListOffsets_;
    std::unique_ptr<std::vector<int>> currentListLengths_;

    /// Maximum list length seen
    int maxListLength_;

    /// If we are storing indices on the CPU (indicesOptions_ is
    /// INDICES_CPU), then this maintains a CPU-side map of what
    /// (inverted list id, offset) maps to which user index
    std::vector<std::vector<idx_t>> listOffsetToUserIndex_;

    bool isMemoryReserved_;

    int numVecs_;
};

} // namespace gpu
} // namespace faiss
