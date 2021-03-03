/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <thrust/device_vector.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <memory>
#include <vector>

namespace faiss {
struct InvertedLists;
}

namespace faiss {
namespace gpu {

class GpuResources;
class FlatIndex;

/// Base inverted list functionality for IVFFlat and IVFPQ
class IVFBase {
   public:
    IVFBase(GpuResources* resources,
            faiss::MetricType metric,
            float metricArg,
            /// We do not own this reference
            FlatIndex* quantizer,
            bool interleavedLayout,
            IndicesOptions indicesOptions,
            MemorySpace space);

    virtual ~IVFBase();

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(size_t numVecs);

    /// Clear out all inverted lists, but retain the coarse quantizer
    /// and the product quantizer info
    void reset();

    /// Return the number of dimensions we are indexing
    int getDim() const;

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory();

    /// Returns the number of inverted lists
    size_t getNumLists() const;

    /// For debugging purposes, return the list length of a particular
    /// list
    int getListLength(int listId) const;

    /// Return the list indices of a particular list back to the CPU
    std::vector<Index::idx_t> getListIndices(int listId) const;

    /// Return the encoded vectors of a particular list back to the CPU
    std::vector<uint8_t> getListVectorData(int listId, bool gpuFormat) const;

    /// Copy all inverted lists from a CPU representation to ourselves
    void copyInvertedListsFrom(const InvertedLists* ivf);

    /// Copy all inverted lists from ourselves to a CPU representation
    void copyInvertedListsTo(InvertedLists* ivf);

    /// Classify and encode/add vectors to our IVF lists.
    /// The input data must be on our current device.
    /// Returns the number of vectors successfully added. Vectors may
    /// not be able to be added because they contain NaNs.
    int addVectors(
            Tensor<float, 2, true>& vecs,
            Tensor<Index::idx_t, 1, true>& indices);

   protected:
    /// Adds a set of codes and indices to a list, with the representation
    /// coming from the CPU equivalent
    void addEncodedVectorsToList_(
            int listId,
            // resident on the host
            const void* codes,
            // resident on the host
            const Index::idx_t* indices,
            size_t numVecs);

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
            Tensor<Index::idx_t, 1, true>& indices,
            Tensor<int, 1, true>& uniqueLists,
            Tensor<int, 1, true>& vectorsByUniqueList,
            Tensor<int, 1, true>& uniqueListVectorStart,
            Tensor<int, 1, true>& uniqueListStartOffset,
            Tensor<int, 1, true>& listIds,
            Tensor<int, 1, true>& listOffset,
            cudaStream_t stream) = 0;

    /// Reclaim memory consumed on the device for our inverted lists
    /// `exact` means we trim exactly to the memory needed
    size_t reclaimMemory_(bool exact);

    /// Update all device-side list pointer and size information
    void updateDeviceListInfo_(cudaStream_t stream);

    /// For a set of list IDs, update device-side list pointer and size
    /// information
    void updateDeviceListInfo_(
            const std::vector<int>& listIds,
            cudaStream_t stream);

    /// Shared function to copy indices from CPU to GPU
    void addIndicesFromCpu_(
            int listId,
            const Index::idx_t* indices,
            size_t numVecs);

   protected:
    /// Collection of GPU resources that we use
    GpuResources* resources_;

    /// Metric type of the index
    faiss::MetricType metric_;

    /// Metric arg
    float metricArg_;

    /// Quantizer object
    FlatIndex* quantizer_;

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

    /// Device representation of all inverted list data
    /// id -> data
    thrust::device_vector<void*> deviceListDataPointers_;

    /// Device representation of all inverted list index pointers
    /// id -> data
    thrust::device_vector<void*> deviceListIndexPointers_;

    /// Device representation of all inverted list lengths
    /// id -> length in number of vectors
    thrust::device_vector<int> deviceListLengths_;

    /// Maximum list length seen
    int maxListLength_;

    struct DeviceIVFList {
        DeviceIVFList(GpuResources* res, const AllocInfo& info);

        /// The on-device memory for this particular IVF list
        DeviceVector<uint8_t> data;

        /// The number of vectors encoded in this list, which may be unrelated
        /// to the above allocated data size
        int numVecs;
    };

    /// Device memory for each separate list, as managed by the host.
    /// Device memory as stored in DeviceVector is stored as unique_ptr
    /// since deviceList*Pointers_ must remain valid despite
    /// resizing (and potential re-allocation) of deviceList*_
    std::vector<std::unique_ptr<DeviceIVFList>> deviceListData_;
    std::vector<std::unique_ptr<DeviceIVFList>> deviceListIndices_;

    /// If we are storing indices on the CPU (indicesOptions_ is
    /// INDICES_CPU), then this maintains a CPU-side map of what
    /// (inverted list id, offset) maps to which user index
    std::vector<std::vector<Index::idx_t>> listOffsetToUserIndex_;
};

} // namespace gpu
} // namespace faiss
