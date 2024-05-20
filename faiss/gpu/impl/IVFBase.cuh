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
            int dim,
            idx_t nlist,
            faiss::MetricType metric,
            float metricArg,
            bool interleavedLayout,
            bool useResidual,
            IndicesOptions indicesOptions,
            MemorySpace space);

    virtual ~IVFBase();

    /// Reserve GPU memory in our inverted lists for this number of vectors
    virtual void reserveMemory(idx_t numVecs);

    /// Clear out all inverted lists, but retain the coarse quantizer
    /// and the product quantizer info
    virtual void reset();

    /// Return the number of dimensions we are indexing
    idx_t getDim() const;

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    /// Returns the number of inverted lists
    idx_t getNumLists() const;

    /// For debugging purposes, return the list length of a particular
    /// list
    virtual idx_t getListLength(idx_t listId) const;

    /// Return the list indices of a particular list back to the CPU
    virtual std::vector<idx_t> getListIndices(idx_t listId) const;

    /// Return the encoded vectors of a particular list back to the CPU
    virtual std::vector<uint8_t> getListVectorData(idx_t listId, bool gpuFormat)
            const;

    /// Copy all inverted lists from a CPU representation to ourselves
    virtual void copyInvertedListsFrom(const InvertedLists* ivf);

    /// Copy all inverted lists from ourselves to a CPU representation
    virtual void copyInvertedListsTo(InvertedLists* ivf);

    /// Update our coarse quantizer with this quantizer instance; may be a CPU
    /// or GPU quantizer
    virtual void updateQuantizer(Index* quantizer);

    /// Classify and encode/add vectors to our IVF lists.
    /// The input data must be on our current device.
    /// Returns the number of vectors successfully added. Vectors may
    /// not be able to be added because they contain NaNs.
    virtual idx_t addVectors(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& vecs,
            Tensor<idx_t, 1, true>& indices);

    /// Find the approximate k nearest neigbors for `queries` against
    /// our database
    virtual void search(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& queries,
            int nprobe,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices) = 0;

    /// Performs search when we are already given the IVF cells to look at
    /// (GpuIndexIVF::search_preassigned implementation)
    virtual void searchPreassigned(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& vecs,
            Tensor<float, 2, true>& ivfDistances,
            Tensor<idx_t, 2, true>& ivfAssignments,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            bool storePairs) = 0;

    /*  It is used to reconstruct a given number of vectors in an Inverted File
     * (IVF) index
     *  @param i0          index of the first vector to reconstruct
     *  @param n           number of vectors to reconstruct
     *  @param out         This is a pointer to a buffer where the reconstructed
     * vectors will be stored.
     */
    virtual void reconstruct_n(idx_t i0, idx_t n, float* out);

   protected:
    /// Adds a set of codes and indices to a list, with the
    /// representation coming from the CPU equivalent
    virtual void addEncodedVectorsToList_(
            idx_t listId,
            // resident on the host
            const void* codes,
            // resident on the host
            const idx_t* indices,
            idx_t numVecs);

    /// Performs search in a CPU or GPU coarse quantizer for IVF cells,
    /// returning residuals as well if necessary
    void searchCoarseQuantizer_(
            Index* coarseQuantizer,
            int nprobe,
            // guaranteed resident on device
            Tensor<float, 2, true>& vecs,
            // Output: the distances to the closest nprobe IVF cell centroids
            // for the query vectors
            // size (#vecs, nprobe)
            Tensor<float, 2, true>& distances,
            // Output: the closest nprobe IVF cells the query vectors lie in
            // size (#vecs, nprobe)
            Tensor<idx_t, 2, true>& indices,
            // optionally compute the residual relative to the IVF cell centroid
            // if passed
            // size (#vecs, nprobe, dim)
            Tensor<float, 3, true>* residuals,
            // optionally return the IVF cell centroids to which the input
            // vectors were assigned
            // size (#vecs, nprobe, dim)
            Tensor<float, 3, true>* centroids);

    /// Returns the number of bytes in which an IVF list containing numVecs
    /// vectors is encoded on the device. Note that due to padding this is not
    /// the same as the encoding size for a subset of vectors in an IVF list;
    /// this is the size for an entire IVF list
    virtual size_t getGpuVectorsEncodingSize_(idx_t numVecs) const = 0;
    virtual size_t getCpuVectorsEncodingSize_(idx_t numVecs) const = 0;

    /// Translate to our preferred GPU encoding
    virtual std::vector<uint8_t> translateCodesToGpu_(
            std::vector<uint8_t> codes,
            idx_t numVecs) const = 0;

    /// Translate from our preferred GPU encoding
    virtual std::vector<uint8_t> translateCodesFromGpu_(
            std::vector<uint8_t> codes,
            idx_t numVecs) const = 0;

    /// Append vectors to our on-device lists
    virtual void appendVectors_(
            Tensor<float, 2, true>& vecs,
            Tensor<float, 2, true>& ivfCentroidResiduals,
            Tensor<idx_t, 1, true>& indices,
            Tensor<idx_t, 1, true>& uniqueLists,
            Tensor<idx_t, 1, true>& vectorsByUniqueList,
            Tensor<idx_t, 1, true>& uniqueListVectorStart,
            Tensor<idx_t, 1, true>& uniqueListStartOffset,
            Tensor<idx_t, 1, true>& listIds,
            Tensor<idx_t, 1, true>& listOffset,
            cudaStream_t stream) = 0;

    /// Reclaim memory consumed on the device for our inverted lists
    /// `exact` means we trim exactly to the memory needed
    size_t reclaimMemory_(bool exact);

    /// Update all device-side list pointer and size information
    void updateDeviceListInfo_(cudaStream_t stream);

    /// For a set of list IDs, update device-side list pointer and size
    /// information
    void updateDeviceListInfo_(
            const std::vector<idx_t>& listIds,
            cudaStream_t stream);

    /// Shared function to copy indices from CPU to GPU
    void addIndicesFromCpu_(idx_t listId, const idx_t* indices, idx_t numVecs);

   protected:
    /// Collection of GPU resources that we use
    GpuResources* resources_;

    /// Metric type of the index
    faiss::MetricType metric_;

    /// Metric arg
    float metricArg_;

    /// Expected dimensionality of the vectors
    const int dim_;

    /// Number of inverted lists we maintain
    const idx_t numLists_;

    /// Do we need to also compute residuals when processing vectors?
    bool useResidual_;

    /// Coarse quantizer centroids available on GPU
    DeviceTensor<float, 2, true> ivfCentroids_;

    /// Whether or not our index uses an interleaved by kWarpSize layout:
    /// The default memory layout is [vector][PQ/SQ component]:
    /// (v0 d0) (v0 d1) ... (v0 dD-1) (v1 d0) (v1 d1) ...
    ///
    /// The interleaved by kWarpSize memory layout is:
    /// [vector / kWarpSize][PQ/SQ component][vector % kWarpSize] with padding:
    /// (v0 d0) (v1 d0) ... (v31 d0) (v0 d1) (v1 d1) ... (v31 dD-1) (v32 d0)
    /// (v33 d0) ... so the list length is always a multiple of num quantizers *
    /// kWarpSize
    bool interleavedLayout_;

    /// How are user indices stored on the GPU?
    const IndicesOptions indicesOptions_;

    /// What memory space our inverted list storage is in
    const MemorySpace space_;

    /// Device representation of all inverted list data
    /// id -> data
    DeviceVector<void*> deviceListDataPointers_;

    /// Device representation of all inverted list index pointers
    /// id -> data
    DeviceVector<void*> deviceListIndexPointers_;

    /// Device representation of all inverted list lengths
    /// id -> length in number of vectors
    DeviceVector<idx_t> deviceListLengths_;

    /// Maximum list length seen
    idx_t maxListLength_;

    struct DeviceIVFList {
        DeviceIVFList(GpuResources* res, const AllocInfo& info);

        /// The on-device memory for this particular IVF list
        DeviceVector<uint8_t> data;

        /// The number of vectors encoded in this list, which may be unrelated
        /// to the above allocated data size
        idx_t numVecs;
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
    std::vector<std::vector<idx_t>> listOffsetToUserIndex_;
};

} // namespace gpu
} // namespace faiss
