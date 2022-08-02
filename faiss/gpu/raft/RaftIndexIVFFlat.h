/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>

#include <raft/core/handle.hpp>
#include <raft/spatial/knn/ivf_flat_types.hpp>

#include <memory>

namespace faiss {
struct IndexIVFFlat;
}

namespace faiss {
namespace gpu {

class RaftIVFFlat;
class GpuIndexFlat;

/// Wrapper around the GPU implementation that looks like
/// faiss::gpu::GpuIndexIVFFlat
class RaftIndexIVFFlat : public GpuIndexIVFFlat {
   public:
    /// Construct from a pre-existing faiss::IndexIVFFlat instance, copying
    /// data over to the given GPU, if the input index is trained.
    RaftIndexIVFFlat(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFFlat* index,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    /// Constructs a new instance with an empty flat quantizer; the user
    /// provides the number of lists desired.
    RaftIndexIVFFlat(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            faiss::MetricType metric,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    ~RaftIndexIVFFlat() override;

    /// Clears out all inverted lists, but retains the coarse centroid
    /// information
    void reset() override;

    /// Trains the coarse quantizer based on the given vector data
    void train(Index::idx_t n, const float* x) override;

    /// Returns the number of vectors present in a particular inverted list
    int getListLength(int listId) const override;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(size_t numVecs);

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory();

    void copyFrom(const faiss::IndexIVFFlat* index);

    /// Return the encoded vector data contained in a particular inverted list,
    /// for debugging purposes.
    /// If gpuFormat is true, the data is returned as it is encoded in the
    /// GPU-side representation.
    /// Otherwise, it is converted to the CPU format.
    /// compliant format, while the native GPU format may differ.
    std::vector<uint8_t> getListVectorData(int listId, bool gpuFormat = false)
    const override;

    /// Return the vector indices contained in a particular inverted list, for
    /// debugging purposes.
    std::vector<Index::idx_t> getListIndices(int listId) const override;

   protected:
    /// Called from GpuIndex for add/add_with_ids
    void addImpl_(int n, const float* x, const Index::idx_t* ids) override;


    /// Called from GpuIndex for search
    void searchImpl_(
            int n,
            const float* x,
            int k,
            float* distances,
            Index::idx_t* labels) const override;

    void rebuildRaftIndex(const float* x, Index::idx_t n_rows);

    const raft::handle_t raft_handle;
    std::optional<raft::spatial::knn::ivf_flat::index<float, Index::idx_t>> raft_knn_index{std::nullopt};
};

} // namespace gpu
} // namespace faiss
