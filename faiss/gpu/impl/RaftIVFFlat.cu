/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/neighbors/ivf_flat.cuh>

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/RaftIVFFlat.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <unordered_map>

namespace faiss {
namespace gpu {

RaftIVFFlat::RaftIVFFlat(
        GpuResources* res,
        int dim,
        int nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        faiss::ScalarQuantizer* scalarQ,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFFlat(res,
                  dim,
                  nlist,
                  metric,
                  metricArg,
                  useResidual,
                  scalarQ,
                  interleavedLayout,
                  indicesOptions,
                  space){}

RaftIVFFlat::~RaftIVFFlat() {}


/// Find the approximate k nearest neighbors for `queries` against
/// our database
void RaftIVFFlat::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices) {
    printf("Inside RaftIVFFlat search()\n");

    // TODO: We probably don't want to ignore the coarse quantizer here...

    std::uint32_t n = queries.getSize(0);
    std::uint32_t cols = queries.getSize(1);
    std::uint32_t k_ = k;

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::handle_t &raft_handle = resources_->getRaftHandleCurrentDevice();
    raft::neighbors::ivf_flat::search_params pams;
    pams.n_probes = nprobe;

    auto queries_view = raft::make_device_matrix_view<const float>(queries.data(), n, cols);
    auto out_inds_view = raft::make_device_matrix_view<Index::idx_t>(outIndices.data(), n, k_);
    auto out_dists_view = raft::make_device_matrix_view<float>(outDistances.data(), n, k_);
    raft::neighbors::ivf_flat::search<float, faiss::Index::idx_t>(
            raft_handle, *raft_knn_index, queries_view,
            out_inds_view, out_dists_view, pams, k_);

    raft_handle.sync_stream();
}

/// Classify and encode/add vectors to our IVF lists.
/// The input data must be on our current device.
/// Returns the number of vectors successfully added. Vectors may
/// not be able to be added because they contain NaNs.
int RaftIVFFlat::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<Index::idx_t, 1, true>& indices) {
    printf("Inside RaftIVFFlat addVectors()\n");

    auto vecs_view = raft::make_device_matrix_view<const float, Index::idx_t>(vecs.data(), vecs.getSize(0), dim_);
    auto inds_view = raft::make_device_vector_view<const Index::idx_t, Index::idx_t>(indices.data(), (Index::idx_t )indices.getSize(0));

    const raft::handle_t &raft_handle = resources_->getRaftHandleCurrentDevice();

    // TODO: We probably don't want to ignore the coarse quantizer here
    raft_knn_index.emplace(raft::neighbors::ivf_flat::extend(
            raft_handle,
            raft_knn_index.value(),
            vecs_view,
            std::make_optional<raft::device_vector_view<const Index::idx_t, Index::idx_t>>(inds_view)));
    return vecs.getSize(0);
}

void RaftIVFFlat::reset() {
    printf("Inside RaftIVFFlat reset()\n");
    raft_knn_index.reset();
}

int RaftIVFFlat::getListLength(int listId) const {
    printf("Inside RaftIVFFlat getListLength\n");

    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::handle_t &raft_handle = resources_->getRaftHandleCurrentDevice();

    uint32_t size;
    raft::copy(&size, raft_knn_index.value().list_sizes().data_handle() + listId,
               1, raft_handle.get_stream());
    raft_handle.sync_stream();
    return int(size);
}

/// Return the list indices of a par
/// ticular list back to the CPU
std::vector<Index::idx_t> RaftIVFFlat::getListIndices(int listId) const {

    printf("Inside RaftIVFFlat getListIndices\n");

    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::handle_t &raft_handle = resources_->getRaftHandleCurrentDevice();

    Index::idx_t offset;
    uint32_t size;

    raft::copy(&offset, raft_knn_index.value().list_offsets().data_handle() + listId, 1, raft_handle.get_stream());
    raft::copy(&size, raft_knn_index.value().list_sizes().data_handle() + listId, 1, raft_handle.get_stream());
    raft_handle.sync_stream();

    std::vector<Index::idx_t> vec(size);
    raft::copy(
            vec.data(),
            raft_knn_index.value().indices().data_handle() + offset,
            size,
            raft_handle.get_stream());
    return vec;
}

/// Return the encoded vectors of a particular list back to the CPU
std::vector<uint8_t> RaftIVFFlat::getListVectorData(int listId, bool gpuFormat) const {

    printf("Inside RaftIVFFlat getListVectorData\n");

    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::handle_t &raft_handle = resources_->getRaftHandleCurrentDevice();

    std::cout << "Calling getListVectorData for " << listId << std::endl;

    using elem_t = decltype(raft_knn_index.value().data())::element_type;
    size_t dim = raft_knn_index.value().dim();
    Index::idx_t offsets[2];
    raft::copy(offsets, raft_knn_index.value().list_offsets().data_handle() + listId, 2, raft_handle.get_stream());

    raft_handle.sync_stream();
    size_t byte_offset = offsets[0] * sizeof(elem_t) * dim;
    // the interleaved block can be slightly larger than the list size (it's
    // rounded up)
    size_t byte_size = size_t(offsets[1]) *
                       sizeof(elem_t) * dim -
                       byte_offset;
    std::vector<uint8_t> vec(byte_size);
    raft::copy(
            vec.data(),
            reinterpret_cast<const uint8_t*>(raft_knn_index.value().data().data_handle()) +
            byte_offset,
            byte_size,
            raft_handle.get_stream());
    return vec;
}

/// Performs search when we are already given the IVF cells to look at
/// (GpuIndexIVF::search_preassigned implementation)
void RaftIVFFlat::searchPreassigned(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfDistances,
        Tensor<Index::idx_t, 2, true>& ivfAssignments,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices,
        bool storePairs) {
    printf("Inside RaftIVFFlat searchPreassigned\n");

    // TODO: Fill this in!
}

/// Copy all inverted lists from a CPU representation to ourselves
void RaftIVFFlat::copyInvertedListsFrom(const InvertedLists* ivf) {
    printf("Inside RaftIVFFlat copyInvertedListsFrom\n");

    ivf->print_stats();

    // TODO: Need to replicate copyInvertedListsFrom() in IVFBase.cu
    // but populate a RAFT index.
}

/// Copy all inverted lists from ourselves to a CPU representation
void RaftIVFFlat::copyInvertedListsTo(InvertedLists* ivf) {
    printf("Inside RaftIVFFlat copyInvertedListsTo\n");

    // TODO: Need to replicate copyInvertedListsTo() in IVFBase.cu
}


} // namespace gpu
} // namespace faiss
