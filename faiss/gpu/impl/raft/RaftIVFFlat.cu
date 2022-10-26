/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/spatial/knn/ivf_flat.cuh>

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/raft/RaftIVFFlat.cuh>
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


/// Find the approximate k nearest neigbors for `queries` against
/// our database
void RaftIVFFlat::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices) {

    // TODO: We probably don't want to ignore the coarse quantizer here...

    std::uint32_t n = queries.getSize(0);
    std::uint32_t cols = queries.getSize(1);
    std::uint32_t k_ = k;

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::handle_t &raft_handle = resources_->getRaftHandleCurrentDevice();
    raft::spatial::knn::ivf_flat::search_params pams;
    pams.n_probes = nprobe;

    auto queries_view = raft::make_device_matrix_view<const float>(queries.data(), n, cols);
    auto out_inds_view = raft::make_device_matrix_view<Index::idx_t>(outIndices.data(), n, k_);
    auto out_dists_view = raft::make_device_matrix_view<float>(outDistances.data(), n, k_);
    raft::spatial::knn::ivf_flat::search<float, faiss::Index::idx_t>(
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

    auto vecs_view = raft::make_device_matrix_view<const float, Index::idx_t>(vecs.data(), vecs.getSize(0), dim_);
    auto inds_view = raft::make_device_vector_view<const Index::idx_t, Index::idx_t>(indices.data(), (Index::idx_t )indices.getSize(0));

    const raft::handle_t &raft_handle = resources_->getRaftHandleCurrentDevice();

    // TODO: We probably don't want to ignore the coarse quantizer here
    raft_knn_index.emplace(raft::neighbors::ivf_flat::extend(
            raft_handle,
            raft_knn_index.value(),
            vecs_view,
            std::make_optional<raft::device_vector_view<const Index::idx_t, Index::idx_t>>(inds_view)));
}


} // namespace gpu
} // namespace faiss
