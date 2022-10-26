/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <raft/core/handle.hpp>
#include <raft/neighbors/ivf_flat_types.hpp>

#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>

#include <optional>

namespace faiss {
namespace gpu {

class RaftIVFFlat : public IVFFlat {
   public:
    RaftIVFFlat(GpuResources* resources,
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


    /// Find the approximate k nearest neigbors for `queries` against
    /// our database
    void search(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& queries,
            int nprobe,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<Index::idx_t, 2, true>& outIndices) override;

    /// Classify and encode/add vectors to our IVF lists.
    /// The input data must be on our current device.
    /// Returns the number of vectors successfully added. Vectors may
    /// not be able to be added because they contain NaNs.
    int addVectors(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& vecs,
            Tensor<Index::idx_t, 1, true>& indices);


   protected:
    std::optional<raft::neighbors::ivf_flat::index<float, Index::idx_t>> raft_knn_index{std::nullopt};

};

} // namespace gpu
} // namespace faiss
