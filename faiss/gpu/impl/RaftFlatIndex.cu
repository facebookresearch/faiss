/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/RaftFlatIndex.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/distance/distance_types.hpp>
//#include <raft/neighbors/brute_force.cuh>
#include <raft/spatial/knn/detail/fused_l2_knn.cuh>

namespace faiss {
namespace gpu {

RaftFlatIndex::RaftFlatIndex(
        GpuResources* res,
        int dim,
        bool useFloat16,
        MemorySpace space)
        : FlatIndex(res, dim, useFloat16, space) {}

void RaftFlatIndex::query(
        Tensor<float, 2, true>& input,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices,
        bool exactDistance) {

    // For now, use RAFT's fused KNN when k <= 64 and L2 metric is used
    if(k <= 64 && metric == MetricType::METRIC_L2 &&
        input.getStride(0) == 0 && vectors_.getStride(0) == 0) {
        raft::handle_t &raft_handle = resources_->getRaftHandleCurrentDevice();

        auto distance = exactDistance ? raft::distance::DistanceType::L2Unexpanded :
                                      raft::distance::DistanceType::L2Expanded;

        auto index = raft::make_device_matrix_view<float>(vectors_.data(), vectors_.getSize(0), vectors_.getSize(1));
        auto search = raft::make_device_matrix_view<float>(input.data(), input.getSize(0), input.getSize(1));
        auto inds = raft::make_device_matrix_view<int>(outIndices.data(), outIndices.getSize(0), outIndices.getSize(1));
        auto dists = raft::make_device_matrix_view<float>(outDistances.data(), outDistances.getSize(0), outDistances.getSize(1));

//        raft::neighbors::brute_force::knn(raft_handle, index, search, inds, dists, k, distance);

        printf("Using RAFT for FLAT!!!!\n");
        // TODO: Expose the fused L2KNN through RAFT's public APIs
        raft::spatial::knn::detail::fusedL2Knn(dim_,
                   inds.data_handle(),
                   dists.data_handle(),
                   index.data_handle(),
                   search.data_handle(),
                   index.extent(0),
                   search.extent(0),
                   k,
                   true,
                   true,
                   raft_handle.get_stream(),
                   distance);

        } else {

            printf("Dispathing to FAISS for FLAT!!!!\n");
        FlatIndex::query(input, k, metric, metricArg, outDistances, outIndices, exactDistance);
    }
}
} // namespace gpu
} // namespace faiss
