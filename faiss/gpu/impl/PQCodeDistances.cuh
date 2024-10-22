/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cublas_v2.h>
#include <faiss/Index.h>
#include <faiss/gpu/utils/NoTypeTensor.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

class DeviceMemory;

/// pqCentroids is of the form (sub q)(sub dim)(code id)
/// Calculates the distance from the (query - centroid) residual to
/// each sub-code vector, for the given list of query results in
/// coarseIndices
template <typename CentroidT>
void runPQCodeDistances(
        GpuResources* res,
        Tensor<float, 3, true>& pqCentroids,
        Tensor<float, 2, true>& queries,
        Tensor<CentroidT, 2, true>& coarseCentroids,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        NoTypeTensor<4, true>& outCodeDistances,
        bool useMMImplementation,
        bool l2Distance,
        bool useFloat16Lookup);

} // namespace gpu
} // namespace faiss

#include <faiss/gpu/impl/PQCodeDistances-inl.cuh>
