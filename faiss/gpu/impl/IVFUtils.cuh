/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

// A collection of utility functions for IVFPQ and IVFFlat, for
// post-processing and k-selecting the results
namespace faiss {
namespace gpu {

class GpuResources;

/// Function for multi-pass scanning that collects the length of
/// intermediate results for all (query, probe) pair
void runCalcListOffsets(
        GpuResources* res,
        Tensor<idx_t, 2, true>& ivfListIds,
        DeviceVector<int>& listLengths,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<char, 1, true>& thrustMem,
        cudaStream_t stream);

/// Performs a first pass of k-selection on the results
void runPass1SelectLists(
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<float, 1, true>& distance,
        int nprobe,
        int k,
        bool chooseLargest,
        Tensor<float, 3, true>& heapDistances,
        Tensor<int, 3, true>& heapIndices,
        cudaStream_t stream);

/// Performs a final pass of k-selection on the results, producing the
/// final indices
void runPass2SelectLists(
        Tensor<float, 2, true>& heapDistances,
        Tensor<int, 2, true>& heapIndices,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<idx_t, 2, true>& ivfListIds,
        int k,
        bool chooseLargest,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
