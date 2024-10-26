/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

/// For the final k-selection of IVF query distances, we perform two passes.
/// The first pass scans some number of per-IVF list distances reducing them to
/// at most 8, then a second pass processes these <= 8 to the single final list
/// of NN candidates
size_t getIVFKSelectionPass2Chunks(size_t nprobe);

/// Function to determine amount of temporary space that we allocate
/// for storing basic IVF list scanning distances during query, based on the
/// memory allocation per query. This is the memory requirement for
/// IVFFlat/IVFSQ but IVFPQ will add some additional allocation as well (see
/// getIVFPQPerQueryTempMemory)
size_t getIVFPerQueryTempMemory(size_t k, size_t nprobe, size_t maxListLength);

/// Function to determine amount of temporary space that we allocate
/// for storing basic IVFPQ list scanning distances during query, based on the
/// memory allocation per query.
size_t getIVFPQPerQueryTempMemory(
        size_t k,
        size_t nprobe,
        size_t maxListLength,
        bool usePrecomputedCodes,
        size_t numSubQuantizers,
        size_t numSubQuantizerCodes);

/// Based on the amount of temporary memory needed per IVF query (determined by
/// one of the above functions) and the amount of current temporary memory
/// available, determine how many queries we will run concurrently in a single
/// tile so as to stay within reasonable temporary memory allocation limits.
size_t getIVFQueryTileSize(
        size_t numQueries,
        size_t tempMemoryAvailable,
        size_t sizePerQuery);

/// Function for multi-pass scanning that collects the length of
/// intermediate results for all (query, probe) pair
void runCalcListOffsets(
        GpuResources* res,
        Tensor<idx_t, 2, true>& ivfListIds,
        DeviceVector<idx_t>& listLengths,
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<char, 1, true>& thrustMem,
        cudaStream_t stream);

/// Performs a first pass of k-selection on the results
void runPass1SelectLists(
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<float, 1, true>& distance,
        int nprobe,
        int k,
        bool use64BitSelection,
        bool chooseLargest,
        Tensor<float, 3, true>& heapDistances,
        Tensor<idx_t, 3, true>& heapIndices,
        cudaStream_t stream);

/// Performs a final pass of k-selection on the results, producing the
/// final indices
void runPass2SelectLists(
        Tensor<float, 2, true>& heapDistances,
        Tensor<idx_t, 2, true>& heapIndices,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<idx_t, 2, true>& ivfListIds,
        int k,
        bool use64BitSelection,
        bool chooseLargest,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
