/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndicesOptions.h>
#include <thrust/device_vector.h>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

/// Append user indices to IVF lists
void runIVFIndicesAppend(
        Tensor<int, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<Index::idx_t, 1, true>& indices,
        IndicesOptions opt,
        thrust::device_vector<void*>& listIndices,
        cudaStream_t stream);

/// Update device-side list pointers in a batch
void runUpdateListPointers(
        Tensor<int, 1, true>& listIds,
        Tensor<int, 1, true>& newListLength,
        Tensor<void*, 1, true>& newCodePointers,
        Tensor<void*, 1, true>& newIndexPointers,
        thrust::device_vector<int>& listLengths,
        thrust::device_vector<void*>& listCodes,
        thrust::device_vector<void*>& listIndices,
        cudaStream_t stream);

/// Append PQ codes to IVF lists (non-interleaved format)
void runIVFPQAppend(
        Tensor<int, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<uint8_t, 2, true>& encodings,
        thrust::device_vector<void*>& listCodes,
        cudaStream_t stream);

/// Append PQ codes to IVF lists (interleaved format)
void runIVFPQInterleavedAppend(
        Tensor<int, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<int, 1, true>& uniqueLists,
        Tensor<int, 1, true>& vectorsByUniqueList,
        Tensor<int, 1, true>& uniqueListVectorStart,
        Tensor<int, 1, true>& uniqueListStartOffset,
        int bitsPerCode,
        Tensor<uint8_t, 2, true>& encodings,
        thrust::device_vector<void*>& listCodes,
        cudaStream_t stream);

/// Append SQ codes to IVF lists (non-interleaved, old format)
void runIVFFlatAppend(
        Tensor<int, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<float, 2, true>& vecs,
        GpuScalarQuantizer* scalarQ,
        thrust::device_vector<void*>& listData,
        cudaStream_t stream);

/// Append SQ codes to IVF lists (interleaved)
void runIVFFlatInterleavedAppend(
        Tensor<int, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<int, 1, true>& uniqueLists,
        Tensor<int, 1, true>& vectorsByUniqueList,
        Tensor<int, 1, true>& uniqueListVectorStart,
        Tensor<int, 1, true>& uniqueListStartOffset,
        Tensor<float, 2, true>& vecs,
        GpuScalarQuantizer* scalarQ,
        thrust::device_vector<void*>& listData,
        GpuResources* res,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
