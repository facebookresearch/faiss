/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

/// Append user indices to IVF lists
void runIVFIndicesAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        DeviceVector<void*>& listIndices,
        cudaStream_t stream);

/// Update device-side list pointers in a batch
void runUpdateListPointers(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& newListLength,
        Tensor<void*, 1, true>& newCodePointers,
        Tensor<void*, 1, true>& newIndexPointers,
        DeviceVector<idx_t>& listLengths,
        DeviceVector<void*>& listCodes,
        DeviceVector<void*>& listIndices,
        cudaStream_t stream);

/// Append PQ codes to IVF lists (non-interleaved format)
void runIVFPQAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<uint8_t, 2, true>& encodings,
        DeviceVector<void*>& listCodes,
        cudaStream_t stream);

/// Append PQ codes to IVF lists (interleaved format)
void runIVFPQInterleavedAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& uniqueLists,
        Tensor<idx_t, 1, true>& vectorsByUniqueList,
        Tensor<idx_t, 1, true>& uniqueListVectorStart,
        Tensor<idx_t, 1, true>& uniqueListStartOffset,
        int bitsPerCode,
        Tensor<uint8_t, 2, true>& encodings,
        DeviceVector<void*>& listCodes,
        cudaStream_t stream);

/// Append SQ codes to IVF lists (non-interleaved, old format)
void runIVFFlatAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<float, 2, true>& vecs,
        GpuScalarQuantizer* scalarQ,
        DeviceVector<void*>& listData,
        cudaStream_t stream);

/// Append SQ codes to IVF lists (interleaved)
void runIVFFlatInterleavedAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& uniqueLists,
        Tensor<idx_t, 1, true>& vectorsByUniqueList,
        Tensor<idx_t, 1, true>& uniqueListVectorStart,
        Tensor<idx_t, 1, true>& uniqueListStartOffset,
        Tensor<float, 2, true>& vecs,
        GpuScalarQuantizer* scalarQ,
        DeviceVector<void*>& listData,
        GpuResources* res,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
