/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

void runIMIUpdateStartOffsets(
        Tensor<unsigned int, 1, true>& listStartOffsets,
        Tensor<unsigned int, 1, true>& newlistStartOffsets,
        cudaStream_t stream);

/// Append user indices to IMI lists
void runIMIIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        Tensor<int, 1, true>& listIndices,
        Tensor<unsigned int, 1, true>& listStartOffsets,
        cudaStream_t stream);

void runIMIIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        Tensor<idx_t, 1, true>& listIndices,
        Tensor<unsigned int, 1, true>& listStartOffsets,
        cudaStream_t stream);

/// Append PQ codes to IMI lists (non-interleaved format)
void runIMIPQAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<uint8_t, 2, true>& encodings,
        Tensor<uint8_t, 1, true, long>& listCodes,
        Tensor<unsigned int, 1, true>& listStartOffsets,
        int encodingNumBytes,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
