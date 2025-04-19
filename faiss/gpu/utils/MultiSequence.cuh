/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

/// Process two k-dimensional lists of distances and respecitve ids,
/// using the multi-sequence algorithm to generate as output the `w`
/// smallest and sorted distance sums with their respective id pairs
void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<ushort, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<ushort2, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream);

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<ushort, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<ushort2, 2, true>& outIndices,
        GpuResources* res);

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<idx_t, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<int2, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream);

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<idx_t, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<int2, 2, true>& outIndices,
        GpuResources* res);

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<ushort, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        const int codebookSize,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream);

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<ushort, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        const int codebookSize,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

} // namespace gpu
} // namespace faiss
