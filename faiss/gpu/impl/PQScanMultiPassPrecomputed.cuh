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
#include <faiss/gpu/utils/NoTypeTensor.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

void runPQScanMultiPassPrecomputed(
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& precompTerm1,
        NoTypeTensor<3, true>& precompTerm2,
        NoTypeTensor<3, true>& precompTerm3,
        Tensor<idx_t, 2, true>& ivfListIds,
        bool useFloat16Lookup,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        DeviceVector<void*>& listCodes,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        int k,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

void runPQScanMultiPassPrecomputed(
        Tensor<float, 2, true>& precompTerm1,
        NoTypeTensor<4, true>& precompTerm2,
        NoTypeTensor<4, true>& precompTerm3,
        int coarseCodebookSize,
        Tensor<ushort2, 2, true>& topQueryToCentroid,
        bool useFloat16Lookup,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        Tensor<uint8_t*, 1, true>& listCodes,
        Tensor<int*, 1, true>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<int, 1, true>& listLengths,
        int maxListLength,
        int k,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

void runPQScanMultiPassPrecomputed(
        Tensor<float, 2, true>& precompTerm1,
        NoTypeTensor<4, true>& precompTerm2,
        NoTypeTensor<4, true>& precompTerm3,
        int coarseCodebookSize,
        Tensor<ushort2, 2, true>& topQueryToCentroid,
        bool useFloat16Lookup,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        Tensor<uint8_t*, 1, true>& listCodes,
        Tensor<idx_t*, 1, true>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<int, 1, true>& listLengths,
        int maxListLength,
        int k,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

void runPQScanMultiPassPrecomputed(
        Tensor<float, 2, true>& precompTerm1,
        NoTypeTensor<4, true>& precompTerm2,
        NoTypeTensor<4, true>& precompTerm3,
        int coarseCodebookSize,
        Tensor<ushort2, 2, true>& topQueryToCentroid,
        bool useFloat16Lookup,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        Tensor<unsigned int, 1, true>& listOffsets,
        Tensor<uint8_t, 1, true, long>& listCodes,
        int codeNumBytes,
        Tensor<int, 1, true>& listIndices,
        IndicesOptions indicesOptions,
        int maxListLength,
        int k,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

void runPQScanMultiPassPrecomputed(
        Tensor<float, 2, true>& precompTerm1,
        NoTypeTensor<4, true>& precompTerm2,
        NoTypeTensor<4, true>& precompTerm3,
        int coarseCodebookSize,
        Tensor<ushort2, 2, true>& topQueryToCentroid,
        bool useFloat16Lookup,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        Tensor<unsigned int, 1, true>& listOffsets,
        Tensor<uint8_t, 1, true, long>& listCodes,
        int codeNumBytes,
        Tensor<idx_t, 1, true>& listIndices,
        IndicesOptions indicesOptions,
        int maxListLength,
        int k,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

} // namespace gpu
} // namespace faiss
