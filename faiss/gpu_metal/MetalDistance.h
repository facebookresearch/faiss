// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * IVF distance computation and scan dispatch for Metal backend.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace faiss {
namespace gpu_metal {

class MetalResources;

int getMetalDistanceMaxK();

bool runMetalComputeNorms(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> vectors,
        int nb,
        int d,
        id<MTLBuffer> normsBuf,
        bool waitForCompletion = true);

bool runMetalIVFFlatScan(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        int nq,
        int d,
        int k,
        int nprobe,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf,
        id<MTLBuffer> interleavedCodes = nil,
        id<MTLBuffer> interleavedCodesOffset = nil,
        bool waitForCompletion = true);

bool runMetalIVFFlatFullSearch(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        int nq,
        int d,
        int k,
        int nprobe,
        bool isL2,
        id<MTLBuffer> centroids,
        int nlist,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf,
        id<MTLBuffer> coarseDistBuf,
        id<MTLBuffer> coarseIdxBuf,
        id<MTLBuffer> distMatrixBuf,
        id<MTLBuffer> centroidNormsBuf = nil,
        int avgListLen = 256,
        id<MTLBuffer> interleavedCodes = nil,
        id<MTLBuffer> interleavedCodesOffset = nil,
        bool centroidsAreFP16 = false,
        bool waitForCompletion = true);

/// Full IVF-PQ search on the GPU: builds per-(query, probe) PQ lookup tables,
/// scans the assigned inverted lists with 8-bit ADC, and merges the per-list
/// top-k into the final top-k. Coarse quantization is performed on the CPU and
/// the assignments are passed in via @p coarseAssign.
bool runMetalIVFPQFullSearch(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> coarseAssign,
        id<MTLBuffer> coarseCentroids,
        id<MTLBuffer> pqCentroids,
        id<MTLBuffer> lookupTable,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        int nq,
        int d,
        int M,
        int k,
        int nprobe,
        int nlist,
        int avgListLen,
        bool lookupFp16,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf,
        bool waitForCompletion = true);

/// Build the query-independent precomputed term of the IVFPQ L2 distance
/// decomposition on the GPU: term2[l][m][c] = ||pq[m][c]||^2 +
/// 2 <coarse_l[m], pq[m][c]>, laid out as nlist * M * 256 floats. Run once
/// per trained index; consumed by runMetalIVFPQPrecompSearch.
bool runMetalIVFPQPrecomputeTerm2(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> coarseCentroids,
        id<MTLBuffer> pqCentroids,
        id<MTLBuffer> outTerm2,
        int nlist,
        int d,
        int M);

/// IVF-PQ search using the precomputed-table decomposition: a per-query
/// M*256 term is built once per batch (instead of a full residual LUT per
/// (query, probe)), combined with @p term2 in threadgroup memory by the scan
/// kernel, and the coarse distance is added as the constant ||x - c||^2 term.
/// The scan keeps an exact running top-k over arbitrarily long lists and the
/// merge runs in rounds, so there is no list-length or nprobe*k cap; the
/// remaining requirements are k <= 512, M <= 16 and d/M <= 256.
/// @p term2 may be nil for inner product (qterm is the whole table);
/// @p useDis0 adds the coarse distance (required for L2, by_residual for IP).
bool runMetalIVFPQPrecompSearch(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> coarseAssign,
        id<MTLBuffer> coarseDist,
        id<MTLBuffer> term2,
        id<MTLBuffer> qtermScratch,
        id<MTLBuffer> pqCentroids,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        int nq,
        int d,
        int M,
        int k,
        int nprobe,
        bool isL2,
        bool useDis0,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf,
        id<MTLBuffer> mergeScratchDistBuf,
        id<MTLBuffer> mergeScratchIdxBuf,
        bool waitForCompletion = true);

} // namespace gpu_metal
} // namespace faiss
