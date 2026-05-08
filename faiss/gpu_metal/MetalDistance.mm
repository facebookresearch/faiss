// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * IVF distance computation and scan dispatch for Metal backend.
 */

#import "MetalDistance.h"
#import "MetalKernels.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace faiss {
namespace gpu_metal {

constexpr uint32_t kIvfReduceThreadgroupSize = 256;
constexpr uint32_t kIvfReduceLocalK = 4;
constexpr uint32_t kIvfReduceExactCandidates =
        kIvfReduceThreadgroupSize * kIvfReduceLocalK; // 1024
constexpr uint32_t kIvfSmallExactCandidates = 32;

inline bool fitsU32(int v) {
    return v >= 0 &&
            static_cast<uint64_t>(v) <=
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
}

inline bool ivfMergeExactnessHolds(int nprobe, int k) {
    if (nprobe <= 0 || k <= 0)
        return false;
    const uint64_t totalCandidates = (uint64_t)nprobe * (uint64_t)k;
    return totalCandidates <= (uint64_t)kIvfReduceExactCandidates;
}

inline bool ivfScanExactnessHoldsForAllLists(
        id<MTLBuffer> listLength,
        int nlist,
        uint32_t maxCandidatesPerList) {
    if (!listLength || nlist <= 0)
        return false;
    const uint32_t* listLens =
            reinterpret_cast<const uint32_t*>([listLength contents]);
    if (!listLens)
        return false;
    for (int i = 0; i < nlist; ++i) {
        if (listLens[i] > maxCandidatesPerList)
            return false;
    }
    return true;
}

inline bool ivfScanExactnessHoldsForAssignments(
        id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        int nq,
        int nprobe) {
    if (!listLength || !coarseAssign || nq <= 0 || nprobe <= 0)
        return false;
    const uint32_t* listLens =
            reinterpret_cast<const uint32_t*>([listLength contents]);
    const int32_t* assign =
            reinterpret_cast<const int32_t*>([coarseAssign contents]);
    if (!listLens || !assign)
        return false;
    const size_t nlist = [listLength length] / sizeof(uint32_t);
    if (nlist == 0)
        return false;
    const size_t total = (size_t)nq * (size_t)nprobe;
    for (size_t i = 0; i < total; ++i) {
        const int32_t listNo = assign[i];
        if (listNo < 0)
            continue;
        if ((size_t)listNo >= nlist)
            return false;
        if (listLens[listNo] > kIvfReduceExactCandidates)
            return false;
    }
    return true;
}

int getMetalDistanceMaxK() {
    return MetalKernels::kMaxK;
}

bool runMetalComputeNorms(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> vectors,
        int nb,
        int d,
        id<MTLBuffer> normsBuf,
        bool waitForCompletion) {
    if (!device || !queue || !vectors || !normsBuf || nb <= 0 || d <= 0)
        return false;
    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid())
        return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    K.encodeComputeNorms(enc, vectors, normsBuf, nb, d);
    [enc endEncoding];
    [cmdBuf commit];
    if (!waitForCompletion)
        return true;
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

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
        id<MTLBuffer> interleavedCodes,
        id<MTLBuffer> interleavedCodesOffset,
        bool waitForCompletion) {
    bool useIL =
            (interleavedCodes != nil && interleavedCodesOffset != nil);
    if (!device || !queue || !queries || (!codes && !useIL) || !ids ||
        !listOffset || !listLength || !coarseAssign || !outDistances ||
        !outIndices || !perListDistBuf || !perListIdxBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0)
        return false;
    if (d <= 0 || d > 2048)
        return false;
    if (useIL && (d % 4 != 0))
        return false;
    if (!ivfMergeExactnessHolds(nprobe, k))
        return false;
    if (!ivfScanExactnessHoldsForAssignments(
                listLength, coarseAssign, nq, nprobe))
        return false;
    if (!fitsU32(nq) || !fitsU32(d) || !fitsU32(k) || !fitsU32(nprobe))
        return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid())
        return false;

    IVFScanVariant variant = useIL ? IVFScanVariant::Interleaved
                                   : IVFScanVariant::Standard;

    uint32_t sp[5] = {
            (uint32_t)nq,
            (uint32_t)d,
            (uint32_t)k,
            (uint32_t)nprobe,
            isL2 ? 1u : 0u};
    id<MTLBuffer> paramsBuf =
            [device newBufferWithBytes:sp
                               length:sizeof(sp)
                              options:MTLResourceStorageModeShared];
    if (!paramsBuf)
        return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    K.encodeIVFScanList(
            enc,
            variant,
            queries,
            useIL ? interleavedCodes : codes,
            ids,
            listOffset,
            listLength,
            coarseAssign,
            perListDistBuf,
            perListIdxBuf,
            paramsBuf,
            nq,
            nprobe,
            useIL ? interleavedCodesOffset : nil);
    K.encodeIVFMergeLists(
            enc,
            perListDistBuf,
            perListIdxBuf,
            outDistances,
            outIndices,
            paramsBuf,
            nq);

    [enc endEncoding];
    [cmdBuf commit];
    if (!waitForCompletion)
        return true;
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

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
        id<MTLBuffer> centroidNormsBuf,
        int avgListLen,
        id<MTLBuffer> interleavedCodes,
        id<MTLBuffer> interleavedCodesOffset,
        bool waitForCompletion) {
    bool useIL =
            (interleavedCodes != nil && interleavedCodesOffset != nil);
    if (!device || !queue || !queries || !centroids ||
        (!codes && !useIL) || !ids || !listOffset || !listLength ||
        !outDistances || !outIndices || !perListDistBuf ||
        !perListIdxBuf || !coarseDistBuf || !coarseIdxBuf ||
        !distMatrixBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0 || nlist <= 0)
        return false;
    if (d <= 0 || d > 2048)
        return false;
    if (useIL && (d % 4 != 0))
        return false;
    (void)avgListLen;
    if (!ivfMergeExactnessHolds(nprobe, k))
        return false;
    if (!ivfScanExactnessHoldsForAllLists(
                listLength, nlist, kIvfReduceExactCandidates))
        return false;
    if (!fitsU32(nq) || !fitsU32(d) || !fitsU32(k) || !fitsU32(nprobe))
        return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid())
        return false;

    const bool useSmall =
            !useIL && k <= (int)kIvfSmallExactCandidates &&
            ivfScanExactnessHoldsForAllLists(
                    listLength, nlist, kIvfSmallExactCandidates);
    IVFScanVariant scanV = useIL    ? IVFScanVariant::Interleaved
                           : useSmall ? IVFScanVariant::Small
                                      : IVFScanVariant::Standard;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    // Step 1: coarse distance matrix
    bool fusedL2 = isL2 && centroidNormsBuf != nil;
    if (fusedL2) {
        K.encodeL2WithNorms(
                enc, queries, centroids, distMatrixBuf,
                centroidNormsBuf, nq, nlist, d);
    } else {
        K.encodeDistanceMatrix(
                enc, queries, centroids, distMatrixBuf, nq, nlist, d,
                isL2 ? METRIC_L2 : METRIC_INNER_PRODUCT);
    }

    // Step 2: coarse top-nprobe
    K.encodeTopKThreadgroup(
            enc, distMatrixBuf, coarseDistBuf, coarseIdxBuf, nq, nlist,
            nprobe, isL2);

    // Step 3: IVF scan
    uint32_t sp[5] = {
            (uint32_t)nq,
            (uint32_t)d,
            (uint32_t)k,
            (uint32_t)nprobe,
            isL2 ? 1u : 0u};
    id<MTLBuffer> paramsBuf =
            [device newBufferWithBytes:sp
                               length:sizeof(sp)
                              options:MTLResourceStorageModeShared];
    if (!paramsBuf) {
        [enc endEncoding];
        return false;
    }

    K.encodeIVFScanList(
            enc,
            scanV,
            queries,
            useIL ? interleavedCodes : codes,
            ids,
            listOffset,
            listLength,
            coarseIdxBuf,
            perListDistBuf,
            perListIdxBuf,
            paramsBuf,
            nq,
            nprobe,
            useIL ? interleavedCodesOffset : nil);

    // Step 4: merge
    K.encodeIVFMergeLists(
            enc, perListDistBuf, perListIdxBuf, outDistances, outIndices,
            paramsBuf, nq);

    [enc endEncoding];
    [cmdBuf commit];
    if (!waitForCompletion)
        return true;
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

} // namespace gpu_metal
} // namespace faiss
