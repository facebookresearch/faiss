// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#import <Metal/Metal.h>
#include <cstdint>

namespace faiss::gpu_metal {
struct MetalIVFSQScanPlan {
    uint32_t nq = 0, k = 0, chunks = 0, chunkSize = 0, parts = 0, fanIn = 0;
    size_t partialItems = 0, mergeItems = 0;
};

// Pure checked sizing, also used by the index before selecting GPU execution.
bool makeMetalIVFSQScanPlan(
        uint32_t nq,
        uint32_t probes,
        uint32_t k,
        uint32_t maxListLength,
        size_t maxBufferBytes,
        MetalIVFSQScanPlan& plan);

// Constructors/copy/destruction are compiled under ARC, including for callers
// compiled without ARC. Callers must serialize access to reusable scratch.
struct MetalIVFSQScanScratch {
    id<MTLBuffer> partialDistances = nil, partialIds = nil;
    id<MTLBuffer> mergeDistances = nil, mergeIds = nil;
    MetalIVFSQScanScratch();
    ~MetalIVFSQScanScratch();
    MetalIVFSQScanScratch(const MetalIVFSQScanScratch&);
    MetalIVFSQScanScratch& operator=(const MetalIVFSQScanScratch&);
    void reserve(id<MTLDevice> device, const MetalIVFSQScanPlan& plan);
};

/// Synchronous direct scan entry point. Buffers are shared, sized by params:
/// nq,d,k,nprobe,nlist,fp16,residual,L2,code_size. IDs must be nonnegative;
/// offsets/lengths describe disjoint ranges inside codes/ids. Throws on
/// failure. flatCoarse computes assignments/scores on GPU from centroids;
/// otherwise assignments/scores are caller inputs.
void runMetalIVFSQScan(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> offsets,
        id<MTLBuffer> lengths,
        id<MTLBuffer> assignments,
        id<MTLBuffer> centroids,
        id<MTLBuffer> ranges,
        id<MTLBuffer> distances,
        id<MTLBuffer> labels,
        id<MTLBuffer> coarseScores,
        const uint32_t* params,
        MetalIVFSQScanScratch* scratch = nullptr,
        bool flatCoarse = false,
        bool coarseL2 = true);
} // namespace faiss::gpu_metal
