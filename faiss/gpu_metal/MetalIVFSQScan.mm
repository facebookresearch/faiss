// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/gpu_metal/MetalIVFSQScan.h>
#include <faiss/gpu_metal/MetalKernels.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <limits>

namespace faiss::gpu_metal {
namespace {
void requireBytes(id<MTLBuffer> buffer, size_t a, size_t b, size_t width) {
    FAISS_THROW_IF_NOT(a == 0 || b <= SIZE_MAX / a);
    size_t count = a * b;
    FAISS_THROW_IF_NOT(count <= SIZE_MAX / width);
    FAISS_THROW_IF_NOT(
            buffer && buffer.contents && buffer.length >= count * width);
}
struct EncoderGuard {
    id<MTLComputeCommandEncoder> encoder;
    ~EncoderGuard() {
        if (encoder)
            [encoder endEncoding];
    }
};
} // namespace

MetalIVFSQScanScratch::MetalIVFSQScanScratch() = default;
MetalIVFSQScanScratch::~MetalIVFSQScanScratch() = default;
MetalIVFSQScanScratch::MetalIVFSQScanScratch(const MetalIVFSQScanScratch&) =
        default;
MetalIVFSQScanScratch& MetalIVFSQScanScratch::operator=(
        const MetalIVFSQScanScratch&) = default;

bool makeMetalIVFSQScanPlan(
        uint32_t nq,
        uint32_t probes,
        uint32_t k,
        uint32_t maxLength,
        size_t maxBytes,
        MetalIVFSQScanPlan& plan) {
    if (!nq || !probes || !k || k > 512)
        return false;
    uint64_t chunkSize =
            std::max(nq < 32 ? 512u : 2048u, k * (nq < 32 ? 2u : 8u));
    uint64_t chunks = std::min(
            uint64_t(64),
            std::max(
                    uint64_t(1),
                    (uint64_t(maxLength) + chunkSize - 1) / chunkSize));
    chunkSize =
            std::max(chunkSize, (uint64_t(maxLength) + chunks - 1) / chunks);
    uint64_t parts = uint64_t(probes) * chunks;
    if (parts > UINT32_MAX)
        return false;
    uint32_t fanIn = std::min(64u, 1024u / k);
    uint64_t groups = (parts - 1) / fanIn + 1;
    auto sized = [&](uint64_t lists, size_t& items) {
        if (lists > SIZE_MAX / nq || lists * nq > SIZE_MAX / k)
            return false;
        uint64_t n = lists * nq * k;
        if (n > maxBytes / sizeof(int64_t))
            return false;
        items = size_t(n);
        return true;
    };
    MetalIVFSQScanPlan next;
    next.nq = nq;
    next.k = k;
    next.chunks = chunks;
    next.chunkSize = chunkSize;
    next.parts = parts;
    next.fanIn = fanIn;
    if (parts > 1 && !sized(parts, next.partialItems))
        return false;
    if (groups > 1 && !sized(groups, next.mergeItems))
        return false;
    plan = next;
    return true;
}

void MetalIVFSQScanScratch::reserve(
        id<MTLDevice> device,
        const MetalIVFSQScanPlan& plan) {
    auto grow = [&](id<MTLBuffer> old, size_t items, size_t width) {
        if (!items)
            return old;
        FAISS_THROW_IF_NOT(items <= SIZE_MAX / width);
        size_t bytes = items * width;
        if (old && old.length >= bytes)
            return old;
        FAISS_THROW_IF_NOT(bytes <= device.maxBufferLength);
        size_t capacity = old
                ? std::min(
                          size_t(device.maxBufferLength),
                          size_t(old.length <= device.maxBufferLength / 2
                                         ? old.length * 2
                                         : device.maxBufferLength))
                : 8;
        auto buffer = [device newBufferWithLength:std::max(bytes, capacity)
                                          options:MTLResourceStorageModeShared];
        FAISS_THROW_IF_NOT_MSG(
                buffer, "Metal IVFSQ scan scratch allocation failed");
        return buffer;
    };
    auto pd = grow(partialDistances, plan.partialItems, 4);
    auto pi = grow(partialIds, plan.partialItems, 8);
    auto md = grow(mergeDistances, plan.mergeItems, 4);
    auto mi = grow(mergeIds, plan.mergeItems, 8);
    partialDistances = pd;
    partialIds = pi;
    mergeDistances = md;
    mergeIds = mi;
}

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
        const uint32_t* p,
        MetalIVFSQScanScratch* scratch,
        bool flatCoarse,
        bool coarseL2) {
    @autoreleasepool {
        FAISS_THROW_IF_NOT(device && queue && p);
        FAISS_THROW_IF_NOT(p[0] && p[1] && p[2] && p[2] <= 512 && p[3] && p[4]);
        FAISS_THROW_IF_NOT(uint64_t(p[1]) * (p[5] ? 2 : 1) == p[8]);
        requireBytes(queries, p[0], p[1], 4);
        requireBytes(assignments, p[0], p[3], 4);
        requireBytes(coarseScores, p[0], p[3], 4);
        requireBytes(offsets, p[4], 1, 4);
        requireBytes(lengths, p[4], 1, 4);
        requireBytes(centroids, p[4], (p[6] || flatCoarse) ? p[1] : 0, 4);
        FAISS_THROW_IF_NOT(
                !flatCoarse ||
                (p[3] <= 512 && p[3] <= p[4] && p[4] <= INT32_MAX));
        requireBytes(ranges, p[5] ? 0 : 2, p[1], 4);
        requireBytes(distances, p[0], p[2], 4);
        requireBytes(labels, p[0], p[2], 8);
        const auto* off = (const uint32_t*)offsets.contents;
        const auto* len = (const uint32_t*)lengths.contents;
        size_t maxEnd = 0;
        uint32_t maxLength = 0;
        uint32_t minLength = UINT32_MAX;
        for (size_t l = 0; l < p[4]; ++l) {
            maxEnd = std::max(maxEnd, size_t(off[l]) + len[l]);
            maxLength = std::max(maxLength, len[l]);
            minLength = std::min(minLength, len[l]);
        }
        requireBytes(codes, maxEnd, p[8], 1);
        requireBytes(ids, maxEnd, 1, 8);
        MetalIVFSQScanPlan plan;
        FAISS_THROW_IF_NOT(makeMetalIVFSQScanPlan(
                p[0], p[3], p[2], maxLength, device.maxBufferLength, plan));
        // A bounded short-list scan can select across all probes in one group,
        // writing final results without partial buffers or a merge dispatch.
        bool fused = p[1] == 128 && p[2] <= 32 && p[3] <= 8 &&
                maxLength <= 256;
        if (fused) {
            plan.parts = 1;
            plan.chunks = 1;
            plan.chunkSize = maxLength * p[3];
            plan.partialItems = plan.mergeItems = 0;
        }
        // An extra dispatch does not amortize for a single query.
        // CPU assignments are available before submission. Verify validity before
        // using a list-length lower bound; invalid probes contribute no work.
        uint32_t adaptiveLimit = p[7] ? 2048u : 512u;
        bool allLong = false;
        if (!flatCoarse && uint64_t(minLength) * p[3] > adaptiveLimit) {
            allLong = true;
            const auto* selected = (const int32_t*)assignments.contents;
            for (uint64_t i = 0; i < uint64_t(p[0]) * p[3]; ++i) {
                if (selected[i] < 0 || uint32_t(selected[i]) >= p[4]) {
                    allLong = false;
                    break;
                }
            }
        }
        // IP uses the regular parallel scan; avoid a second pass for cheap scores.
        bool adaptive = p[7] && p[0] > 1 && !fused && !allLong &&
                p[1] == 128 && p[2] <= 32 && p[3] <= 8;
        MetalIVFSQScanScratch local;
        if (!scratch)
            scratch = &local;
        scratch->reserve(device, plan);
        auto& kernels = getMetalKernels(device);
        FAISS_THROW_IF_NOT(kernels.isValid());
        id<MTLCommandBuffer> command = [queue commandBuffer];
        FAISS_THROW_IF_NOT(command);
        EncoderGuard guard{[command computeCommandEncoder]};
        FAISS_THROW_IF_NOT(guard.encoder);
        if (flatCoarse) {
            kernels.encodeIVFSQCoarse(
                    guard.encoder,
                    queries,
                    centroids,
                    coarseScores,
                    assignments,
                    p[0],
                    p[1],
                    p[4],
                    p[3],
                    coarseL2);
            [guard.encoder endEncoding];
            guard.encoder = nil;
            guard.encoder = [command computeCommandEncoder];
            FAISS_THROW_IF_NOT(guard.encoder);
        }
        id<MTLBuffer> __unsafe_unretained buffers[] = {
                queries,
                codes,
                ids,
                offsets,
                lengths,
                assignments,
                centroids,
                ranges,
                plan.parts > 1 ? scratch->partialDistances : distances,
                plan.parts > 1 ? scratch->partialIds : labels,
                coarseScores};
        kernels.encodeIVFSQScan(
                guard.encoder, buffers, p, plan.chunks, plan.chunkSize, fused, adaptive);
        [guard.encoder endEncoding];
        guard.encoder = nil;
        if (adaptive) {
            // Both passes share the same partial layout. Separate encoders
            // order padding writes before fused results replace part zero.
            guard.encoder = [command computeCommandEncoder];
            FAISS_THROW_IF_NOT(guard.encoder);
            kernels.encodeIVFSQScan(
                    guard.encoder, buffers, p, plan.chunks, plan.chunkSize, true, true);
            [guard.encoder endEncoding];
            guard.encoder = nil;
        }
        uint32_t parts = plan.parts;
        id<MTLBuffer> inputD = scratch->partialDistances,
                      inputI = scratch->partialIds;
        bool writeMerge = true;
        while (parts > 1) {
            uint32_t groups = (parts - 1) / plan.fanIn + 1;
            id<MTLBuffer> outputD = groups == 1
                    ? distances
                    : (writeMerge ? scratch->mergeDistances
                                  : scratch->partialDistances);
            id<MTLBuffer> outputI = groups == 1
                    ? labels
                    : (writeMerge ? scratch->mergeIds : scratch->partialIds);
            guard.encoder = [command computeCommandEncoder];
            FAISS_THROW_IF_NOT(guard.encoder);
            uint32_t mergeParams[] = {p[0], parts, plan.fanIn, p[2], p[7]};
            kernels.encodeIVFSQMerge(
                    guard.encoder,
                    inputD,
                    inputI,
                    outputD,
                    outputI,
                    mergeParams);
            [guard.encoder endEncoding];
            guard.encoder = nil;
            inputD = outputD;
            inputI = outputI;
            parts = groups;
            writeMerge = !writeMerge;
        }
        [command commit];
        [command waitUntilCompleted];
        FAISS_THROW_IF_NOT_MSG(
                command.status == MTLCommandBufferStatusCompleted,
                "Metal IVFSQ command execution failed");
    }
}
} // namespace faiss::gpu_metal
