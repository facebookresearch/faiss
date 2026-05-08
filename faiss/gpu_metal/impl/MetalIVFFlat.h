// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal IVF Flat implementation: GPU-resident IVF list storage and helpers.
 * Mirrors the roles of faiss/gpu/impl/IVFFlat.cuh (storage side only).
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu_metal/MetalResources.h>

namespace faiss {
namespace gpu_metal {

/// GPU-resident IVF list storage for flat (float32) codes.
/// Layout: all lists are stored contiguously in a single codes/ids buffer;
/// lists are described by (listOffset[list], listLength[list]).
class MetalIVFFlatImpl {
public:
    struct AppendDebugStats {
        size_t relayoutEvents = 0;
        size_t movedLists = 0;
        size_t movedVectors = 0;
        size_t reusedSegmentAllocs = 0;
        size_t tailSegmentAllocs = 0;
        size_t reusedCapacityVecs = 0;
        size_t tailCapacityVecs = 0;
        size_t tailShrinkEvents = 0;
        size_t tailShrunkVecs = 0;
    };

    MetalIVFFlatImpl(
            std::shared_ptr<MetalResources> resources,
            int dim,
            idx_t nlist,
            faiss::MetricType metric,
            float metricArg,
            faiss::gpu::IndicesOptions indicesOptions,
            bool interleavedLayout);

    ~MetalIVFFlatImpl();

    /// Reset all IVF lists and free GPU storage.
    void reset();

    /// Reserve host/GPU storage for at least totalVecs vectors.
    void reserveMemory(idx_t totalVecs);

    /// Append a batch of vectors to IVF lists.
    /// - x: host pointer, size n * dim
    /// - list_nos: host pointer, size n; -1 entries are skipped
    /// - xids: host pointer, size n (may be null to use internal ids)
    void appendVectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            const idx_t* xids);

    /// Accessors for future GPU search path.
    int dim() const {
        return dim_;
    }
    idx_t nlist() const {
        return nlist_;
    }
    faiss::MetricType metricType() const {
        return metric_type_;
    }
    float metricArg() const {
        return metric_arg_;
    }

    const std::vector<size_t>& listLength() const {
        return listLength_;
    }
    const std::vector<size_t>& listOffset() const {
        return listOffset_;
    }

    id<MTLBuffer> codesBuffer() const {
        return codesBuffer_;
    }
    id<MTLBuffer> idsBuffer() const {
        return idsBuffer_;
    }
    /// Pre-built GPU buffer of (nlist) uint32_t offsets (updated on every add).
    id<MTLBuffer> listOffsetGpuBuffer() const {
        return listOffsetBuf_;
    }
    /// Pre-built GPU buffer of (nlist) uint32_t lengths (updated on every add).
    id<MTLBuffer> listLengthGpuBuffer() const {
        return listLengthBuf_;
    }

    size_t totalVecs() const {
        return totalVecs_;
    }

    /// Interleaved codes buffer (blocks of 32 vectors, dims interleaved).
    id<MTLBuffer> interleavedCodesBuffer() const {
        return interleavedCodesBuf_;
    }
    /// Per-list float offsets into the interleaved codes buffer.
    id<MTLBuffer> interleavedCodesOffsetBuffer() const {
        return interleavedCodesOffsetBuf_;
    }
    bool interleavedLayout() const {
        return interleavedLayout_;
    }
    /// Rebuild interleaved buffers from host storage if they are stale.
    void ensureInterleavedLayoutUpToDate();
    const AppendDebugStats& appendDebugStats() const {
        return appendStats_;
    }
    void resetAppendDebugStats() {
        appendStats_ = AppendDebugStats{};
    }

private:
    struct FreeSegment {
        size_t offset = 0;
        size_t length = 0;
    };

    bool ensureCapacityForAppend_(
            const std::vector<size_t>& addPerList,
            std::vector<uint8_t>* movedLists);
    void uploadToGpu_(
            const std::vector<size_t>& oldLength,
            const std::vector<size_t>& addPerList,
            const std::vector<uint8_t>& movedLists,
            bool forceFullUpload);
    void rebuildInterleavedBuffers_();
    size_t allocSegment_(size_t length);
    void freeSegment_(size_t offset, size_t length, bool allowTailShrink = true);
    void coalesceFreeSegments_();
    void tryShrinkTail_();

    std::shared_ptr<MetalResources> resources_;

    int dim_;
    idx_t nlist_;
    faiss::MetricType metric_type_;
    float metric_arg_;
    faiss::gpu::IndicesOptions indicesOptions_;
    bool interleavedLayout_;

    // Per-list metadata
    std::vector<size_t> listLength_;
    std::vector<size_t> listOffset_;
    std::vector<size_t> listCapacity_;

    // Host copies of IVF data (flat layout)
    std::vector<float> hostCodes_; // size = totalVecs_ * dim_
    std::vector<idx_t> hostIds_;   // size = totalVecs_
    std::vector<FreeSegment> freeSegments_;
    AppendDebugStats appendStats_;
    size_t totalVecs_;
    size_t totalCapacityVecs_;

    // GPU storage
    id<MTLBuffer> codesBuffer_;
    id<MTLBuffer> idsBuffer_;
    id<MTLBuffer> listOffsetBuf_;  // (nlist) uint32_t, list element offsets
    id<MTLBuffer> listLengthBuf_;  // (nlist) uint32_t, list sizes

    // Interleaved codes layout (blocks of 32 vectors, dims interleaved)
    id<MTLBuffer> interleavedCodesBuf_;
    id<MTLBuffer> interleavedCodesOffsetBuf_; // (nlist) uint32_t, float offsets
    bool interleavedDirty_ = true;

    static constexpr int kInterleavedGroupSize = 32;
};

} // namespace gpu_metal
} // namespace faiss
