// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Minimal Metal IVFFlat wrapper.
 *
 */

#pragma once

#import <Metal/Metal.h>

#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu_metal/MetalIndex.h>

#include <memory>

namespace faiss {
namespace gpu_metal {
class MetalIVFFlatImpl;
} // namespace gpu_metal
} // namespace faiss

namespace faiss {
namespace gpu_metal {

/// IVFFlat index wrapper for Metal backend.
/// Currently delegates to an internal CPU IndexIVFFlat; later phases
/// may move list scanning to GPU.
class MetalIndexIVFFlat : public MetalIndex {
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

    /// Construct empty IVFFlat index with its own CPU quantizer.
    MetalIndexIVFFlat(
            std::shared_ptr<MetalResources> resources,
            int dims,
            idx_t nlist,
            faiss::MetricType metric,
            float metricArg = 0.0f,
            MetalIndexConfig config = MetalIndexConfig());

    /// Construct empty IVFFlat index with caller-provided coarse quantizer.
    /// If ownFields is true, this index takes ownership of `coarseQuantizer`.
    MetalIndexIVFFlat(
            std::shared_ptr<MetalResources> resources,
            faiss::Index* coarseQuantizer,
            int dims,
            idx_t nlist,
            faiss::MetricType metric,
            float metricArg = 0.0f,
            MetalIndexConfig config = MetalIndexConfig(),
            bool ownFields = false);

    /// Construct from an existing CPU IndexIVFFlat (used by cloners later).
    MetalIndexIVFFlat(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexIVFFlat* cpuIndex,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndexIVFFlat() override;

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Search with caller-provided coarse assignments (skips coarse quantizer).
    /// @param assign      Coarse list assignments (n x nprobe), row-major idx_t
    /// @param centroid_dis Distances to assigned centroids (n x nprobe); unused
    ///                     by GPU scan but accepted for API compatibility
    /// @param store_pairs  Ignored (always false for GPU path)
    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const;

    /// Copy from a CPU IndexIVFFlat (helper for future cloner support).
    void copyFrom(const faiss::IndexIVFFlat* index);

    /// Copy to a CPU IndexIVFFlat.
    void copyTo(faiss::IndexIVFFlat* index) const;

    /// Reconstruct a single stored vector by internal key.
    void reconstruct(idx_t key, float* recons) const override;

    /// Reconstruct n contiguous stored vectors starting at i0.
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    /// Re-upload coarse quantizer centroids to GPU after external changes.
    void updateQuantizer();

    /// Return the vector indices in inverted list `listId`.
    std::vector<idx_t> getListIndices(idx_t listId) const;

    /// Return raw vector data from inverted list `listId`.
    std::vector<float> getListVectorData(idx_t listId) const;

    /// Release unused GPU memory.
    void reclaimMemory();

    /// Pre-allocate GPU storage for the given total number of vectors.
    void reserveMemory(idx_t numVecs);

    /// Accessors (needed by cloner and tests).
    idx_t nlist() const;
    size_t nprobe() const;
    bool interleavedLayout() const;
    faiss::gpu::IndicesOptions indicesOptions() const;
    AppendDebugStats appendDebugStats() const;
    void resetAppendDebugStats();

private:
    std::unique_ptr<faiss::IndexIVFFlat> cpuIndex_;
    std::unique_ptr<MetalIVFFlatImpl> gpuIvf_;
    faiss::gpu::IndicesOptions indicesOptions_;
    bool interleavedLayout_;

    // Persistent search buffers — allocated once, grown lazily.
    // Declared mutable so search() (const) can resize them.
    mutable id<MTLBuffer> searchQueriesBuf_ = nil;
    mutable id<MTLBuffer> searchCoarseBuf_  = nil;
    mutable id<MTLBuffer> searchOutDistBuf_ = nil;
    mutable id<MTLBuffer> searchOutIdxBuf_  = nil;
    mutable size_t searchQueriesCap_ = 0; // bytes
    mutable size_t searchCoarseCap_  = 0;
    mutable size_t searchOutDistCap_ = 0;
    mutable size_t searchOutIdxCap_  = 0;
    mutable id<MTLBuffer> searchPerListDistBuf_ = nil;
    mutable id<MTLBuffer> searchPerListIdxBuf_  = nil;
    mutable size_t searchPerListDistCap_ = 0;
    mutable size_t searchPerListIdxCap_  = 0;

    // GPU coarse quantizer buffers (cached, rebuilt on train)
    mutable id<MTLBuffer> centroidBuf_          = nil;
    mutable id<MTLBuffer> centroidNormsBuf_     = nil; // pre-computed ||c||²
    mutable id<MTLBuffer> coarseOutDistBuf_     = nil;
    mutable id<MTLBuffer> coarseOutIdxBuf_      = nil;
    mutable size_t coarseOutDistCap_  = 0;
    mutable size_t coarseOutIdxCap_   = 0;
    mutable id<MTLBuffer> distMatrixBuf_        = nil;
    mutable size_t distMatrixCap_     = 0;

    /// Ensures buf is at least `needed` bytes, reallocating if necessary.
    void ensureSearchBuf_(
            id<MTLBuffer>& buf,
            size_t& cap,
            size_t needed) const;

    /// (Re)uploads quantizer centroids to centroidBuf_.
    void uploadCentroids_() const;
};

} // namespace gpu_metal
} // namespace faiss
