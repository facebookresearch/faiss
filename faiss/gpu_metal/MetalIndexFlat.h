// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Objective-C++ header (uses Metal types).
 */

#pragma once

#import <Metal/Metal.h>

#include <faiss/Index.h>
#include <faiss/gpu_metal/MetalIndex.h>

namespace faiss {
struct IndexFlat;
}
#include <memory>

namespace faiss {
namespace gpu_metal {

/// Flat index that stores vectors in an MTLBuffer. Supports L2 and inner
/// product. Search runs on GPU via Metal compute (distance + top-k kernels).
class MetalIndexFlat : public MetalIndex {
   public:
    MetalIndexFlat(
            std::shared_ptr<MetalResources> resources,
            int dims,
            faiss::MetricType metric,
            float metricArg = 0.0f,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndexFlat() override;

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

    /// Copy vectors to a CPU IndexFlat (e.g. for index_metal_gpu_to_cpu).
    void copyTo(::faiss::IndexFlat* index) const;

   private:
    /// Ensures vector buffer can hold at least \p newNtotal vectors; grows
    /// buffer if necessary.
    void ensureCapacity(idx_t newNtotal);

    /// Vector storage (row-major, ntotal * d floats). Nil when empty.
    id<MTLBuffer> vectorsBuffer_;
    /// Capacity of vectorsBuffer_ in number of vectors (0 if buffer is nil).
    size_t capacityVecs_;
};

} // namespace gpu_metal
} // namespace faiss
