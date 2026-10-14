// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexBinaryFlat.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <memory>

namespace faiss {
namespace gpu_metal {

/// Exhaustive Hamming search over GPU-resident binary vectors.
/// Every positive dimension divisible by 8 is supported. GPU search is exact
/// for k <= 512, with ties ordered by increasing label. Larger k, selectors,
/// and counts exceeding uint32_t use the CPU copy. GPU failures throw rather
/// than silently falling back. Range search and remove_ids are not supported.
/// As with other Metal indexes, callers must serialize mutation and search.
class MetalIndexBinaryFlat : public faiss::IndexBinary {
   public:
    static constexpr int kMaxGpuK = 512;

    MetalIndexBinaryFlat(std::shared_ptr<MetalResources> resources, int d);
    MetalIndexBinaryFlat(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexBinaryFlat* index);
    ~MetalIndexBinaryFlat() override;

    MetalIndexBinaryFlat(const MetalIndexBinaryFlat&) = delete;
    MetalIndexBinaryFlat& operator=(const MetalIndexBinaryFlat&) = delete;

    void add(idx_t n, const uint8_t* x) override;
    void reset() override;
    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    void reconstruct(idx_t key, uint8_t* recons) const override;

    void copyFrom(const faiss::IndexBinaryFlat* index);
    void copyTo(faiss::IndexBinaryFlat* index) const;

    std::shared_ptr<MetalResources> getResources() const {
        return resources_;
    }

   private:
    std::shared_ptr<MetalResources> resources_;
    faiss::IndexBinaryFlat cpuIndex_;
    id<MTLBuffer> vectorsBuf_ = nil;
    // Search is synchronous and callers serialize access, so scratch buffers
    // can be reused after waitUntilCompleted. ARC releases them on destruction.
    mutable id<MTLBuffer> queryBuf_ = nil;
    mutable id<MTLBuffer> distanceBuf_ = nil;
    mutable id<MTLBuffer> labelBuf_ = nil;

    id<MTLBuffer> upload(const uint8_t* data, size_t bytes) const;
};

} // namespace gpu_metal
} // namespace faiss
