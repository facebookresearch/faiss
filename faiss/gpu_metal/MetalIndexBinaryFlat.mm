// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexBinaryFlat.h"
#import "MetalKernels.h"

#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace faiss {
namespace gpu_metal {
namespace {

// Metal requires endEncoding even when pipeline lookup throws.
struct ComputeEncoderScope {
    id<MTLComputeCommandEncoder> encoder;
    ~ComputeEncoderScope() {
        [encoder endEncoding];
    }
};

size_t checkedBytes(idx_t count, size_t stride) {
    FAISS_THROW_IF_NOT(count >= 0);
    FAISS_THROW_IF_NOT(
            uint64_t(count) <= std::numeric_limits<size_t>::max() / stride);
    return size_t(count) * stride;
}

int checkedDimension(const faiss::IndexBinaryFlat* index) {
    FAISS_THROW_IF_NOT(index);
    FAISS_THROW_IF_NOT(index->d > 0 && index->d % 8 == 0);
    return index->d;
}

} // namespace

MetalIndexBinaryFlat::MetalIndexBinaryFlat(
        std::shared_ptr<MetalResources> resources,
        int dim)
        : IndexBinary(dim), resources_(std::move(resources)), cpuIndex_(dim) {
    FAISS_THROW_IF_NOT(dim > 0 && dim % 8 == 0);
    FAISS_THROW_IF_NOT(resources_ && resources_->isAvailable());
    is_trained = true;
}

MetalIndexBinaryFlat::MetalIndexBinaryFlat(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexBinaryFlat* index)
        : MetalIndexBinaryFlat(std::move(resources), checkedDimension(index)) {
    copyFrom(index);
}

MetalIndexBinaryFlat::~MetalIndexBinaryFlat() = default;

id<MTLBuffer> MetalIndexBinaryFlat::upload(const uint8_t* data, size_t bytes)
        const {
    if (bytes == 0) {
        return nil;
    }
    auto device = resources_->getDevice();
    FAISS_THROW_IF_NOT(bytes <= device.maxBufferLength);
    id<MTLBuffer> buffer =
            [device newBufferWithBytes:data
                                length:bytes
                               options:MTLResourceStorageModeShared];
    FAISS_THROW_IF_NOT_MSG(buffer, "MetalIndexBinaryFlat: allocation failed");
    return buffer;
}

void MetalIndexBinaryFlat::add(idx_t n, const uint8_t* x) {
    FAISS_THROW_IF_NOT(n >= 0);
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(x);
    FAISS_THROW_IF_NOT(n <= std::numeric_limits<idx_t>::max() - ntotal);
    size_t bytes = checkedBytes(ntotal + n, code_size);
    auto device = resources_->getDevice();
    size_t limit = device.maxBufferLength;
    FAISS_THROW_IF_NOT(bytes <= limit);
    // Grow geometrically, capped before doubling to avoid overflow. Keep the
    // replacement local until the CPU append succeeds, so allocation failure
    // leaves both published copies and ntotal unchanged.
    id<MTLBuffer> buffer = vectorsBuf_;
    size_t capacity = buffer.length;
    size_t oldBytes = checkedBytes(ntotal, code_size);
    if (bytes > capacity) {
        size_t doubled = capacity > limit / 2 ? limit : capacity * 2;
        size_t newCapacity = std::max(bytes, doubled);
        buffer = [device newBufferWithLength:newCapacity
                                     options:MTLResourceStorageModeShared];
        FAISS_THROW_IF_NOT_MSG(
                buffer, "MetalIndexBinaryFlat: allocation failed");
        if (oldBytes) {
            std::memcpy(buffer.contents, vectorsBuf_.contents, oldBytes);
        }
    }
    cpuIndex_.add(n, x);
    // The CPU append is the last throwing operation. Copy only the new codes;
    // use the owned CPU bytes in case its vector reallocation invalidated x.
    std::memcpy(
            static_cast<uint8_t*>(buffer.contents) + oldBytes,
            cpuIndex_.xb.data() + oldBytes,
            bytes - oldBytes);
    vectorsBuf_ = buffer;
    ntotal = cpuIndex_.ntotal;
}

void MetalIndexBinaryFlat::reset() {
    cpuIndex_.reset();
    vectorsBuf_ = nil;
    ntotal = 0;
}

void MetalIndexBinaryFlat::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(n >= 0 && k > 0);
    size_t count = checkedBytes(n, size_t(k));
    checkedBytes(n, checkedBytes(k, sizeof(idx_t)));
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(distances && labels);
    if (ntotal == 0) {
        std::fill_n(distances, count, std::numeric_limits<int32_t>::max());
        std::fill_n(labels, count, idx_t(-1));
        return;
    }
    FAISS_THROW_IF_NOT(x);
    checkedBytes(n, code_size);
    if (k > kMaxGpuK || (params && params->sel) ||
        uint64_t(n) > std::numeric_limits<uint32_t>::max() ||
        uint64_t(ntotal) > std::numeric_limits<uint32_t>::max()) {
        cpuIndex_.search(n, x, k, distances, labels, params);
        return;
    }

    @autoreleasepool {
        auto device = resources_->getDevice();
        id<MTLBuffer> queries = upload(x, checkedBytes(n, code_size));
        size_t distBytes = checkedBytes(n, checkedBytes(k, sizeof(int32_t)));
        size_t labelBytes = checkedBytes(n, checkedBytes(k, sizeof(idx_t)));
        FAISS_THROW_IF_NOT(labelBytes <= device.maxBufferLength);
        id<MTLBuffer> outDist =
                [device newBufferWithLength:distBytes
                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIdx =
                [device newBufferWithLength:labelBytes
                                    options:MTLResourceStorageModeShared];
        FAISS_THROW_IF_NOT(outDist && outIdx && vectorsBuf_);
        auto& kernels = getMetalKernels(device);
        FAISS_THROW_IF_NOT_MSG(
                kernels.isValid(), "Metal Hamming library unavailable");
        id<MTLCommandBuffer> command =
                [resources_->getCommandQueue() commandBuffer];
        FAISS_THROW_IF_NOT(command);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        FAISS_THROW_IF_NOT(encoder);
        {
            ComputeEncoderScope scope{encoder};
            kernels.encodeHammingDistanceTopK(
                    encoder,
                    queries,
                    vectorsBuf_,
                    outDist,
                    outIdx,
                    uint32_t(n),
                    uint32_t(ntotal),
                    uint32_t(code_size),
                    uint32_t(k));
        }
        [command commit];
        [command waitUntilCompleted];
        FAISS_THROW_IF_NOT_MSG(
                command.status == MTLCommandBufferStatusCompleted,
                "MetalIndexBinaryFlat: GPU Hamming search failed");
        static_assert(sizeof(idx_t) == sizeof(int64_t));
        std::memcpy(distances, outDist.contents, distBytes);
        std::memcpy(labels, outIdx.contents, labelBytes);
    }
}

void MetalIndexBinaryFlat::reconstruct(idx_t key, uint8_t* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal && recons);
    cpuIndex_.reconstruct(key, recons);
}

void MetalIndexBinaryFlat::copyFrom(const faiss::IndexBinaryFlat* index) {
    checkedDimension(index);
    FAISS_THROW_IF_NOT(index->code_size == index->d / 8);
    FAISS_THROW_IF_NOT(index->is_trained && index->metric_type == METRIC_L2);
    FAISS_THROW_IF_NOT(index->query_batch_size > 0);
    size_t bytes = checkedBytes(index->ntotal, index->code_size);
    FAISS_THROW_IF_NOT(index->xb.size() == bytes);
    faiss::IndexBinaryFlat replacement(*index);
    // MaybeOwnedVector copies can retain a borrowed view. Always own the CPU
    // backing bytes so reconstruction/fallback survive the source's lifetime.
    replacement.xb = faiss::MaybeOwnedVector<uint8_t>(bytes);
    if (bytes) {
        std::memcpy(replacement.xb.data(), index->xb.data(), bytes);
    }
    id<MTLBuffer> buffer = upload(replacement.xb.data(), bytes);
    cpuIndex_ = std::move(replacement);
    vectorsBuf_ = buffer;
    d = cpuIndex_.d;
    code_size = cpuIndex_.code_size;
    ntotal = cpuIndex_.ntotal;
    is_trained = cpuIndex_.is_trained;
    metric_type = cpuIndex_.metric_type;
    verbose = cpuIndex_.verbose;
}

void MetalIndexBinaryFlat::copyTo(faiss::IndexBinaryFlat* index) const {
    FAISS_THROW_IF_NOT(index);
    *index = cpuIndex_;
    index->verbose = verbose;
}

} // namespace gpu_metal
} // namespace faiss
