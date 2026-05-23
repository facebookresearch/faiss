// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexFlat.h"
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <cstring>
#import "MetalFlatKernels.h"

namespace faiss {
namespace gpu_metal {

MetalIndexFlat::MetalIndexFlat(
        std::shared_ptr<MetalResources> resources,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, metricArg, config),
          vectorsBuffer_(nil),
          capacityVecs_(0) {
    FAISS_THROW_IF_NOT(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);
}

MetalIndexFlat::~MetalIndexFlat() {
    if (vectorsBuffer_ != nil) {
        resources_->deallocBuffer(vectorsBuffer_, MetalAllocType::FlatData);
        vectorsBuffer_ = nil;
    }
    capacityVecs_ = 0;
}

void MetalIndexFlat::ensureCapacity(idx_t newNtotal) {
    if (newNtotal <= (idx_t)capacityVecs_) {
        return;
    }
    size_t newCap = (capacityVecs_ == 0)
            ? (size_t)newNtotal
            : std::max((size_t)newNtotal, capacityVecs_ * 2);
    size_t newSize = newCap * (size_t)d * sizeof(float);
    id<MTLBuffer> newBuf =
            resources_->allocBuffer(newSize, MetalAllocType::FlatData);
    FAISS_THROW_IF_NOT_MSG(
            newBuf != nil, "MetalIndexFlat: failed to allocate buffer");
    if (ntotal > 0 && vectorsBuffer_ != nil) {
        std::memcpy(
                [newBuf contents],
                [vectorsBuffer_ contents],
                (size_t)ntotal * (size_t)d * sizeof(float));
        resources_->deallocBuffer(vectorsBuffer_, MetalAllocType::FlatData);
    }
    vectorsBuffer_ = newBuf;
    capacityVecs_ = newCap;
}

void MetalIndexFlat::add(idx_t n, const float* x) {
    if (n == 0) {
        return;
    }
    ensureCapacity(ntotal + n);
    float* ptr = (float*)[vectorsBuffer_ contents];
    std::memcpy(
            ptr + (size_t)ntotal * (size_t)d,
            x,
            (size_t)n * (size_t)d * sizeof(float));
    ntotal += n;
}

void MetalIndexFlat::add_with_ids(
        idx_t /*n*/,
        const float* /*x*/,
        const idx_t* /*xids*/) {
    FAISS_THROW_MSG("add_with_ids not supported");
}

void MetalIndexFlat::reset() {
    if (vectorsBuffer_ != nil) {
        resources_->deallocBuffer(vectorsBuffer_, MetalAllocType::FlatData);
        vectorsBuffer_ = nil;
    }
    capacityVecs_ = 0;
    ntotal = 0;
}

void MetalIndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    (void)params;
    FAISS_THROW_IF_NOT(k > 0);
    if (ntotal == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i] = -1;
        }
        return;
    }
    const int maxK = getMetalFlatSearchMaxK();
    FAISS_THROW_IF_NOT_MSG(
            k <= maxK,
            "MetalIndexFlat: k exceeds GPU limit (see getMetalFlatSearchMaxK())");

    id<MTLDevice> device = resources_->getDevice();
    id<MTLCommandQueue> queue = resources_->getCommandQueue();
    if (!device || !queue) {
        FAISS_THROW_MSG("MetalIndexFlat: device or queue not available");
    }

    const size_t queryBytes = (size_t)n * (size_t)d * sizeof(float);
    const size_t outDistBytes = (size_t)n * (size_t)k * sizeof(float);
    const size_t outIdxBytes = (size_t)n * (size_t)k * sizeof(int32_t);

    id<MTLBuffer> queryBuf = resources_->allocBuffer(
            queryBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> outDistBuf = resources_->allocBuffer(
            outDistBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> outIdxBuf = resources_->allocBuffer(
            outIdxBytes, MetalAllocType::TemporaryMemoryBuffer);
    FAISS_THROW_IF_NOT_MSG(
            queryBuf && outDistBuf && outIdxBuf,
            "MetalIndexFlat: failed to allocate temp buffers");

    std::memcpy([queryBuf contents], x, queryBytes);

    const bool isL2 = (metric_type == METRIC_L2);
    const bool ok = runFlatSearchGPU(
            device,
            queue,
            queryBuf,
            vectorsBuffer_,
            (int)n,
            (int)ntotal,
            d,
            (int)k,
            isL2,
            outDistBuf,
            outIdxBuf);

    resources_->deallocBuffer(queryBuf, MetalAllocType::TemporaryMemoryBuffer);
    if (!ok) {
        resources_->deallocBuffer(
                outDistBuf, MetalAllocType::TemporaryMemoryBuffer);
        resources_->deallocBuffer(
                outIdxBuf, MetalAllocType::TemporaryMemoryBuffer);
        FAISS_THROW_MSG(
                "MetalIndexFlat: GPU search failed (pipeline or dispatch error)");
    }

    std::memcpy(distances, [outDistBuf contents], outDistBytes);
    const int32_t* idxPtr = (const int32_t*)[outIdxBuf contents];
    for (idx_t i = 0; i < n * k; ++i) {
        int32_t idx = idxPtr[i];
        labels[i] = (idx >= 0 && idx < (int32_t)ntotal) ? (idx_t)idx : -1;
    }

    resources_->deallocBuffer(
            outDistBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources_->deallocBuffer(outIdxBuf, MetalAllocType::TemporaryMemoryBuffer);
}

void MetalIndexFlat::copyTo(faiss::IndexFlat* index) const {
    FAISS_THROW_IF_NOT(index != nullptr);
    FAISS_THROW_IF_NOT(index->d == d);
    FAISS_THROW_IF_NOT(index->metric_type == metric_type);
    if (ntotal == 0 || vectorsBuffer_ == nil) {
        return;
    }
    std::vector<float> host((size_t)ntotal * (size_t)d);
    std::memcpy(
            host.data(),
            [vectorsBuffer_ contents],
            host.size() * sizeof(float));
    index->add(ntotal, host.data());
}

} // namespace gpu_metal
} // namespace faiss
