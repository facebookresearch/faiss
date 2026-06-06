// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIVFPQ.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu_metal {

MetalIVFPQImpl::MetalIVFPQImpl(
        std::shared_ptr<MetalResources> resources,
        int dim,
        idx_t nlist,
        int numSubQuantizers,
        int bitsPerCode,
        faiss::MetricType metric,
        float metricArg)
        : resources_(std::move(resources)),
          dim_(dim),
          nlist_(nlist),
          M_(numSubQuantizers),
          bitsPerCode_(bitsPerCode),
          ksub_(1 << bitsPerCode),
          dsub_(dim / numSubQuantizers),
          codeSize_((size_t)numSubQuantizers),
          metric_type_(metric),
          metric_arg_(metricArg),
          listLength_(nlist_, 0),
          listOffset_(nlist_, 0),
          totalVecs_(0),
          codesBuffer_(nil),
          idsBuffer_(nil),
          listOffsetBuf_(nil),
          listLengthBuf_(nil),
          pqCentroidsBuf_(nil) {
    FAISS_THROW_IF_NOT(dim_ > 0);
    FAISS_THROW_IF_NOT(nlist_ >= 0);
    FAISS_THROW_IF_NOT(M_ > 0);
    FAISS_THROW_IF_NOT(bitsPerCode_ == 8);
    FAISS_THROW_IF_NOT(dim_ % M_ == 0);
}

MetalIVFPQImpl::~MetalIVFPQImpl() {
    reset();
}

void MetalIVFPQImpl::reset() {
    hostCodes_.clear();
    hostIds_.clear();
    totalVecs_ = 0;
    std::fill(listLength_.begin(), listLength_.end(), 0);
    std::fill(listOffset_.begin(), listOffset_.end(), 0);

    auto dealloc = [&](id<MTLBuffer>& buf) {
        if (buf != nil) {
            resources_->deallocBuffer(buf, MetalAllocType::IVFLists);
            buf = nil;
        }
    };
    dealloc(codesBuffer_);
    dealloc(idsBuffer_);
    dealloc(listOffsetBuf_);
    dealloc(listLengthBuf_);
}

void MetalIVFPQImpl::reserveMemory(idx_t totalVecs) {
    if (totalVecs <= 0)
        return;
    size_t t = (size_t)totalVecs;
    if (t <= totalVecs_)
        return;
    hostCodes_.reserve(t * codeSize_);
    hostIds_.reserve(t);
}

void MetalIVFPQImpl::appendCodes(
        idx_t n,
        const uint8_t* codes,
        const idx_t* list_nos,
        const idx_t* xids) {
    if (n == 0)
        return;
    FAISS_THROW_IF_NOT(list_nos != nullptr);
    FAISS_THROW_IF_NOT(codes != nullptr);

    std::vector<size_t> addPerList(nlist_, 0);
    for (idx_t i = 0; i < n; ++i) {
        idx_t list = list_nos[i];
        if (list >= 0 && list < nlist_) {
            addPerList[(size_t)list]++;
        }
    }

    size_t batchNew = 0;
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        batchNew += addPerList[l];
    }
    if (batchNew == 0)
        return;

    std::vector<size_t> newLength(nlist_);
    std::vector<size_t> newOffset(nlist_);
    size_t prefix = 0;
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        newOffset[l] = prefix;
        newLength[l] = listLength_[l] + addPerList[l];
        prefix += newLength[l];
    }
    size_t newTotalVecs = prefix;

    std::vector<uint8_t> newCodes(newTotalVecs * codeSize_);
    std::vector<idx_t> newIds(newTotalVecs);

    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        size_t oldLen = listLength_[l];
        if (oldLen == 0)
            continue;
        size_t oldOff = listOffset_[l];
        size_t newOff = newOffset[l];
        std::memcpy(
                newCodes.data() + newOff * codeSize_,
                hostCodes_.data() + oldOff * codeSize_,
                oldLen * codeSize_);
        std::memcpy(
                newIds.data() + newOff,
                hostIds_.data() + oldOff,
                oldLen * sizeof(idx_t));
    }

    std::vector<size_t> curPerList(nlist_);
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        curPerList[l] = listLength_[l];
    }
    for (idx_t i = 0; i < n; ++i) {
        idx_t list = list_nos[i];
        if (list < 0 || list >= nlist_)
            continue;
        size_t l = (size_t)list;
        size_t dstIndex = newOffset[l] + curPerList[l];
        curPerList[l]++;

        std::memcpy(
                newCodes.data() + dstIndex * codeSize_,
                codes + (size_t)i * codeSize_,
                codeSize_);
        idx_t id = xids ? xids[i] : (idx_t)(totalVecs_ + (size_t)i);
        newIds[dstIndex] = id;
    }

    hostCodes_.swap(newCodes);
    hostIds_.swap(newIds);
    listLength_.swap(newLength);
    listOffset_.swap(newOffset);
    totalVecs_ = newTotalVecs;
    uploadToGpu();
}

void MetalIVFPQImpl::setPQCentroids(const float* centroids) {
    FAISS_THROW_IF_NOT(centroids != nullptr);
    size_t bytes = (size_t)M_ * (size_t)ksub_ * (size_t)dsub_ * sizeof(float);
    if (pqCentroidsBuf_ != nil) {
        resources_->deallocBuffer(pqCentroidsBuf_, MetalAllocType::IVFLists);
        pqCentroidsBuf_ = nil;
    }
    pqCentroidsBuf_ = resources_->allocBuffer(bytes, MetalAllocType::IVFLists);
    FAISS_THROW_IF_NOT_MSG(
            pqCentroidsBuf_, "Failed to allocate PQ centroids buffer");
    std::memcpy([pqCentroidsBuf_ contents], centroids, bytes);
}

void MetalIVFPQImpl::uploadToGpu() {
    if (!resources_ || !resources_->isAvailable())
        return;

    size_t codesBytes = hostCodes_.size();
    size_t idsBytes = hostIds_.size() * sizeof(idx_t);
    size_t metaBytes = (size_t)nlist_ * sizeof(uint32_t);

    auto dealloc = [&](id<MTLBuffer>& buf) {
        if (buf != nil) {
            resources_->deallocBuffer(buf, MetalAllocType::IVFLists);
            buf = nil;
        }
    };
    dealloc(codesBuffer_);
    dealloc(idsBuffer_);
    dealloc(listOffsetBuf_);
    dealloc(listLengthBuf_);

    if (metaBytes > 0) {
        listOffsetBuf_ =
                resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);
        listLengthBuf_ =
                resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);
        FAISS_THROW_IF_NOT_MSG(
                listOffsetBuf_ && listLengthBuf_,
                "MetalIVFPQImpl: failed to allocate metadata GPU buffers");

        auto* offPtr = reinterpret_cast<uint32_t*>([listOffsetBuf_ contents]);
        auto* lenPtr = reinterpret_cast<uint32_t*>([listLengthBuf_ contents]);
        for (size_t i = 0; i < (size_t)nlist_; ++i) {
            offPtr[i] = (uint32_t)listOffset_[i];
            lenPtr[i] = (uint32_t)listLength_[i];
        }
    }

    if (codesBytes == 0 || idsBytes == 0)
        return;

    codesBuffer_ =
            resources_->allocBuffer(codesBytes, MetalAllocType::IVFLists);
    idsBuffer_ = resources_->allocBuffer(idsBytes, MetalAllocType::IVFLists);
    FAISS_THROW_IF_NOT_MSG(
            codesBuffer_ && idsBuffer_,
            "MetalIVFPQImpl: failed to allocate IVF Metal buffers");

    std::memcpy([codesBuffer_ contents], hostCodes_.data(), codesBytes);
    std::memcpy([idsBuffer_ contents], hostIds_.data(), idsBytes);
}

} // namespace gpu_metal
} // namespace faiss
