// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIVFFlat.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/DirectMap.h>

namespace faiss {
namespace gpu_metal {

namespace {
inline uint32_t checkedToU32(size_t v, const char* what) {
    if (v > (size_t)std::numeric_limits<uint32_t>::max()) {
        FAISS_THROW_MSG(what);
    }
    return static_cast<uint32_t>(v);
}

constexpr size_t kDefaultMinTailShrinkVecs = 8192;
constexpr size_t kMaxU32SizeT =
        (size_t)std::numeric_limits<uint32_t>::max();

size_t getTailShrinkMinVecs() {
    const char* env = std::getenv("FAISS_METAL_IVF_TAIL_SHRINK_MIN_VECS");
    if (!env || env[0] == '\0') {
        return kDefaultMinTailShrinkVecs;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(env, &end, 10);
    if (end == env) {
        return kDefaultMinTailShrinkVecs;
    }
    return (size_t)v;
}
} // namespace

MetalIVFFlatImpl::MetalIVFFlatImpl(
        std::shared_ptr<MetalResources> resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        faiss::gpu::IndicesOptions indicesOptions,
        bool interleavedLayout)
        : resources_(std::move(resources)),
          dim_(dim),
          nlist_(nlist),
          metric_type_(metric),
          metric_arg_(metricArg),
          indicesOptions_(indicesOptions),
          interleavedLayout_(interleavedLayout),
          listLength_(nlist_, 0),
          listOffset_(nlist_, 0),
          listCapacity_(nlist_, 0),
          totalVecs_(0),
          totalCapacityVecs_(0),
          codesBuffer_(nil),
          idsBuffer_(nil),
          listOffsetBuf_(nil),
          listLengthBuf_(nil),
          interleavedCodesBuf_(nil),
          interleavedCodesOffsetBuf_(nil),
          interleavedDirty_(true) {
    FAISS_THROW_IF_NOT(dim_ > 0);
    FAISS_THROW_IF_NOT(nlist_ >= 0);
}

MetalIVFFlatImpl::~MetalIVFFlatImpl() {
    reset();
}

void MetalIVFFlatImpl::reset() {
    hostCodes_.clear();
    hostIds_.clear();
    freeSegments_.clear();
    appendStats_ = AppendDebugStats{};
    totalVecs_ = 0;
    totalCapacityVecs_ = 0;

    std::fill(listLength_.begin(), listLength_.end(), 0);
    std::fill(listOffset_.begin(), listOffset_.end(), 0);
    std::fill(listCapacity_.begin(), listCapacity_.end(), 0);

    if (codesBuffer_ != nil) {
        resources_->deallocBuffer(codesBuffer_, MetalAllocType::IVFLists);
        codesBuffer_ = nil;
    }
    if (idsBuffer_ != nil) {
        resources_->deallocBuffer(idsBuffer_, MetalAllocType::IVFLists);
        idsBuffer_ = nil;
    }
    if (listOffsetBuf_ != nil) {
        resources_->deallocBuffer(listOffsetBuf_, MetalAllocType::IVFLists);
        listOffsetBuf_ = nil;
    }
    if (listLengthBuf_ != nil) {
        resources_->deallocBuffer(listLengthBuf_, MetalAllocType::IVFLists);
        listLengthBuf_ = nil;
    }
    if (interleavedCodesBuf_ != nil) {
        resources_->deallocBuffer(interleavedCodesBuf_, MetalAllocType::IVFLists);
        interleavedCodesBuf_ = nil;
    }
    if (interleavedCodesOffsetBuf_ != nil) {
        resources_->deallocBuffer(interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
        interleavedCodesOffsetBuf_ = nil;
    }
    interleavedDirty_ = true;
}

void MetalIVFFlatImpl::reserveMemory(idx_t totalVecs) {
    if (totalVecs <= 0) {
        return;
    }
    size_t t = (size_t)totalVecs;
    FAISS_THROW_IF_NOT_MSG(
            t <= kMaxU32SizeT,
            "MetalIVFFlatImpl::reserveMemory exceeds uint32 list-offset range");
    if (t <= totalCapacityVecs_) {
        return;
    }
    const size_t oldCap = totalCapacityVecs_;
    const size_t extra = t - oldCap;
    hostCodes_.resize(t * (size_t)dim_, 0.0f);
    hostIds_.resize(t, (idx_t)-1);
    totalCapacityVecs_ = t;
    // Add reserved tail as a reusable free segment for future list growth.
    freeSegment_(oldCap, extra, false);

    // Materialize GPU allocations up-front so future appends can avoid
    // reallocating until reserved capacity is exceeded.
    std::vector<size_t> oldLength((size_t)nlist_, 0);
    std::vector<size_t> addPerList((size_t)nlist_, 0);
    std::vector<uint8_t> movedLists((size_t)nlist_, 0);
    uploadToGpu_(oldLength, addPerList, movedLists, true);
}

void MetalIVFFlatImpl::appendVectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        const idx_t* xids) {
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(list_nos != nullptr);

    // Count how many vectors go to each list.
    std::vector<size_t> addPerList(nlist_, 0);
    for (idx_t i = 0; i < n; ++i) {
        idx_t list = list_nos[i];
        if (list < 0 || list >= nlist_) {
            continue;
        }
        addPerList[(size_t)list]++;
    }

    // Early-out: nothing to append.
    size_t batchNew = 0;
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        batchNew += addPerList[l];
    }
    if (batchNew == 0) {
        return;
    }

    const std::vector<size_t> oldLength = listLength_;
    std::vector<uint8_t> movedLists((size_t)nlist_, 0);
    bool forceFullUpload = ensureCapacityForAppend_(addPerList, &movedLists);

    for (idx_t i = 0; i < n; ++i) {
        idx_t list = list_nos[i];
        if (list < 0 || list >= nlist_) {
            continue;
        }
        size_t l = (size_t)list;
        size_t dstIndex = listOffset_[l] + listLength_[l];
        size_t listOffset = listLength_[l];
        listLength_[l]++;

        // Copy vector
        const float* xi = x + (size_t)i * (size_t)dim_;
        std::memcpy(
                hostCodes_.data() + dstIndex * (size_t)dim_,
                xi,
                (size_t)dim_ * sizeof(float));

        // Copy id
        idx_t id = -1;
        if (indicesOptions_ == faiss::gpu::INDICES_CPU ||
            indicesOptions_ == faiss::gpu::INDICES_IVF) {
            id = (idx_t)faiss::lo_build((uint64_t)l, (uint64_t)listOffset);
        } else {
            id = xids ? xids[i] : (idx_t)(totalVecs_ + (size_t)(i));
            if (indicesOptions_ == faiss::gpu::INDICES_32_BIT) {
                FAISS_THROW_IF_NOT_MSG(
                        id >= (idx_t)std::numeric_limits<int32_t>::min() &&
                                id <= (idx_t)std::numeric_limits<int32_t>::max(),
                        "MetalIVFFlatImpl: id out of int32 range");
                id = (idx_t)(int32_t)id;
            }
        }
        hostIds_[dstIndex] = id;
    }

    totalVecs_ += batchNew;
    uploadToGpu_(oldLength, addPerList, movedLists, forceFullUpload);
}

bool MetalIVFFlatImpl::ensureCapacityForAppend_(
        const std::vector<size_t>& addPerList,
        std::vector<uint8_t>* movedLists) {
    if (movedLists) {
        movedLists->assign((size_t)nlist_, 0);
    }

    std::vector<size_t> newCap((size_t)nlist_, 0);
    bool anyMoved = false;

    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        size_t need = listLength_[l] + addPerList[l];
        if (need <= listCapacity_[l]) {
            continue;
        }
        size_t cap = std::max<size_t>(1, listCapacity_[l]);
        while (cap < need) {
            cap *= 2;
        }
        FAISS_THROW_IF_NOT_MSG(
                cap <= kMaxU32SizeT,
                "MetalIVFFlatImpl: list capacity exceeds uint32 range");
        newCap[l] = cap;
        anyMoved = true;
        if (movedLists) {
            (*movedLists)[l] = 1;
        }
    }

    if (!anyMoved) {
        return false;
    }
    appendStats_.relayoutEvents += 1;

    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        if (newCap[l] == 0) {
            continue;
        }

        size_t len = listLength_[l];
        size_t oldOff = listOffset_[l];
        size_t oldCap = listCapacity_[l];
        size_t newOff = allocSegment_(newCap[l]);
        appendStats_.movedLists += 1;
        appendStats_.movedVectors += len;

        if (len > 0) {
            std::memcpy(
                    hostCodes_.data() + newOff * (size_t)dim_,
                    hostCodes_.data() + oldOff * (size_t)dim_,
                    len * (size_t)dim_ * sizeof(float));
            std::memcpy(
                    hostIds_.data() + newOff,
                    hostIds_.data() + oldOff,
                    len * sizeof(idx_t));
        }

        listOffset_[l] = newOff;
        listCapacity_[l] = newCap[l];
        freeSegment_(oldOff, oldCap, true);
    }

    return true;
}

size_t MetalIVFFlatImpl::allocSegment_(size_t length) {
    if (length == 0) {
        return 0;
    }

    size_t bestIdx = (size_t)-1;
    size_t bestLen = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < freeSegments_.size(); ++i) {
        const auto& seg = freeSegments_[i];
        if (seg.length < length) {
            continue;
        }
        if (seg.length < bestLen) {
            bestLen = seg.length;
            bestIdx = i;
        }
    }

    if (bestIdx != (size_t)-1) {
        auto seg = freeSegments_[bestIdx];
        size_t out = seg.offset;
        if (seg.length == length) {
            freeSegments_.erase(freeSegments_.begin() + (long)bestIdx);
        } else {
            freeSegments_[bestIdx].offset += length;
            freeSegments_[bestIdx].length -= length;
        }
        appendStats_.reusedSegmentAllocs += 1;
        appendStats_.reusedCapacityVecs += length;
        return out;
    }

    size_t out = totalCapacityVecs_;
    FAISS_THROW_IF_NOT_MSG(
            out <= kMaxU32SizeT && length <= (kMaxU32SizeT - out),
            "MetalIVFFlatImpl: total capacity exceeds uint32 list-offset range");
    totalCapacityVecs_ += length;
    hostCodes_.resize(totalCapacityVecs_ * (size_t)dim_, 0.0f);
    hostIds_.resize(totalCapacityVecs_, (idx_t)-1);
    appendStats_.tailSegmentAllocs += 1;
    appendStats_.tailCapacityVecs += length;
    return out;
}

void MetalIVFFlatImpl::freeSegment_(
        size_t offset,
        size_t length,
        bool allowTailShrink) {
    if (length == 0) {
        return;
    }
    freeSegments_.push_back({offset, length});
    coalesceFreeSegments_();
    if (allowTailShrink) {
        tryShrinkTail_();
    }
}

void MetalIVFFlatImpl::coalesceFreeSegments_() {
    if (freeSegments_.empty()) {
        return;
    }
    std::sort(
            freeSegments_.begin(),
            freeSegments_.end(),
            [](const FreeSegment& a, const FreeSegment& b) {
                return a.offset < b.offset;
            });

    std::vector<FreeSegment> merged;
    merged.reserve(freeSegments_.size());
    merged.push_back(freeSegments_[0]);
    for (size_t i = 1; i < freeSegments_.size(); ++i) {
        auto& back = merged.back();
        const auto& cur = freeSegments_[i];
        if (back.offset + back.length == cur.offset) {
            back.length += cur.length;
        } else {
            merged.push_back(cur);
        }
    }
    freeSegments_.swap(merged);
}

void MetalIVFFlatImpl::tryShrinkTail_() {
    const size_t minTailShrinkVecs = getTailShrinkMinVecs();
    if (minTailShrinkVecs == 0) {
        return;
    }
    if (freeSegments_.empty()) {
        return;
    }
    while (!freeSegments_.empty()) {
        const auto& tail = freeSegments_.back();
        if (tail.offset + tail.length != totalCapacityVecs_) {
            break;
        }
        if (tail.length < minTailShrinkVecs) {
            break;
        }
        totalCapacityVecs_ = tail.offset;
        appendStats_.tailShrinkEvents += 1;
        appendStats_.tailShrunkVecs += tail.length;
        freeSegments_.pop_back();
    }
    hostCodes_.resize(totalCapacityVecs_ * (size_t)dim_);
    hostIds_.resize(totalCapacityVecs_);
}

void MetalIVFFlatImpl::uploadToGpu_(
        const std::vector<size_t>& oldLength,
        const std::vector<size_t>& addPerList,
        const std::vector<uint8_t>& movedLists,
        bool forceFullUpload) {
    if (!resources_ || !resources_->isAvailable()) {
        return;
    }

    size_t codesBytes = totalCapacityVecs_ * (size_t)dim_ * sizeof(float);
    size_t idsBytes = totalCapacityVecs_ * sizeof(idx_t);
    size_t metaBytes  = (size_t)nlist_ * sizeof(uint32_t);
    FAISS_THROW_IF_NOT_MSG(
            (size_t)nlist_ <= (size_t)std::numeric_limits<uint32_t>::max(),
            "MetalIVFFlatImpl: nlist exceeds uint32 metadata range");

    // Always update metadata buffers — they reflect current list layout.
    if (metaBytes > 0) {
        if (listOffsetBuf_ == nil || [listOffsetBuf_ length] < metaBytes) {
            if (listOffsetBuf_ != nil) {
                resources_->deallocBuffer(listOffsetBuf_, MetalAllocType::IVFLists);
            }
            listOffsetBuf_ = resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);
        }
        if (listLengthBuf_ == nil || [listLengthBuf_ length] < metaBytes) {
            if (listLengthBuf_ != nil) {
                resources_->deallocBuffer(listLengthBuf_, MetalAllocType::IVFLists);
            }
            listLengthBuf_ = resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);
        }
        FAISS_THROW_IF_NOT_MSG(
                listOffsetBuf_ && listLengthBuf_,
                "MetalIVFFlatImpl: failed to allocate metadata GPU buffers");

        auto* offPtr = reinterpret_cast<uint32_t*>([listOffsetBuf_ contents]);
        auto* lenPtr = reinterpret_cast<uint32_t*>([listLengthBuf_ contents]);
        for (size_t i = 0; i < (size_t)nlist_; ++i) {
            offPtr[i] = checkedToU32(
                    listOffset_[i],
                    "MetalIVFFlatImpl: list offset exceeds uint32 range");
            lenPtr[i] = checkedToU32(
                    listLength_[i],
                    "MetalIVFFlatImpl: list length exceeds uint32 range");
        }
    }

    if (codesBytes == 0 || idsBytes == 0) {
        return;
    }

    if (idsBuffer_ == nil || [idsBuffer_ length] < idsBytes) {
        if (idsBuffer_ != nil) {
            resources_->deallocBuffer(idsBuffer_, MetalAllocType::IVFLists);
        }
        idsBuffer_ = resources_->allocBuffer(idsBytes, MetalAllocType::IVFLists);
        forceFullUpload = true;
    }

    FAISS_THROW_IF_NOT_MSG(
            idsBuffer_,
            "MetalIVFFlatImpl: failed to allocate IVF ids buffer");

    if (codesBuffer_ == nil || [codesBuffer_ length] < codesBytes) {
        if (codesBuffer_ != nil) {
            resources_->deallocBuffer(codesBuffer_, MetalAllocType::IVFLists);
        }
        codesBuffer_ = resources_->allocBuffer(codesBytes, MetalAllocType::IVFLists);
        forceFullUpload = true;
    }
    FAISS_THROW_IF_NOT_MSG(
            codesBuffer_,
            "MetalIVFFlatImpl: failed to allocate IVF codes buffer");

    if (forceFullUpload) {
        std::memcpy([idsBuffer_ contents], hostIds_.data(), idsBytes);
        std::memcpy([codesBuffer_ contents], hostCodes_.data(), codesBytes);
    } else {
        for (size_t l = 0; l < (size_t)nlist_; ++l) {
            size_t add = addPerList[l];
            bool moved = (l < movedLists.size() && movedLists[l] != 0);
            if (add == 0 && !moved) {
                continue;
            }
            size_t start = moved ? listOffset_[l] : (listOffset_[l] + oldLength[l]);
            size_t count = moved ? listLength_[l] : add;
            if (count == 0) {
                continue;
            }

            std::memcpy(
                    reinterpret_cast<idx_t*>([idsBuffer_ contents]) + start,
                    hostIds_.data() + start,
                    count * sizeof(idx_t));
            std::memcpy(
                    reinterpret_cast<float*>([codesBuffer_ contents]) +
                            start * (size_t)dim_,
                    hostCodes_.data() + start * (size_t)dim_,
                    count * (size_t)dim_ * sizeof(float));
        }
    }

    if (interleavedLayout_) {
        interleavedDirty_ = true;
    } else {
        if (interleavedCodesBuf_ != nil) {
            resources_->deallocBuffer(interleavedCodesBuf_, MetalAllocType::IVFLists);
            interleavedCodesBuf_ = nil;
        }
        if (interleavedCodesOffsetBuf_ != nil) {
            resources_->deallocBuffer(interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
            interleavedCodesOffsetBuf_ = nil;
        }
        interleavedDirty_ = false;
    }
}

void MetalIVFFlatImpl::rebuildInterleavedBuffers_() {
    if (!interleavedLayout_ || !resources_ || !resources_->isAvailable()) {
        return;
    }

    const size_t metaBytes = (size_t)nlist_ * sizeof(uint32_t);
    constexpr int G = kInterleavedGroupSize; // 32

    std::vector<uint32_t> ilOffsets((size_t)nlist_);
    size_t totalIlFloats = 0;
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        ilOffsets[l] = checkedToU32(
                totalIlFloats,
                "MetalIVFFlatImpl: interleaved offset exceeds uint32 range");
        size_t len = listLength_[l];
        size_t numBlocks = (len + G - 1) / G;
        totalIlFloats += numBlocks * G * (size_t)dim_;
    }

    if (totalIlFloats == 0) {
        if (interleavedCodesBuf_ != nil) {
            resources_->deallocBuffer(interleavedCodesBuf_, MetalAllocType::IVFLists);
            interleavedCodesBuf_ = nil;
        }
        if (interleavedCodesOffsetBuf_ != nil) {
            resources_->deallocBuffer(
                    interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
            interleavedCodesOffsetBuf_ = nil;
        }
        interleavedDirty_ = false;
        return;
    }

    const size_t ilBytes = totalIlFloats * sizeof(float);
    if (interleavedCodesBuf_ != nil) {
        resources_->deallocBuffer(interleavedCodesBuf_, MetalAllocType::IVFLists);
        interleavedCodesBuf_ = nil;
    }
    interleavedCodesBuf_ =
            resources_->allocBuffer(ilBytes, MetalAllocType::IVFLists);

    if (interleavedCodesOffsetBuf_ != nil) {
        resources_->deallocBuffer(interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
        interleavedCodesOffsetBuf_ = nil;
    }
    interleavedCodesOffsetBuf_ =
            resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);

    if (!interleavedCodesBuf_ || !interleavedCodesOffsetBuf_) {
        if (interleavedCodesBuf_ != nil) {
            resources_->deallocBuffer(interleavedCodesBuf_, MetalAllocType::IVFLists);
            interleavedCodesBuf_ = nil;
        }
        if (interleavedCodesOffsetBuf_ != nil) {
            resources_->deallocBuffer(
                    interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
            interleavedCodesOffsetBuf_ = nil;
        }
        interleavedDirty_ = true;
        return;
    }

    auto* dst = reinterpret_cast<float*>([interleavedCodesBuf_ contents]);
    std::memset(dst, 0, ilBytes);

    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        size_t len = listLength_[l];
        if (len == 0) continue;
        size_t srcOff = listOffset_[l];
        size_t dstOff = ilOffsets[l];
        size_t numBlocks = (len + G - 1) / G;

        for (size_t b = 0; b < numBlocks; ++b) {
            for (int dd = 0; dd < dim_; ++dd) {
                for (int g = 0; g < G; ++g) {
                    size_t vi = b * G + g;
                    float val = (vi < len)
                            ? hostCodes_[(srcOff + vi) * (size_t)dim_ + dd]
                            : 0.0f;
                    dst[dstOff + b * G * (size_t)dim_ + dd * G + g] = val;
                }
            }
        }
    }

    auto* ptr = reinterpret_cast<uint32_t*>([interleavedCodesOffsetBuf_ contents]);
    for (size_t i = 0; i < (size_t)nlist_; ++i) {
        ptr[i] = ilOffsets[i];
    }
    interleavedDirty_ = false;
}

void MetalIVFFlatImpl::ensureInterleavedLayoutUpToDate() {
    if (!interleavedLayout_) {
        return;
    }
    if (!interleavedDirty_ && interleavedCodesBuf_ && interleavedCodesOffsetBuf_) {
        return;
    }
    rebuildInterleavedBuffers_();
}

} // namespace gpu_metal
} // namespace faiss
