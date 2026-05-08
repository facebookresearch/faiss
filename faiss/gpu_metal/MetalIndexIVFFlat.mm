// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexIVFFlat.h"

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu_metal/MetalDistance.h>
#include <faiss/gpu_metal/impl/MetalIVFFlat.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/InvertedLists.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr size_t kDefaultIvfQueryTileBudgetBytes = 256ULL * 1024 * 1024;
constexpr size_t kMinIvfQueryTileBudgetBytes = 16ULL * 1024 * 1024;
constexpr size_t kMaxIvfQueryTileBudgetBytes = 4ULL * 1024 * 1024 * 1024;
constexpr size_t kDefaultIvfFullCoarseMaxBytes = 16ULL * 1024 * 1024;
constexpr faiss::idx_t kIVFFlatSupportedMaxK = 1024;
constexpr faiss::idx_t kAutoReserveMinBatch = 0;
constexpr size_t kIvfExactCandidates = 1024;
constexpr size_t kMinIvfExactCandidates = 64;

size_t getIvfQueryTileBudgetBytes() {
    const char* envBytes = std::getenv("FAISS_METAL_IVF_QUERY_TILE_BYTES");
    if (envBytes && envBytes[0] != '\0') {
        char* end = nullptr;
        unsigned long long val = std::strtoull(envBytes, &end, 10);
        if (end != envBytes && val != 0) {
            size_t out = static_cast<size_t>(val);
            out = std::max(out, kMinIvfQueryTileBudgetBytes);
            out = std::min(out, kMaxIvfQueryTileBudgetBytes);
            return out;
        }
    }
    return kDefaultIvfQueryTileBudgetBytes;
}

faiss::idx_t getIvfAutoReserveMinBatch() {
    const char* env = std::getenv("FAISS_METAL_IVF_AUTO_RESERVE_MIN_BATCH");
    if (!env || env[0] == '\0') {
        return kAutoReserveMinBatch;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(env, &end, 10);
    if (end == env) {
        return kAutoReserveMinBatch;
    }
    if (v > (unsigned long long)std::numeric_limits<faiss::idx_t>::max()) {
        return std::numeric_limits<faiss::idx_t>::max();
    }
    return (faiss::idx_t)v;
}

size_t chooseIvfSearchTileRows(
        size_t nq,
        int d,
        faiss::idx_t k,
        size_t nprobe,
        int nlist) {
    size_t perQuery = 0;
    perQuery += (size_t)d * sizeof(float);
    perQuery += (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * (sizeof(float) + sizeof(int32_t));
    perQuery += (size_t)nlist * sizeof(float);
    if (perQuery == 0) {
        return nq;
    }

    size_t tile = getIvfQueryTileBudgetBytes() / perQuery;
    tile = std::max<size_t>(tile, 1);
    tile = std::min(tile, nq);
    return tile;
}

size_t chooseIvfPreassignedTileRows(
        size_t nq,
        int d,
        faiss::idx_t k,
        size_t nprobe) {
    size_t perQuery = 0;
    perQuery += (size_t)d * sizeof(float);
    perQuery += (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * sizeof(int32_t);
    if (perQuery == 0) {
        return nq;
    }

    size_t tile = getIvfQueryTileBudgetBytes() / perQuery;
    tile = std::max<size_t>(tile, 1);
    tile = std::min(tile, nq);
    return tile;
}

bool allowCpuFallbackForIvf() {
    const char* env = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    if (!env || env[0] == '\0') {
        return true;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' ||
        env[0] == 'F') {
        return false;
    }
    return true;
}

bool logCpuFallbackForIvf() {
    const char* env = std::getenv("FAISS_METAL_IVF_LOG_CPU_FALLBACK");
    if (!env || env[0] == '\0') {
        return true;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' ||
        env[0] == 'F') {
        return false;
    }
    return true;
}

bool logSyncProfileForIvf() {
    const char* env = std::getenv("FAISS_METAL_IVF_LOG_SYNC_PROFILE");
    if (!env || env[0] == '\0') {
        return false;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' ||
        env[0] == 'F') {
        return false;
    }
    return true;
}

size_t getIvfFullCoarseMaxBytes() {
    const char* env = std::getenv("FAISS_METAL_IVF_FULL_COARSE_MAX_BYTES");
    if (!env || env[0] == '\0') {
        return kDefaultIvfFullCoarseMaxBytes;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(env, &end, 10);
    if (end == env) {
        return kDefaultIvfFullCoarseMaxBytes;
    }
    return static_cast<size_t>(v);
}

bool useFullCoarseGpuForIvf() {
    const char* env = std::getenv("FAISS_METAL_IVF_USE_FULL_COARSE");
    if (!env || env[0] == '\0') {
        return true;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' ||
        env[0] == 'F') {
        return false;
    }
    return true;
}

bool forceChunkedIvfSelectionPath() {
    const char* env = std::getenv("FAISS_METAL_IVF_FORCE_CHUNKED_SELECTION");
    if (!env || env[0] == '\0') {
        return false;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' ||
        env[0] == 'F') {
        return false;
    }
    return true;
}

size_t getIvfExactCandidateBudget() {
    const char* env = std::getenv("FAISS_METAL_IVF_EXACT_CANDIDATES");
    if (!env || env[0] == '\0') {
        return kIvfExactCandidates;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(env, &end, 10);
    if (end == env || v == 0) {
        return kIvfExactCandidates;
    }
    size_t out = static_cast<size_t>(v);
    out = std::max(out, kMinIvfExactCandidates);
    out = std::min(out, kIvfExactCandidates);
    return out;
}

struct ScopedIds {
    const faiss::InvertedLists* inv = nullptr;
    size_t listNo = 0;
    const faiss::idx_t* ptr = nullptr;

    ScopedIds() = default;
    ScopedIds(const faiss::InvertedLists* inv_, size_t listNo_)
            : inv(inv_), listNo(listNo_) {
        if (inv) {
            ptr = inv->get_ids(listNo);
        }
    }
    ~ScopedIds() {
        if (inv && ptr) {
            inv->release_ids(listNo, ptr);
        }
    }
};

struct ScopedCodes {
    const faiss::InvertedLists* inv = nullptr;
    size_t listNo = 0;
    const uint8_t* ptr = nullptr;

    ScopedCodes() = default;
    ScopedCodes(const faiss::InvertedLists* inv_, size_t listNo_)
            : inv(inv_), listNo(listNo_) {
        if (inv) {
            ptr = inv->get_codes(listNo);
        }
    }
    ~ScopedCodes() {
        if (inv && ptr) {
            inv->release_codes(listNo, ptr);
        }
    }
};

void floatToHalf(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        __fp16 h = (__fp16)src[i];
        std::memcpy(&dst[i], &h, sizeof(uint16_t));
    }
}

inline faiss::idx_t decodeCpuLabelFromPair(
        const faiss::IndexIVFFlat* cpuIndex,
        int64_t pairLabel) {
    const uint64_t pair = static_cast<uint64_t>(pairLabel);
    const uint64_t listNo = faiss::lo_listno(pair);
    const uint64_t offset = faiss::lo_offset(pair);
    if (!cpuIndex || listNo >= (uint64_t)cpuIndex->nlist) {
        return -1;
    }
    const size_t sz = cpuIndex->invlists->list_size((size_t)listNo);
    if (offset >= sz) {
        return -1;
    }
    ScopedIds ids(cpuIndex->invlists, (size_t)listNo);
    return ids.ptr ? ids.ptr[offset] : -1;
}

inline int checkedIdxToInt(faiss::idx_t v, const char* what) {
    if (!(v >= 0 && v <= (faiss::idx_t)std::numeric_limits<int>::max())) {
        FAISS_THROW_FMT("%s", what);
    }
    return (int)v;
}

inline int checkedSizeToInt(size_t v, const char* what) {
    if (!(v <= (size_t)std::numeric_limits<int>::max())) {
        FAISS_THROW_FMT("%s", what);
    }
    return (int)v;
}

bool assignedListsWithinLimit(
        const faiss::IndexIVFFlat* cpuIndex,
        const faiss::idx_t* assign,
        size_t count,
        size_t maxListSize,
        faiss::idx_t& badListId,
        size_t& badListSize) {
    if (!cpuIndex || !cpuIndex->invlists || !assign) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const faiss::idx_t listId = assign[i];
        if (listId < 0 || listId >= cpuIndex->nlist) {
            badListId = listId;
            badListSize = 0;
            return false;
        }
        const size_t sz = cpuIndex->invlists->list_size((size_t)listId);
        if (sz > maxListSize) {
            badListId = listId;
            badListSize = sz;
            return false;
        }
    }
    return true;
}

std::string explainIvfScanFailureEnvelope(
        int d,
        faiss::idx_t k,
        size_t exactCandidateBudget,
        size_t nprobe,
        bool useInterleaved,
        const faiss::IndexIVFFlat* cpuIndex,
        const faiss::idx_t* assign,
        size_t assignCount) {
    if (!(d > 0 && d <= 512)) {
        return "scan envelope: d must be in [1,512]";
    }
    if (useInterleaved && (d % 4 != 0)) {
        return "scan envelope: interleaved layout requires d % 4 == 0";
    }
    if (nprobe > 0 && (size_t)k > (exactCandidateBudget / nprobe)) {
        return std::string("scan envelope: nprobe*k exceeds ") +
                std::to_string(exactCandidateBudget) +
                " exact-candidate bound";
    }
    faiss::idx_t badListId = -1;
    size_t badListSize = 0;
    if (!assignedListsWithinLimit(
                cpuIndex,
                assign,
                assignCount,
                exactCandidateBudget,
                badListId,
                badListSize)) {
        if (badListSize > 0) {
            return std::string("scan envelope: selected list size exceeds ") +
                    std::to_string(exactCandidateBudget) + " (list=" +
                    std::to_string((long long)badListId) +
                    ", size=" + std::to_string(badListSize) + ")";
        }
        return "scan envelope: invalid coarse assignment list id";
    }
    return {};
}

bool selectedListStats(
        const faiss::IndexIVFFlat* cpuIndex,
        const faiss::idx_t* assign,
        size_t count,
        std::vector<uint8_t>& usedLists,
        size_t& maxSelectedListSize,
        faiss::idx_t& badListId) {
    if (!cpuIndex || !cpuIndex->invlists || !assign || cpuIndex->nlist < 0) {
        return false;
    }
    usedLists.assign((size_t)cpuIndex->nlist, 0);
    maxSelectedListSize = 0;
    badListId = -1;
    for (size_t i = 0; i < count; ++i) {
        faiss::idx_t li = assign[i];
        if (li < 0 || li >= cpuIndex->nlist) {
            badListId = li;
            return false;
        }
        if (!usedLists[(size_t)li]) {
            usedLists[(size_t)li] = 1;
            const size_t sz = cpuIndex->invlists->list_size((size_t)li);
            maxSelectedListSize = std::max(maxSelectedListSize, sz);
        }
    }
    return true;
}

inline void insertTopKCandidate(
        float candDist,
        int64_t candIdx,
        bool isL2,
        float* bestDist,
        int64_t* bestIdx,
        int k) {
    if (candIdx < 0 || k <= 0) {
        return;
    }
    int pos = -1;
    if (isL2) {
        for (int j = 0; j < k; ++j) {
            if (candDist < bestDist[j]) {
                pos = j;
                break;
            }
        }
    } else {
        for (int j = 0; j < k; ++j) {
            if (candDist > bestDist[j]) {
                pos = j;
                break;
            }
        }
    }
    if (pos < 0) {
        return;
    }
    for (int j = k - 1; j > pos; --j) {
        bestDist[j] = bestDist[j - 1];
        bestIdx[j] = bestIdx[j - 1];
    }
    bestDist[pos] = candDist;
    bestIdx[pos] = candIdx;
}
} // namespace

namespace faiss {
namespace gpu_metal {

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, metricArg, config),
          indicesOptions_(config.indicesOptions),
          interleavedLayout_(config.interleavedLayout) {
    // Simple CPU quantizer: IndexFlatL2 or IndexFlatIP
    faiss::IndexFlat* quantizer = (metric == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dims)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dims);
    cpuIndex_ = std::make_unique<faiss::IndexIVFFlat>(
            quantizer, (size_t)d, (size_t)nlist, metric);
    cpuIndex_->own_fields = true;
    gpuIvf_ = std::make_unique<MetalIVFFlatImpl>(
            resources,
            (int)d,
            nlist,
            metric,
            metricArg,
            indicesOptions_,
            interleavedLayout_);
}

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        faiss::Index* coarseQuantizer,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config,
        bool ownFields)
        : MetalIndex(resources, dims, metric, metricArg, config),
          indicesOptions_(config.indicesOptions),
          interleavedLayout_(config.interleavedLayout) {
    FAISS_THROW_IF_NOT_MSG(
            coarseQuantizer != nullptr,
            "MetalIndexIVFFlat: coarseQuantizer must be non-null");
    cpuIndex_ = std::make_unique<faiss::IndexIVFFlat>(
            coarseQuantizer, (size_t)d, (size_t)nlist, metric);
    cpuIndex_->own_fields = ownFields;
    gpuIvf_ = std::make_unique<MetalIVFFlatImpl>(
            resources,
            (int)d,
            nlist,
            metric,
            metricArg,
            indicesOptions_,
            interleavedLayout_);
}

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexIVFFlat* cpuIndex,
        MetalIndexConfig config)
        : MetalIndex(
                  resources,
                  (int)cpuIndex->d,
                  cpuIndex->metric_type,
                  cpuIndex->metric_arg,
                  config),
          indicesOptions_(config.indicesOptions),
          interleavedLayout_(config.interleavedLayout) {
    faiss::IndexFlat* quantizer = (cpuIndex->metric_type == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP((int)cpuIndex->d)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2((int)cpuIndex->d);
    cpuIndex_ = std::make_unique<faiss::IndexIVFFlat>(
            quantizer, cpuIndex->d, cpuIndex->nlist, cpuIndex->metric_type);
    cpuIndex_->own_fields = true;
    gpuIvf_ = std::make_unique<MetalIVFFlatImpl>(
            resources,
            (int)cpuIndex->d,
            cpuIndex->nlist,
            cpuIndex->metric_type,
            cpuIndex->metric_arg,
            indicesOptions_,
            interleavedLayout_);
    copyFrom(cpuIndex);
}

MetalIndexIVFFlat::~MetalIndexIVFFlat() = default;

void MetalIndexIVFFlat::uploadCentroids_() const {
    if (!cpuIndex_ || !cpuIndex_->quantizer || !resources_) {
        return;
    }
    auto* q = cpuIndex_->quantizer;
    if (!q || q->ntotal == 0) {
        centroidBuf_ = nil;
        centroidNormsBuf_ = nil;
        return;
    }
    size_t nCentroids = (size_t)q->ntotal;
    std::vector<float> centroids(nCentroids * (size_t)d);
    q->reconstruct_n(0, (idx_t)nCentroids, centroids.data());

    const bool fp16 = config_.useFloat16CoarseQuantizer;
    size_t elemSize = fp16 ? sizeof(uint16_t) : sizeof(float);
    size_t bytes = nCentroids * (size_t)d * elemSize;
    id<MTLDevice> device = resources_->getDevice();
    if (!device) {
        return;
    }
    centroidBuf_ = [device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
    if (centroidBuf_) {
        const float* src = centroids.data();
        if (fp16) {
            floatToHalf(src,
                        reinterpret_cast<uint16_t*>([centroidBuf_ contents]),
                        nCentroids * (size_t)d);
        } else {
            std::memcpy([centroidBuf_ contents], src, bytes);
        }
    }

    // Pre-compute centroid L2 norms on GPU (float32 centroids only).
    if (centroidBuf_ && metric_type == METRIC_L2 && !fp16) {
        size_t normBytes = nCentroids * sizeof(float);
        centroidNormsBuf_ = [device newBufferWithLength:normBytes
                                               options:MTLResourceStorageModeShared];
        if (centroidNormsBuf_) {
            id<MTLCommandQueue> queue = resources_->getCommandQueue();
            const int nCentroidsI = checkedSizeToInt(
                    nCentroids,
                    "MetalIndexIVFFlat: centroid count exceeds int range");
            if (!runMetalComputeNorms(device, queue, centroidBuf_,
                                      nCentroidsI, d, centroidNormsBuf_, false)) {
                centroidNormsBuf_ = nil;
            }
        }
    } else {
        centroidNormsBuf_ = nil;
    }
}

void MetalIndexIVFFlat::ensureSearchBuf_(
        id<MTLBuffer>& buf,
        size_t& cap,
        size_t needed) const {
    if (buf != nil && cap >= needed) {
        return; // already large enough
    }
    // Grow by 2× to amortise future reallocations.
    size_t newCap = std::max(needed, cap * 2);
    id<MTLDevice> device = resources_->getDevice();
    buf = [device newBufferWithLength:newCap
                              options:MTLResourceStorageModeShared];
    cap = buf ? newCap : 0;
}

void MetalIndexIVFFlat::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->train(n, x);
    is_trained = cpuIndex_->is_trained;
    if (is_trained) {
        uploadCentroids_();
    }
}

void MetalIndexIVFFlat::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    if (n == 0) {
        return;
    }
    // Compute list assignments for this batch (mirrors IndexIVF::add path).
    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    idx_t oldNt = cpuIndex_->ntotal;
    const idx_t autoReserveMinBatch = getIvfAutoReserveMinBatch();
    if (gpuIvf_ && autoReserveMinBatch > 0 && n >= autoReserveMinBatch) {
        // Pre-reserve before append to reduce relayout spikes on large batches.
        gpuIvf_->reserveMemory(oldNt + n);
    }
    cpuIndex_->add(n, x);
    ntotal = cpuIndex_->ntotal;

    // Mirror IVF data into Metal IVF storage.
    if (gpuIvf_) {
        std::vector<idx_t> ids(n);
        for (idx_t i = 0; i < n; ++i) {
            ids[i] = oldNt + i;
        }
        gpuIvf_->appendVectors(n, x, list_nos.data(), ids.data());
    }
}

void MetalIndexIVFFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(xids != nullptr);

    idx_t oldNt = cpuIndex_->ntotal;
    const idx_t autoReserveMinBatch = getIvfAutoReserveMinBatch();
    if (gpuIvf_ && autoReserveMinBatch > 0 && n >= autoReserveMinBatch) {
        // Pre-reserve before append to reduce relayout spikes on large batches.
        gpuIvf_->reserveMemory(oldNt + n);
    }

    // Compute list assignments for this batch.
    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    cpuIndex_->add_with_ids(n, x, xids);
    ntotal = cpuIndex_->ntotal;

    if (gpuIvf_) {
        gpuIvf_->appendVectors(n, x, list_nos.data(), xids);
    }
}

void MetalIndexIVFFlat::reset() {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->reset();
    ntotal = 0;
    if (gpuIvf_) {
        gpuIvf_->reset();
    }
}

void MetalIndexIVFFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            k <= kIVFFlatSupportedMaxK,
            "MetalIndexIVFFlat supports k <= 1024; larger k is not yet supported");

    const float inf    = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();

    // Empty index: mirror CPU IndexIVF behavior (labels = -1).
    if (cpuIndex_->ntotal == 0 || n == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i]    = -1;
            distances[i] = (metric_type == METRIC_L2) ? inf : negInf;
        }
        return;
    }

    // Only L2 and IP are supported.
    FAISS_THROW_IF_NOT(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);
    const bool isL2 = (metric_type == METRIC_L2);

    // Determine nprobe from params or index.
    size_t nprobe = cpuIndex_->nprobe;
    if (auto* ivfParams = dynamic_cast<const IVFSearchParameters*>(params)) {
        if (ivfParams->nprobe > 0) {
            nprobe = ivfParams->nprobe;
        }
    }
    nprobe = std::min(nprobe, cpuIndex_->nlist);
    if (nprobe == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i]    = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    auto cpuFallbackSearch = [&](idx_t qBase, idx_t qCount) {
        const float* xTile = x + (size_t)qBase * (size_t)d;
        float* distancesTile = distances + (size_t)qBase * (size_t)k;
        idx_t* labelsTile = labels + (size_t)qBase * (size_t)k;
        if (indicesOptions_ == faiss::gpu::INDICES_IVF) {
            std::vector<float> coarseDist((size_t)qCount * nprobe);
            std::vector<idx_t> coarseAssign((size_t)qCount * nprobe);
            cpuIndex_->quantizer->search(
                    qCount,
                    xTile,
                    (idx_t)nprobe,
                    coarseDist.data(),
                    coarseAssign.data());
            cpuIndex_->search_preassigned(
                    qCount,
                    xTile,
                    k,
                    coarseAssign.data(),
                    coarseDist.data(),
                    distancesTile,
                    labelsTile,
                    true,
                    dynamic_cast<const IVFSearchParameters*>(params),
                    nullptr);
        } else {
            cpuIndex_->search(qCount, xTile, k, distancesTile, labelsTile);
        }
    };
    const bool allowCpuFallback = allowCpuFallbackForIvf();
    const bool logCpuFallback = logCpuFallbackForIvf();
    const bool logSyncProfile = logSyncProfileForIvf();
    const bool forceChunkedSelection = forceChunkedIvfSelectionPath();
    const size_t exactCandidateBudget = getIvfExactCandidateBudget();
    idx_t fallbackCount = 0;
    std::string firstFallbackReason;
    idx_t fullSearchCalls = 0;
    idx_t scanCalls = 0;
    idx_t syncScanCalls = 0;
    idx_t asyncScanCalls = 0;
    idx_t asyncBatchSyncs = 0;
    auto fallbackOrThrow = [&](idx_t qBase, idx_t qCount, const char* reason) {
        if (!allowCpuFallback) {
            FAISS_THROW_FMT(
                    "MetalIndexIVFFlat::search requires GPU execution but hit fallback (%s) "
                    "at qBase=%lld qCount=%lld",
                    reason,
                    (long long)qBase,
                    (long long)qCount);
        }
        ++fallbackCount;
        if (firstFallbackReason.empty()) {
            firstFallbackReason = reason ? reason : "unknown";
        }
        cpuFallbackSearch(qBase, qCount);
    };

    id<MTLDevice>      device = resources_->getDevice();
    id<MTLCommandQueue> queue = resources_->getCommandQueue();
    if (gpuIvf_) {
        gpuIvf_->ensureInterleavedLayoutUpToDate();
    }

    const bool hasFlatCodes = gpuIvf_ && gpuIvf_->codesBuffer();
    const bool hasInterleavedCodes =
            gpuIvf_ && gpuIvf_->interleavedCodesBuffer() &&
            gpuIvf_->interleavedCodesOffsetBuffer();
    const bool hasScanCodes = hasFlatCodes || hasInterleavedCodes;

    // Fall back to CPU if Metal is not available or GPU IVF storage not ready.
    if (!device || !queue || !gpuIvf_ || !hasScanCodes || !gpuIvf_->idsBuffer() ||
        !gpuIvf_->listOffsetGpuBuffer() || !gpuIvf_->listLengthGpuBuffer()) {
        fallbackOrThrow(0, n, "missing Metal device/queue or IVF buffers");
        return;
    }

    // ---- Upload centroid buffer if needed -----------------------------------
    if (!centroidBuf_) {
        uploadCentroids_();
    }

    int nlist = (int)cpuIndex_->nlist;

    const size_t tileRows = chooseIvfSearchTileRows((size_t)n, d, k, nprobe, nlist);
    const int avgListLen = (ntotal > 0 && nlist > 0) ? (int)(ntotal / nlist) : 256;

    for (idx_t qBase = 0; qBase < n; qBase += (idx_t)tileRows) {
        idx_t qCount = std::min<idx_t>((idx_t)tileRows, n - qBase);
        const float* xTile = x + (size_t)qBase * (size_t)d;

        size_t queriesBytes = (size_t)qCount * (size_t)d * sizeof(float);
        size_t outDistBytes = (size_t)qCount * (size_t)k * sizeof(float);
        size_t outIdxBytes = (size_t)qCount * (size_t)k * sizeof(int64_t);
        size_t perListBytes = (size_t)qCount * nprobe * (size_t)k * sizeof(float);
        size_t perListIdxB = (size_t)qCount * nprobe * (size_t)k * sizeof(int64_t);
        size_t coarseDistBytes = (size_t)qCount * nprobe * sizeof(float);
        size_t coarseIdxBytes = (size_t)qCount * nprobe * sizeof(int32_t);
        size_t distMatBytes = (size_t)qCount * (size_t)nlist * sizeof(float);

        ensureSearchBuf_(searchQueriesBuf_, searchQueriesCap_, queriesBytes);
        ensureSearchBuf_(searchOutDistBuf_, searchOutDistCap_, outDistBytes);
        ensureSearchBuf_(searchOutIdxBuf_, searchOutIdxCap_, outIdxBytes);
        const bool wantsFullCoarse = useFullCoarseGpuForIvf();
        const size_t fullCoarseMaxBytes = getIvfFullCoarseMaxBytes();
        bool tryFullCoarse = wantsFullCoarse && centroidBuf_ &&
                nprobe <= (size_t)getMetalDistanceMaxK() &&
                distMatBytes <= fullCoarseMaxBytes;
        if (forceChunkedSelection) {
            // Experimental path for selection-scalability work: skip monolithic
            // full-search path and route through coarse+chunked scan/merge.
            tryFullCoarse = false;
        }

        ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytes);
        ensureSearchBuf_(searchPerListIdxBuf_, searchPerListIdxCap_, perListIdxB);
        if (tryFullCoarse) {
            ensureSearchBuf_(coarseOutDistBuf_, coarseOutDistCap_, coarseDistBytes);
            ensureSearchBuf_(coarseOutIdxBuf_, coarseOutIdxCap_, coarseIdxBytes);
            ensureSearchBuf_(distMatrixBuf_, distMatrixCap_, distMatBytes);
            if (!coarseOutDistBuf_ || !coarseOutIdxBuf_ || !distMatrixBuf_) {
                // Fall back to CPU coarse assignment path if matrix scratch
                // cannot be allocated for this tile.
                tryFullCoarse = false;
            }
        }

        if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
            !searchPerListDistBuf_ || !searchPerListIdxBuf_) {
            fallbackOrThrow(qBase, qCount, "failed to allocate tiled search buffers");
            continue;
        }

        std::memcpy([searchQueriesBuf_ contents], xTile, queriesBytes);
        const int qCountI = checkedIdxToInt(
                qCount, "MetalIndexIVFFlat: qCount exceeds int range");
        const int kI = checkedIdxToInt(k, "MetalIndexIVFFlat: k exceeds int range");
        const int nprobeI = checkedSizeToInt(
                nprobe, "MetalIndexIVFFlat: nprobe exceeds int range");

        bool ok = false;
        if (tryFullCoarse) {
            ++fullSearchCalls;
            ok = runMetalIVFFlatFullSearch(
                    device, queue,
                    searchQueriesBuf_,
                    qCountI, d, kI, nprobeI, isL2,
                    centroidBuf_, nlist,
                    gpuIvf_->codesBuffer(),
                    gpuIvf_->idsBuffer(),
                    gpuIvf_->listOffsetGpuBuffer(),
                    gpuIvf_->listLengthGpuBuffer(),
                    searchOutDistBuf_,
                    searchOutIdxBuf_,
                    searchPerListDistBuf_,
                    searchPerListIdxBuf_,
                    coarseOutDistBuf_,
                    coarseOutIdxBuf_,
                    distMatrixBuf_,
                    centroidNormsBuf_,
                    avgListLen,
                    gpuIvf_->interleavedCodesBuffer(),
                    gpuIvf_->interleavedCodesOffsetBuffer(),
                    config_.useFloat16CoarseQuantizer);
        }

        if (!ok) {
            std::vector<float> coarseDistVec((size_t)qCount * nprobe);
            std::vector<idx_t> coarseAssignVec((size_t)qCount * nprobe);
            cpuIndex_->quantizer->search(
                    qCount,
                    xTile,
                    (idx_t)nprobe,
                    coarseDistVec.data(),
                    coarseAssignVec.data());
            size_t coarseBytes = (size_t)qCount * nprobe * sizeof(int32_t);
            ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseBytes);
            if (!searchCoarseBuf_) {
                fallbackOrThrow(qBase, qCount, "failed to allocate coarse assign buffer");
                continue;
            }
            auto* dst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
            for (size_t i = 0; i < (size_t)qCount * nprobe; ++i) {
                FAISS_THROW_IF_NOT_MSG(
                        coarseAssignVec[i] >=
                                        (idx_t)std::numeric_limits<int32_t>::min() &&
                                coarseAssignVec[i] <=
                                        (idx_t)std::numeric_limits<int32_t>::max(),
                        "MetalIndexIVFFlat: coarse assignment exceeds int32 range");
                dst[i] = (int32_t)coarseAssignVec[i];
            }
            const bool useInterleaved =
                    gpuIvf_->interleavedCodesBuffer() != nil &&
                    gpuIvf_->interleavedCodesOffsetBuffer() != nil;
            const std::string envelopeReason = explainIvfScanFailureEnvelope(
                    d,
                    k,
                    exactCandidateBudget,
                    nprobe,
                    useInterleaved,
                    cpuIndex_.get(),
                    coarseAssignVec.data(),
                    (size_t)qCount * nprobe);
            bool usedChunkedScan = false;
            std::vector<float> chunkMergedDist;
            std::vector<int64_t> chunkMergedIdx;

            std::vector<uint8_t> usedLists;
            size_t maxSelectedListSize = 0;
            idx_t badListId = -1;
            bool haveListStats = selectedListStats(
                    cpuIndex_.get(),
                    coarseAssignVec.data(),
                    (size_t)qCount * nprobe,
                    usedLists,
                    maxSelectedListSize,
                    badListId);
                const bool needProbeChunk =
                    nprobe > 0 && (size_t)k > (exactCandidateBudget / nprobe);
                const bool needListChunk =
                    haveListStats && maxSelectedListSize > exactCandidateBudget;

            if (needProbeChunk || needListChunk) {
                usedChunkedScan = true;
                chunkMergedDist.resize((size_t)qCount * (size_t)k, isL2 ? inf : negInf);
                chunkMergedIdx.resize((size_t)qCount * (size_t)k, -1);
                ok = true;

                const size_t maxProbePerChunk = needProbeChunk
                        ? std::max<size_t>(1, exactCandidateBudget / (size_t)k)
                        : nprobe;
                const size_t listChunk = exactCandidateBudget;
                const size_t numListPass = needListChunk
                        ? ((maxSelectedListSize + listChunk - 1) / listChunk)
                        : 1;
                // List-chunk remapping only updates list offset/length metadata.
                // Interleaved offset metadata is list-base-relative, so force flat
                // code path for list-chunked passes to keep ids/codes aligned.
                id<MTLBuffer> chunkInterleavedCodes = needListChunk
                        ? nil
                        : gpuIvf_->interleavedCodesBuffer();
                id<MTLBuffer> chunkInterleavedOffsets = needListChunk
                        ? nil
                        : gpuIvf_->interleavedCodesOffsetBuffer();

                const uint32_t* baseListOffset = reinterpret_cast<const uint32_t*>(
                        [gpuIvf_->listOffsetGpuBuffer() contents]);
                const uint32_t* baseListLength = reinterpret_cast<const uint32_t*>(
                        [gpuIvf_->listLengthGpuBuffer() contents]);
                id<MTLBuffer> listOffsetPassBuf = nil;
                id<MTLBuffer> listLengthPassBuf = nil;
                std::vector<uint32_t> passOffset;
                std::vector<uint32_t> passLength;
                if (needListChunk) {
                    size_t metaBytes = (size_t)nlist * sizeof(uint32_t);
                    listOffsetPassBuf = [device newBufferWithLength:metaBytes
                                                             options:MTLResourceStorageModeShared];
                    listLengthPassBuf = [device newBufferWithLength:metaBytes
                                                             options:MTLResourceStorageModeShared];
                    if (!listOffsetPassBuf || !listLengthPassBuf) {
                        ok = false;
                    } else {
                        passOffset.resize((size_t)nlist);
                        passLength.resize((size_t)nlist);
                    }
                }

                const bool canAsyncProbeBatch =
                        needProbeChunk && !needListChunk && numListPass == 1;
                struct ProbePass {
                    size_t p0 = 0;
                    size_t chunkProbe = 0;
                    id<MTLBuffer> coarseBuf = nil;
                    id<MTLBuffer> outDistBuf = nil;
                    id<MTLBuffer> outIdxBuf = nil;
                };
                std::vector<ProbePass> probePasses;
                std::vector<int32_t> coarseChunk;
                for (size_t p0 = 0; ok && p0 < nprobe; p0 += maxProbePerChunk) {
                    size_t chunkProbe = std::min(maxProbePerChunk, nprobe - p0);
                    coarseChunk.resize((size_t)qCount * chunkProbe);
                    for (idx_t qi = 0; qi < qCount; ++qi) {
                        const size_t srcBase = (size_t)qi * nprobe + p0;
                        const size_t dstBase = (size_t)qi * chunkProbe;
                        std::memcpy(
                                coarseChunk.data() + dstBase,
                                dst + srcBase,
                                chunkProbe * sizeof(int32_t));
                    }

                    size_t coarseChunkBytes = coarseChunk.size() * sizeof(int32_t);
                    id<MTLBuffer> coarseChunkBuf = searchCoarseBuf_;
                    if (canAsyncProbeBatch) {
                        coarseChunkBuf =
                                [device newBufferWithLength:coarseChunkBytes
                                                    options:MTLResourceStorageModeShared];
                        if (!coarseChunkBuf) {
                            ok = false;
                            break;
                        }
                    } else {
                        ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseChunkBytes);
                        coarseChunkBuf = searchCoarseBuf_;
                        if (!coarseChunkBuf) {
                            ok = false;
                            break;
                        }
                    }
                    std::memcpy([coarseChunkBuf contents], coarseChunk.data(), coarseChunkBytes);

                    size_t perListBytesChunk =
                            (size_t)qCount * chunkProbe * (size_t)k * sizeof(float);
                    size_t perListIdxBytesChunk =
                            (size_t)qCount * chunkProbe * (size_t)k * sizeof(int64_t);
                    ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytesChunk);
                    ensureSearchBuf_(searchPerListIdxBuf_, searchPerListIdxCap_, perListIdxBytesChunk);
                    if (!searchPerListDistBuf_ || !searchPerListIdxBuf_) {
                        ok = false;
                        break;
                    }

                    const int chunkProbeI = checkedSizeToInt(
                            chunkProbe,
                            "MetalIndexIVFFlat: chunk nprobe exceeds int range");
                    if (canAsyncProbeBatch) {
                        id<MTLBuffer> passOutDistBuf =
                                [device newBufferWithLength:outDistBytes
                                                    options:MTLResourceStorageModeShared];
                        id<MTLBuffer> passOutIdxBuf =
                                [device newBufferWithLength:outIdxBytes
                                                    options:MTLResourceStorageModeShared];
                        if (!passOutDistBuf || !passOutIdxBuf) {
                            ok = false;
                            break;
                        }

                        bool chunkOk = runMetalIVFFlatScan(
                                device, queue,
                                searchQueriesBuf_,
                                gpuIvf_->codesBuffer(),
                                gpuIvf_->idsBuffer(),
                                gpuIvf_->listOffsetGpuBuffer(),
                                gpuIvf_->listLengthGpuBuffer(),
                                coarseChunkBuf,
                                qCountI, d, kI, chunkProbeI, isL2,
                                passOutDistBuf, passOutIdxBuf,
                                searchPerListDistBuf_, searchPerListIdxBuf_,
                                chunkInterleavedCodes,
                                chunkInterleavedOffsets,
                                false /* waitForCompletion */);
                        ++scanCalls;
                        ++asyncScanCalls;
                        if (!chunkOk) {
                            ok = false;
                            break;
                        }
                        probePasses.push_back(
                                ProbePass{p0, chunkProbe, coarseChunkBuf, passOutDistBuf, passOutIdxBuf});
                        continue;
                    }

                    if (needListChunk && numListPass > 1) {
                        std::vector<id<MTLBuffer>> passOutDistBufs(numListPass, nil);
                        std::vector<id<MTLBuffer>> passOutIdxBufs(numListPass, nil);
                        std::vector<id<MTLBuffer>> passListOffsetBufs(numListPass, nil);
                        std::vector<id<MTLBuffer>> passListLengthBufs(numListPass, nil);

                        for (size_t lp = 0; ok && lp < numListPass; ++lp) {
                            const uint64_t shift = (uint64_t)lp * (uint64_t)listChunk;
                            for (int li = 0; li < nlist; ++li) {
                                uint32_t off = baseListOffset[li];
                                uint32_t len = baseListLength[li];
                                if (!usedLists[(size_t)li]) {
                                    passOffset[(size_t)li] = off;
                                    passLength[(size_t)li] = 0;
                                    continue;
                                }
                                if (shift >= len) {
                                    passOffset[(size_t)li] = off + len;
                                    passLength[(size_t)li] = 0;
                                    continue;
                                }
                                const uint32_t delta = (uint32_t)shift;
                                const uint32_t rem = len - delta;
                                passOffset[(size_t)li] = off + delta;
                                passLength[(size_t)li] =
                                        std::min<uint32_t>((uint32_t)listChunk, rem);
                            }

                            passListOffsetBufs[lp] =
                                    [device newBufferWithLength:(size_t)nlist * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
                            passListLengthBufs[lp] =
                                    [device newBufferWithLength:(size_t)nlist * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
                            passOutDistBufs[lp] =
                                    [device newBufferWithLength:outDistBytes
                                                        options:MTLResourceStorageModeShared];
                            passOutIdxBufs[lp] =
                                    [device newBufferWithLength:outIdxBytes
                                                        options:MTLResourceStorageModeShared];
                            if (!passListOffsetBufs[lp] || !passListLengthBufs[lp] ||
                                !passOutDistBufs[lp] || !passOutIdxBufs[lp]) {
                                ok = false;
                                break;
                            }
                            std::memcpy([passListOffsetBufs[lp] contents],
                                        passOffset.data(),
                                        passOffset.size() * sizeof(uint32_t));
                            std::memcpy([passListLengthBufs[lp] contents],
                                        passLength.data(),
                                        passLength.size() * sizeof(uint32_t));

                            bool chunkOk = runMetalIVFFlatScan(
                                    device, queue,
                                    searchQueriesBuf_,
                                    gpuIvf_->codesBuffer(),
                                    gpuIvf_->idsBuffer(),
                                    passListOffsetBufs[lp],
                                    passListLengthBufs[lp],
                                    searchCoarseBuf_,
                                    qCountI, d, kI, chunkProbeI, isL2,
                                    passOutDistBufs[lp], passOutIdxBufs[lp],
                                    searchPerListDistBuf_, searchPerListIdxBuf_,
                                    chunkInterleavedCodes,
                                    chunkInterleavedOffsets,
                                    false /* waitForCompletion */);
                            ++scanCalls;
                            ++asyncScanCalls;
                            if (!chunkOk) {
                                ok = false;
                                break;
                            }
                        }

                        if (ok) {
                            resources_->synchronize();
                            ++asyncBatchSyncs;
                            for (size_t lp = 0; lp < numListPass; ++lp) {
                                const float* chunkDist = reinterpret_cast<const float*>(
                                        [passOutDistBufs[lp] contents]);
                                const int64_t* chunkIdx = reinterpret_cast<const int64_t*>(
                                        [passOutIdxBufs[lp] contents]);
                                for (idx_t qi = 0; qi < qCount; ++qi) {
                                    float* bestDist =
                                            chunkMergedDist.data() + (size_t)qi * (size_t)k;
                                    int64_t* bestIdx =
                                            chunkMergedIdx.data() + (size_t)qi * (size_t)k;
                                    for (idx_t j = 0; j < k; ++j) {
                                        const size_t pos =
                                                (size_t)qi * (size_t)k + (size_t)j;
                                        insertTopKCandidate(
                                                chunkDist[pos],
                                                chunkIdx[pos],
                                                isL2,
                                                bestDist,
                                                bestIdx,
                                                kI);
                                    }
                                }
                            }
                        }
                    } else {
                        for (size_t lp = 0; ok && lp < numListPass; ++lp) {
                            id<MTLBuffer> passListOffsetBuf = gpuIvf_->listOffsetGpuBuffer();
                            id<MTLBuffer> passListLengthBuf = gpuIvf_->listLengthGpuBuffer();
                            if (needListChunk) {
                                const uint64_t shift = (uint64_t)lp * (uint64_t)listChunk;
                                for (int li = 0; li < nlist; ++li) {
                                    uint32_t off = baseListOffset[li];
                                    uint32_t len = baseListLength[li];
                                    if (!usedLists[(size_t)li]) {
                                        passOffset[(size_t)li] = off;
                                        passLength[(size_t)li] = 0;
                                        continue;
                                    }
                                    if (shift >= len) {
                                        passOffset[(size_t)li] = off + len;
                                        passLength[(size_t)li] = 0;
                                        continue;
                                    }
                                    const uint32_t delta = (uint32_t)shift;
                                    const uint32_t rem = len - delta;
                                    passOffset[(size_t)li] = off + delta;
                                    passLength[(size_t)li] =
                                            std::min<uint32_t>((uint32_t)listChunk, rem);
                                }
                                std::memcpy([listOffsetPassBuf contents],
                                            passOffset.data(),
                                            passOffset.size() * sizeof(uint32_t));
                                std::memcpy([listLengthPassBuf contents],
                                            passLength.data(),
                                            passLength.size() * sizeof(uint32_t));
                                passListOffsetBuf = listOffsetPassBuf;
                                passListLengthBuf = listLengthPassBuf;
                            }

                            bool chunkOk = runMetalIVFFlatScan(
                                    device, queue,
                                    searchQueriesBuf_,
                                    gpuIvf_->codesBuffer(),
                                    gpuIvf_->idsBuffer(),
                                    passListOffsetBuf,
                                    passListLengthBuf,
                                    searchCoarseBuf_,
                                    qCountI, d, kI, chunkProbeI, isL2,
                                    searchOutDistBuf_, searchOutIdxBuf_,
                                    searchPerListDistBuf_, searchPerListIdxBuf_,
                                    chunkInterleavedCodes,
                                    chunkInterleavedOffsets);
                            ++scanCalls;
                            ++syncScanCalls;
                            if (!chunkOk) {
                                ok = false;
                                break;
                            }

                            const float* chunkDist = reinterpret_cast<const float*>(
                                    [searchOutDistBuf_ contents]);
                            const int64_t* chunkIdx = reinterpret_cast<const int64_t*>(
                                    [searchOutIdxBuf_ contents]);
                            for (idx_t qi = 0; qi < qCount; ++qi) {
                                float* bestDist =
                                        chunkMergedDist.data() + (size_t)qi * (size_t)k;
                                int64_t* bestIdx =
                                        chunkMergedIdx.data() + (size_t)qi * (size_t)k;
                                for (idx_t j = 0; j < k; ++j) {
                                    const size_t pos =
                                            (size_t)qi * (size_t)k + (size_t)j;
                                    insertTopKCandidate(
                                            chunkDist[pos],
                                            chunkIdx[pos],
                                            isL2,
                                            bestDist,
                                            bestIdx,
                                            kI);
                                }
                            }
                        }
                    }
                }
                if (ok && canAsyncProbeBatch && !probePasses.empty()) {
                    resources_->synchronize();
                    ++asyncBatchSyncs;
                    for (const auto& pass : probePasses) {
                        (void)pass.p0;
                        (void)pass.chunkProbe;
                        (void)pass.coarseBuf;
                        const float* chunkDist = reinterpret_cast<const float*>(
                                [pass.outDistBuf contents]);
                        const int64_t* chunkIdx = reinterpret_cast<const int64_t*>(
                                [pass.outIdxBuf contents]);
                        for (idx_t qi = 0; qi < qCount; ++qi) {
                            float* bestDist =
                                    chunkMergedDist.data() + (size_t)qi * (size_t)k;
                            int64_t* bestIdx =
                                    chunkMergedIdx.data() + (size_t)qi * (size_t)k;
                            for (idx_t j = 0; j < k; ++j) {
                                const size_t pos =
                                        (size_t)qi * (size_t)k + (size_t)j;
                                insertTopKCandidate(
                                        chunkDist[pos],
                                        chunkIdx[pos],
                                        isL2,
                                        bestDist,
                                        bestIdx,
                                        kI);
                            }
                        }
                    }
                }
            } else {
                ok = runMetalIVFFlatScan(
                        device, queue,
                        searchQueriesBuf_,
                        gpuIvf_->codesBuffer(),
                        gpuIvf_->idsBuffer(),
                        gpuIvf_->listOffsetGpuBuffer(),
                        gpuIvf_->listLengthGpuBuffer(),
                        searchCoarseBuf_,
                        qCountI, d, kI, nprobeI, isL2,
                        searchOutDistBuf_, searchOutIdxBuf_,
                        searchPerListDistBuf_, searchPerListIdxBuf_,
                        gpuIvf_->interleavedCodesBuffer(),
                        gpuIvf_->interleavedCodesOffsetBuffer());
                ++scanCalls;
                ++syncScanCalls;
            }
            if (!ok && !envelopeReason.empty() && !usedChunkedScan) {
                fallbackOrThrow(qBase, qCount, envelopeReason.c_str());
                continue;
            }
            if (ok && usedChunkedScan) {
                for (idx_t qi = 0; qi < qCount; ++qi) {
                    for (idx_t j = 0; j < k; ++j) {
                        const size_t pos = (size_t)qi * (size_t)k + (size_t)j;
                        const size_t globalPos = (size_t)(qBase + qi) * (size_t)k + (size_t)j;
                        const int64_t globalId = chunkMergedIdx[pos];
                        if (globalId < 0) {
                            labels[globalPos] = -1;
                        } else if (indicesOptions_ == faiss::gpu::INDICES_CPU) {
                            labels[globalPos] = decodeCpuLabelFromPair(cpuIndex_.get(), globalId);
                        } else if (indicesOptions_ == faiss::gpu::INDICES_32_BIT) {
                            labels[globalPos] = (idx_t)(int32_t)globalId;
                        } else {
                            labels[globalPos] = (idx_t)globalId;
                        }
                        distances[globalPos] = chunkMergedDist[pos];
                    }
                }
                continue;
            }
        }

        if (!ok) {
            fallbackOrThrow(qBase, qCount, "GPU IVF scan failed (runtime/kernel)");
            continue;
        }

        const float* outDistPtr = reinterpret_cast<const float*>(
                [searchOutDistBuf_ contents]);
        const int64_t* outIdxPtr = reinterpret_cast<const int64_t*>(
                [searchOutIdxBuf_ contents]);

        for (idx_t qi = 0; qi < qCount; ++qi) {
            for (idx_t j = 0; j < k; ++j) {
                size_t localPos = (size_t)qi * (size_t)k + (size_t)j;
                size_t globalPos = (size_t)(qBase + qi) * (size_t)k + (size_t)j;
                int64_t globalId = outIdxPtr[localPos];
                if (globalId < 0) {
                    labels[globalPos] = -1;
                } else if (indicesOptions_ == faiss::gpu::INDICES_CPU) {
                    labels[globalPos] =
                            decodeCpuLabelFromPair(cpuIndex_.get(), globalId);
                } else if (indicesOptions_ == faiss::gpu::INDICES_32_BIT) {
                    labels[globalPos] = (idx_t)(int32_t)globalId;
                } else {
                    labels[globalPos] = (idx_t)globalId;
                }
                distances[globalPos] = outDistPtr[localPos];
            }
        }
    }

    if (allowCpuFallback && logCpuFallback && fallbackCount > 0) {
        std::fprintf(
                stderr,
                "IVF_CPU_FALLBACK,api=search,count=%lld,first_reason=%s\n",
                (long long)fallbackCount,
                firstFallbackReason.c_str());
    }
    if (logSyncProfile) {
        const idx_t estimatedWaits = fullSearchCalls + syncScanCalls + asyncBatchSyncs;
        std::fprintf(
                stderr,
                "IVF_SYNC_PROFILE,api=search,full_calls=%lld,scan_calls=%lld,sync_scan_calls=%lld,async_scan_calls=%lld,async_batch_syncs=%lld,estimated_waits=%lld\n",
                (long long)fullSearchCalls,
                (long long)scanCalls,
                (long long)syncScanCalls,
                (long long)asyncScanCalls,
                (long long)asyncBatchSyncs,
                (long long)estimatedWaits);
    }
}

void MetalIndexIVFFlat::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(assign);
    FAISS_THROW_IF_NOT_MSG(stats == nullptr, "IVF stats not supported");
    FAISS_THROW_IF_NOT_MSG(
            !store_pairs,
            "MetalIndexIVFFlat::search_preassigned does not currently support store_pairs");
    if (params) {
        FAISS_THROW_IF_NOT_FMT(
                params->max_codes == 0,
                "Metal IVF index does not currently support "
                "SearchParametersIVF::max_codes (passed %zu, must be 0)",
                params->max_codes);
    }
    FAISS_THROW_IF_NOT_MSG(
            k <= kIVFFlatSupportedMaxK,
            "MetalIndexIVFFlat supports k <= 1024; larger k is not yet supported");

    const float inf    = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();
    const bool isL2 = (metric_type == METRIC_L2);

    if (cpuIndex_->ntotal == 0 || n == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i]    = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    size_t nprobe = cpuIndex_->nprobe;
    if (params && params->nprobe > 0) {
        nprobe = params->nprobe;
    }
    nprobe = std::min(nprobe, cpuIndex_->nlist);

    id<MTLDevice>       device = resources_->getDevice();
    id<MTLCommandQueue> queue  = resources_->getCommandQueue();
    if (gpuIvf_) {
        gpuIvf_->ensureInterleavedLayoutUpToDate();
    }

    const bool hasFlatCodes = gpuIvf_ && gpuIvf_->codesBuffer();
    const bool hasInterleavedCodes =
            gpuIvf_ && gpuIvf_->interleavedCodesBuffer() &&
            gpuIvf_->interleavedCodesOffsetBuffer();
    const bool hasScanCodes = hasFlatCodes || hasInterleavedCodes;

    auto cpuFallbackSearch = [&](idx_t qBase, idx_t qCount) {
        const float* xTile = x + (size_t)qBase * (size_t)d;
        const idx_t* assignTile = assign + (size_t)qBase * nprobe;
        const float* centroidTile =
                centroid_dis ? (centroid_dis + (size_t)qBase * nprobe) : nullptr;
        float* distTile = distances + (size_t)qBase * (size_t)k;
        idx_t* labelsTile = labels + (size_t)qBase * (size_t)k;
        cpuIndex_->search_preassigned(
                qCount,
                xTile,
                k,
                assignTile,
                centroidTile,
                distTile,
                labelsTile,
                indicesOptions_ == faiss::gpu::INDICES_IVF ? true : store_pairs,
                params,
                stats);
    };
    const bool allowCpuFallback = allowCpuFallbackForIvf();
    const bool logCpuFallback = logCpuFallbackForIvf();
    const bool logSyncProfile = logSyncProfileForIvf();
    const size_t exactCandidateBudget = getIvfExactCandidateBudget();
    idx_t fallbackCount = 0;
    std::string firstFallbackReason;
    idx_t scanCalls = 0;
    idx_t syncScanCalls = 0;
    idx_t asyncScanCalls = 0;
    idx_t asyncBatchSyncs = 0;
    auto fallbackOrThrow = [&](idx_t qBase, idx_t qCount, const char* reason) {
        if (!allowCpuFallback) {
            FAISS_THROW_FMT(
                    "MetalIndexIVFFlat::search_preassigned requires GPU execution but hit "
                    "fallback (%s) at qBase=%lld qCount=%lld",
                    reason,
                    (long long)qBase,
                    (long long)qCount);
        }
        ++fallbackCount;
        if (firstFallbackReason.empty()) {
            firstFallbackReason = reason ? reason : "unknown";
        }
        cpuFallbackSearch(qBase, qCount);
    };

    if (!device || !queue || !gpuIvf_ || !hasScanCodes || !gpuIvf_->idsBuffer() ||
        !gpuIvf_->listOffsetGpuBuffer() || !gpuIvf_->listLengthGpuBuffer()) {
        fallbackOrThrow(0, n, "missing Metal device/queue or IVF buffers");
        return;
    }

    const int nlist = (int)cpuIndex_->nlist;
    const size_t tileRows = chooseIvfPreassignedTileRows((size_t)n, d, k, nprobe);
    for (idx_t qBase = 0; qBase < n; qBase += (idx_t)tileRows) {
        idx_t qCount = std::min<idx_t>((idx_t)tileRows, n - qBase);
        const float* xTile = x + (size_t)qBase * (size_t)d;
        const idx_t* assignTile = assign + (size_t)qBase * nprobe;

        size_t queriesBytes = (size_t)qCount * (size_t)d * sizeof(float);
        size_t outDistBytes = (size_t)qCount * (size_t)k * sizeof(float);
        size_t outIdxBytes = (size_t)qCount * (size_t)k * sizeof(int64_t);
        size_t perListBytes = (size_t)qCount * nprobe * (size_t)k * sizeof(float);
        size_t perListIdxB = (size_t)qCount * nprobe * (size_t)k * sizeof(int64_t);
        size_t coarseBytes = (size_t)qCount * nprobe * sizeof(int32_t);

        ensureSearchBuf_(searchQueriesBuf_, searchQueriesCap_, queriesBytes);
        ensureSearchBuf_(searchOutDistBuf_, searchOutDistCap_, outDistBytes);
        ensureSearchBuf_(searchOutIdxBuf_, searchOutIdxCap_, outIdxBytes);
        ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytes);
        ensureSearchBuf_(searchPerListIdxBuf_, searchPerListIdxCap_, perListIdxB);
        ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseBytes);

        if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
            !searchPerListDistBuf_ || !searchPerListIdxBuf_ || !searchCoarseBuf_) {
            fallbackOrThrow(qBase, qCount, "failed to allocate tiled preassigned buffers");
            continue;
        }

        std::memcpy([searchQueriesBuf_ contents], xTile, queriesBytes);
        const int qCountI = checkedIdxToInt(
                qCount,
                "MetalIndexIVFFlat::search_preassigned: qCount exceeds int range");
        const int kI = checkedIdxToInt(
                k, "MetalIndexIVFFlat::search_preassigned: k exceeds int range");
        const int nprobeI = checkedSizeToInt(
                nprobe,
                "MetalIndexIVFFlat::search_preassigned: nprobe exceeds int range");

        auto* coarseDst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
        for (size_t i = 0; i < (size_t)qCount * nprobe; ++i) {
            FAISS_THROW_IF_NOT_MSG(
                    assignTile[i] >= (idx_t)std::numeric_limits<int32_t>::min() &&
                            assignTile[i] <=
                                    (idx_t)std::numeric_limits<int32_t>::max(),
                    "MetalIndexIVFFlat: preassigned list id exceeds int32 range");
            coarseDst[i] = (int32_t)assignTile[i];
        }

        const bool useInterleaved =
                gpuIvf_->interleavedCodesBuffer() != nil &&
                gpuIvf_->interleavedCodesOffsetBuffer() != nil;
        const std::string envelopeReason = explainIvfScanFailureEnvelope(
                d,
                k,
                exactCandidateBudget,
                nprobe,
                useInterleaved,
                cpuIndex_.get(),
                assignTile,
                (size_t)qCount * nprobe);
        bool ok = false;
        bool usedChunkedScan = false;
        std::vector<float> chunkMergedDist;
        std::vector<int64_t> chunkMergedIdx;
        std::vector<uint8_t> usedLists;
        size_t maxSelectedListSize = 0;
        idx_t badListId = -1;
        bool haveListStats = selectedListStats(
                cpuIndex_.get(),
                assignTile,
                (size_t)qCount * nprobe,
                usedLists,
                maxSelectedListSize,
                badListId);
            const bool needProbeChunk =
                nprobe > 0 && (size_t)k > (exactCandidateBudget / nprobe);
            const bool needListChunk =
                haveListStats && maxSelectedListSize > exactCandidateBudget;

        if (needProbeChunk || needListChunk) {
            const size_t maxProbePerChunk = needProbeChunk
                    ? std::max<size_t>(1, exactCandidateBudget / (size_t)k)
                    : nprobe;
            usedChunkedScan = true;
            chunkMergedDist.resize((size_t)qCount * (size_t)k, isL2 ? inf : negInf);
            chunkMergedIdx.resize((size_t)qCount * (size_t)k, -1);
            ok = true;

            const size_t listChunk = exactCandidateBudget;
            const size_t numListPass = needListChunk
                    ? ((maxSelectedListSize + listChunk - 1) / listChunk)
                    : 1;
            // List-chunk remapping only updates list offset/length metadata.
            // Interleaved offset metadata is list-base-relative, so force flat
            // code path for list-chunked passes to keep ids/codes aligned.
            id<MTLBuffer> chunkInterleavedCodes = needListChunk
                    ? nil
                    : gpuIvf_->interleavedCodesBuffer();
            id<MTLBuffer> chunkInterleavedOffsets = needListChunk
                    ? nil
                    : gpuIvf_->interleavedCodesOffsetBuffer();
            const uint32_t* baseListOffset = reinterpret_cast<const uint32_t*>(
                    [gpuIvf_->listOffsetGpuBuffer() contents]);
            const uint32_t* baseListLength = reinterpret_cast<const uint32_t*>(
                    [gpuIvf_->listLengthGpuBuffer() contents]);
            id<MTLBuffer> listOffsetPassBuf = nil;
            id<MTLBuffer> listLengthPassBuf = nil;
            std::vector<uint32_t> passOffset;
            std::vector<uint32_t> passLength;
            if (needListChunk) {
                size_t metaBytes = (size_t)nlist * sizeof(uint32_t);
                listOffsetPassBuf = [device newBufferWithLength:metaBytes
                                                         options:MTLResourceStorageModeShared];
                listLengthPassBuf = [device newBufferWithLength:metaBytes
                                                         options:MTLResourceStorageModeShared];
                if (!listOffsetPassBuf || !listLengthPassBuf) {
                    ok = false;
                } else {
                    passOffset.resize((size_t)nlist);
                    passLength.resize((size_t)nlist);
                }
            }

            std::vector<int32_t> coarseChunk;
            const int32_t* fullCoarse = reinterpret_cast<const int32_t*>([searchCoarseBuf_ contents]);

            for (size_t p0 = 0; ok && p0 < nprobe; p0 += maxProbePerChunk) {
                size_t chunkProbe = std::min(maxProbePerChunk, nprobe - p0);
                coarseChunk.resize((size_t)qCount * chunkProbe);
                for (idx_t qi = 0; qi < qCount; ++qi) {
                    const size_t srcBase = (size_t)qi * nprobe + p0;
                    const size_t dstBase = (size_t)qi * chunkProbe;
                    std::memcpy(
                            coarseChunk.data() + dstBase,
                            fullCoarse + srcBase,
                            chunkProbe * sizeof(int32_t));
                }

                size_t coarseChunkBytes = coarseChunk.size() * sizeof(int32_t);
                ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseChunkBytes);
                if (!searchCoarseBuf_) {
                    ok = false;
                    break;
                }
                std::memcpy([searchCoarseBuf_ contents], coarseChunk.data(), coarseChunkBytes);

                size_t perListBytesChunk =
                        (size_t)qCount * chunkProbe * (size_t)k * sizeof(float);
                size_t perListIdxBytesChunk =
                        (size_t)qCount * chunkProbe * (size_t)k * sizeof(int64_t);
                ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytesChunk);
                ensureSearchBuf_(searchPerListIdxBuf_, searchPerListIdxCap_, perListIdxBytesChunk);
                if (!searchPerListDistBuf_ || !searchPerListIdxBuf_) {
                    ok = false;
                    break;
                }

                const int chunkProbeI = checkedSizeToInt(
                        chunkProbe,
                        "MetalIndexIVFFlat::search_preassigned: chunk nprobe exceeds int range");
                if (needListChunk && numListPass > 1) {
                    std::vector<id<MTLBuffer>> passOutDistBufs(numListPass, nil);
                    std::vector<id<MTLBuffer>> passOutIdxBufs(numListPass, nil);
                    std::vector<id<MTLBuffer>> passListOffsetBufs(numListPass, nil);
                    std::vector<id<MTLBuffer>> passListLengthBufs(numListPass, nil);

                    for (size_t lp = 0; ok && lp < numListPass; ++lp) {
                        const uint64_t shift = (uint64_t)lp * (uint64_t)listChunk;
                        for (int li = 0; li < nlist; ++li) {
                            uint32_t off = baseListOffset[li];
                            uint32_t len = baseListLength[li];
                            if (!usedLists[(size_t)li]) {
                                passOffset[(size_t)li] = off;
                                passLength[(size_t)li] = 0;
                                continue;
                            }
                            if (shift >= len) {
                                passOffset[(size_t)li] = off + len;
                                passLength[(size_t)li] = 0;
                                continue;
                            }
                            const uint32_t delta = (uint32_t)shift;
                            const uint32_t rem = len - delta;
                            passOffset[(size_t)li] = off + delta;
                            passLength[(size_t)li] =
                                    std::min<uint32_t>((uint32_t)listChunk, rem);
                        }

                        passListOffsetBufs[lp] =
                                [device newBufferWithLength:(size_t)nlist * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
                        passListLengthBufs[lp] =
                                [device newBufferWithLength:(size_t)nlist * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
                        passOutDistBufs[lp] =
                                [device newBufferWithLength:outDistBytes
                                                    options:MTLResourceStorageModeShared];
                        passOutIdxBufs[lp] =
                                [device newBufferWithLength:outIdxBytes
                                                    options:MTLResourceStorageModeShared];
                        if (!passListOffsetBufs[lp] || !passListLengthBufs[lp] ||
                            !passOutDistBufs[lp] || !passOutIdxBufs[lp]) {
                            ok = false;
                            break;
                        }
                        std::memcpy([passListOffsetBufs[lp] contents],
                                    passOffset.data(),
                                    passOffset.size() * sizeof(uint32_t));
                        std::memcpy([passListLengthBufs[lp] contents],
                                    passLength.data(),
                                    passLength.size() * sizeof(uint32_t));

                        bool chunkOk = runMetalIVFFlatScan(
                                device, queue,
                                searchQueriesBuf_,
                                gpuIvf_->codesBuffer(),
                                gpuIvf_->idsBuffer(),
                                passListOffsetBufs[lp],
                                passListLengthBufs[lp],
                                searchCoarseBuf_,
                                qCountI, d, kI, chunkProbeI, isL2,
                                passOutDistBufs[lp], passOutIdxBufs[lp],
                                searchPerListDistBuf_, searchPerListIdxBuf_,
                                chunkInterleavedCodes,
                                chunkInterleavedOffsets,
                                false /* waitForCompletion */);
                        ++scanCalls;
                        ++asyncScanCalls;
                        if (!chunkOk) {
                            ok = false;
                            break;
                        }
                    }

                    if (ok) {
                        resources_->synchronize();
                        ++asyncBatchSyncs;
                        for (size_t lp = 0; lp < numListPass; ++lp) {
                            const float* chunkDist = reinterpret_cast<const float*>(
                                    [passOutDistBufs[lp] contents]);
                            const int64_t* chunkIdx = reinterpret_cast<const int64_t*>(
                                    [passOutIdxBufs[lp] contents]);
                            for (idx_t qi = 0; qi < qCount; ++qi) {
                                float* bestDist =
                                        chunkMergedDist.data() + (size_t)qi * (size_t)k;
                                int64_t* bestIdx =
                                        chunkMergedIdx.data() + (size_t)qi * (size_t)k;
                                for (idx_t j = 0; j < k; ++j) {
                                    const size_t pos = (size_t)qi * (size_t)k + (size_t)j;
                                    insertTopKCandidate(
                                            chunkDist[pos],
                                            chunkIdx[pos],
                                            isL2,
                                            bestDist,
                                            bestIdx,
                                            kI);
                                }
                            }
                        }
                    }
                } else {
                    for (size_t lp = 0; ok && lp < numListPass; ++lp) {
                        id<MTLBuffer> passListOffsetBuf = gpuIvf_->listOffsetGpuBuffer();
                        id<MTLBuffer> passListLengthBuf = gpuIvf_->listLengthGpuBuffer();
                        if (needListChunk) {
                            const uint64_t shift = (uint64_t)lp * (uint64_t)listChunk;
                            for (int li = 0; li < nlist; ++li) {
                                uint32_t off = baseListOffset[li];
                                uint32_t len = baseListLength[li];
                                if (!usedLists[(size_t)li]) {
                                    passOffset[(size_t)li] = off;
                                    passLength[(size_t)li] = 0;
                                    continue;
                                }
                                if (shift >= len) {
                                    passOffset[(size_t)li] = off + len;
                                    passLength[(size_t)li] = 0;
                                    continue;
                                }
                                const uint32_t delta = (uint32_t)shift;
                                const uint32_t rem = len - delta;
                                passOffset[(size_t)li] = off + delta;
                                passLength[(size_t)li] =
                                        std::min<uint32_t>((uint32_t)listChunk, rem);
                            }
                            std::memcpy([listOffsetPassBuf contents],
                                        passOffset.data(),
                                        passOffset.size() * sizeof(uint32_t));
                            std::memcpy([listLengthPassBuf contents],
                                        passLength.data(),
                                        passLength.size() * sizeof(uint32_t));
                            passListOffsetBuf = listOffsetPassBuf;
                            passListLengthBuf = listLengthPassBuf;
                        }

                        bool chunkOk = runMetalIVFFlatScan(
                                device, queue,
                                searchQueriesBuf_,
                                gpuIvf_->codesBuffer(),
                                gpuIvf_->idsBuffer(),
                                passListOffsetBuf,
                                passListLengthBuf,
                                searchCoarseBuf_,
                                qCountI, d, kI, chunkProbeI, isL2,
                                searchOutDistBuf_, searchOutIdxBuf_,
                                searchPerListDistBuf_, searchPerListIdxBuf_,
                                chunkInterleavedCodes,
                                chunkInterleavedOffsets);
                        ++scanCalls;
                        ++syncScanCalls;
                        if (!chunkOk) {
                            ok = false;
                            break;
                        }

                        const float* chunkDist = reinterpret_cast<const float*>(
                                [searchOutDistBuf_ contents]);
                        const int64_t* chunkIdx = reinterpret_cast<const int64_t*>(
                                [searchOutIdxBuf_ contents]);
                        for (idx_t qi = 0; qi < qCount; ++qi) {
                            float* bestDist = chunkMergedDist.data() + (size_t)qi * (size_t)k;
                            int64_t* bestIdx = chunkMergedIdx.data() + (size_t)qi * (size_t)k;
                            for (idx_t j = 0; j < k; ++j) {
                                const size_t pos = (size_t)qi * (size_t)k + (size_t)j;
                                insertTopKCandidate(
                                        chunkDist[pos],
                                        chunkIdx[pos],
                                        isL2,
                                        bestDist,
                                        bestIdx,
                                        kI);
                            }
                        }
                    }
                }
            }
        } else {
            ok = runMetalIVFFlatScan(
                    device, queue,
                    searchQueriesBuf_,
                    gpuIvf_->codesBuffer(),
                    gpuIvf_->idsBuffer(),
                    gpuIvf_->listOffsetGpuBuffer(),
                    gpuIvf_->listLengthGpuBuffer(),
                    searchCoarseBuf_,
                    qCountI, d, kI, nprobeI, isL2,
                    searchOutDistBuf_, searchOutIdxBuf_,
                    searchPerListDistBuf_, searchPerListIdxBuf_,
                    gpuIvf_->interleavedCodesBuffer(),
                    gpuIvf_->interleavedCodesOffsetBuffer());
            ++scanCalls;
            ++syncScanCalls;
        }
        if (!ok && !envelopeReason.empty() && !usedChunkedScan) {
            fallbackOrThrow(qBase, qCount, envelopeReason.c_str());
            continue;
        }

        if (!ok) {
            fallbackOrThrow(qBase, qCount, "GPU IVF scan failed (runtime/kernel)");
            continue;
        }

        if (usedChunkedScan) {
            for (idx_t qi = 0; qi < qCount; ++qi) {
                for (idx_t j = 0; j < k; ++j) {
                    const size_t pos = (size_t)qi * (size_t)k + (size_t)j;
                    const size_t globalPos = (size_t)(qBase + qi) * (size_t)k + (size_t)j;
                    const int64_t globalId = chunkMergedIdx[pos];
                    if (globalId < 0) {
                        labels[globalPos] = -1;
                    } else if (indicesOptions_ == faiss::gpu::INDICES_CPU) {
                        labels[globalPos] = decodeCpuLabelFromPair(cpuIndex_.get(), globalId);
                    } else if (indicesOptions_ == faiss::gpu::INDICES_32_BIT) {
                        labels[globalPos] = (idx_t)(int32_t)globalId;
                    } else {
                        labels[globalPos] = (idx_t)globalId;
                    }
                    distances[globalPos] = chunkMergedDist[pos];
                }
            }
            continue;
        }

        const float* outDistPtr = reinterpret_cast<const float*>(
                [searchOutDistBuf_ contents]);
        const int64_t* outIdxPtr = reinterpret_cast<const int64_t*>(
                [searchOutIdxBuf_ contents]);
        for (idx_t qi = 0; qi < qCount; ++qi) {
            for (idx_t j = 0; j < k; ++j) {
                size_t localPos = (size_t)qi * (size_t)k + (size_t)j;
                size_t globalPos = (size_t)(qBase + qi) * (size_t)k + (size_t)j;
                int64_t globalId = outIdxPtr[localPos];
                if (globalId < 0) {
                    labels[globalPos] = -1;
                } else if (indicesOptions_ == faiss::gpu::INDICES_CPU) {
                    labels[globalPos] =
                            decodeCpuLabelFromPair(cpuIndex_.get(), globalId);
                } else if (indicesOptions_ == faiss::gpu::INDICES_32_BIT) {
                    labels[globalPos] = (idx_t)(int32_t)globalId;
                } else {
                    labels[globalPos] = (idx_t)globalId;
                }
                distances[globalPos] = outDistPtr[localPos];
            }
        }
    }

    if (allowCpuFallback && logCpuFallback && fallbackCount > 0) {
        std::fprintf(
                stderr,
                "IVF_CPU_FALLBACK,api=search_preassigned,count=%lld,first_reason=%s\n",
                (long long)fallbackCount,
                firstFallbackReason.c_str());
    }
    if (logSyncProfile) {
        const idx_t estimatedWaits = syncScanCalls + asyncBatchSyncs;
        std::fprintf(
                stderr,
                "IVF_SYNC_PROFILE,api=search_preassigned,scan_calls=%lld,sync_scan_calls=%lld,async_scan_calls=%lld,async_batch_syncs=%lld,estimated_waits=%lld\n",
                (long long)scanCalls,
                (long long)syncScanCalls,
                (long long)asyncScanCalls,
                (long long)asyncBatchSyncs,
                (long long)estimatedWaits);
    }
}

void MetalIndexIVFFlat::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "MetalIndexIVFFlat: no internal index");
    cpuIndex_->reconstruct(key, recons);
}

void MetalIndexIVFFlat::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "MetalIndexIVFFlat: no internal index");
    cpuIndex_->reconstruct_n(i0, ni, recons);
}

void MetalIndexIVFFlat::updateQuantizer() {
    uploadCentroids_();
}

std::vector<idx_t> MetalIndexIVFFlat::getListIndices(idx_t listId) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(listId >= 0 && listId < cpuIndex_->nlist);
    size_t ls = cpuIndex_->invlists->list_size(listId);
    if (ls == 0) return {};
    ScopedIds ids(cpuIndex_->invlists, (size_t)listId);
    return ids.ptr ? std::vector<idx_t>(ids.ptr, ids.ptr + ls) : std::vector<idx_t>{};
}

std::vector<float> MetalIndexIVFFlat::getListVectorData(idx_t listId) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(listId >= 0 && listId < cpuIndex_->nlist);
    size_t ls = cpuIndex_->invlists->list_size(listId);
    if (ls == 0) return {};
    ScopedCodes codes(cpuIndex_->invlists, (size_t)listId);
    if (!codes.ptr) {
        return {};
    }
    size_t floatCount = ls * (size_t)d;
    const float* fptr = reinterpret_cast<const float*>(codes.ptr);
    return std::vector<float>(fptr, fptr + floatCount);
}

void MetalIndexIVFFlat::reclaimMemory() {
    // No-op for now: Metal unified memory doesn't benefit from explicit
    // reclaim in the same way as discrete CUDA GPU memory.
}

void MetalIndexIVFFlat::reserveMemory(idx_t numVecs) {
    if (gpuIvf_) {
        gpuIvf_->reserveMemory(numVecs);
    }
}

idx_t MetalIndexIVFFlat::nlist() const {
    return cpuIndex_ ? cpuIndex_->nlist : 0;
}

size_t MetalIndexIVFFlat::nprobe() const {
    return cpuIndex_ ? cpuIndex_->nprobe : 1;
}

bool MetalIndexIVFFlat::interleavedLayout() const {
    return interleavedLayout_;
}

faiss::gpu::IndicesOptions MetalIndexIVFFlat::indicesOptions() const {
    return indicesOptions_;
}

MetalIndexIVFFlat::AppendDebugStats MetalIndexIVFFlat::appendDebugStats() const {
    AppendDebugStats out{};
    if (!gpuIvf_) {
        return out;
    }
    const auto& s = gpuIvf_->appendDebugStats();
    out.relayoutEvents = s.relayoutEvents;
    out.movedLists = s.movedLists;
    out.movedVectors = s.movedVectors;
    out.reusedSegmentAllocs = s.reusedSegmentAllocs;
    out.tailSegmentAllocs = s.tailSegmentAllocs;
    out.reusedCapacityVecs = s.reusedCapacityVecs;
    out.tailCapacityVecs = s.tailCapacityVecs;
    out.tailShrinkEvents = s.tailShrinkEvents;
    out.tailShrunkVecs = s.tailShrunkVecs;
    return out;
}

void MetalIndexIVFFlat::resetAppendDebugStats() {
    if (gpuIvf_) {
        gpuIvf_->resetAppendDebugStats();
    }
}

void MetalIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* src) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(src);
    FAISS_THROW_IF_NOT_FMT(
            src->nlist == cpuIndex_->nlist,
            "copyFrom: nlist mismatch (%zd vs %zd)",
            (size_t)src->nlist,
            (size_t)cpuIndex_->nlist);
    reset();

    if (!src->is_trained) {
        is_trained = false;
        return;
    }

    // Copy quantizer centroids (allow non-IndexFlat CPU coarse quantizers by
    // reconstructing centroid vectors).
    FAISS_THROW_IF_NOT_MSG(src->quantizer, "copyFrom: source quantizer is null");
    auto* ourQ = cpuIndex_->quantizer;
    FAISS_THROW_IF_NOT_MSG(ourQ, "copyFrom: internal quantizer is null");
    ourQ->reset();
    if (src->nlist > 0) {
        std::vector<float> coarse((size_t)src->nlist * d);
        src->quantizer->reconstruct_n(0, src->nlist, coarse.data());
        if (!ourQ->is_trained) {
            ourQ->train(src->nlist, coarse.data());
        }
        ourQ->add(src->nlist, coarse.data());
    }
    cpuIndex_->is_trained = true;
    cpuIndex_->nprobe = src->nprobe;
    is_trained = true;

    // Gather all vectors from inverted lists for a single GPU upload.
    size_t totalN = 0;
    for (size_t l = 0; l < (size_t)src->nlist; ++l) {
        totalN += src->invlists->list_size(l);
    }

    if (totalN > 0) {
        std::vector<float> allCodes(totalN * (size_t)d);
        std::vector<idx_t> allListNos(totalN);
        std::vector<idx_t> allIds(totalN);
        size_t pos = 0;

        for (size_t l = 0; l < (size_t)src->nlist; ++l) {
            size_t ls = src->invlists->list_size(l);
            if (ls == 0) {
                continue;
            }
            ScopedCodes codes(src->invlists, l);
            ScopedIds ids(src->invlists, l);

            cpuIndex_->invlists->add_entries(l, ls, ids.ptr, codes.ptr);

            std::memcpy(
                    allCodes.data() + pos * (size_t)d,
                    codes.ptr,
                    ls * (size_t)d * sizeof(float));
            std::memcpy(allIds.data() + pos, ids.ptr, ls * sizeof(idx_t));
            for (size_t i = 0; i < ls; ++i) {
                allListNos[pos + i] = (idx_t)l;
            }
            pos += ls;
        }

        cpuIndex_->ntotal = (idx_t)totalN;
        ntotal = (idx_t)totalN;

        if (gpuIvf_) {
            gpuIvf_->appendVectors(
                    (idx_t)totalN,
                    allCodes.data(),
                    allListNos.data(),
                    allIds.data());
        }
    }

    uploadCentroids_();
}

void MetalIndexIVFFlat::copyTo(faiss::IndexIVFFlat* dst) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(dst);

    auto* srcQ = cpuIndex_->quantizer;
    auto* dstQ = dst->quantizer;
    FAISS_THROW_IF_NOT_MSG(srcQ, "copyTo: internal quantizer is null");
    FAISS_THROW_IF_NOT_MSG(dstQ, "copyTo: destination quantizer is null");

    dstQ->reset();
    if (srcQ->ntotal > 0) {
        std::vector<float> coarse((size_t)srcQ->ntotal * d);
        srcQ->reconstruct_n(0, srcQ->ntotal, coarse.data());
        if (!dstQ->is_trained) {
            dstQ->train(srcQ->ntotal, coarse.data());
        }
        dstQ->add(srcQ->ntotal, coarse.data());
    }

    dst->metric_type = cpuIndex_->metric_type;
    dst->metric_arg = cpuIndex_->metric_arg;
    dst->d = cpuIndex_->d;
    dst->nlist = cpuIndex_->nlist;
    dst->nprobe = cpuIndex_->nprobe;
    dst->is_trained = cpuIndex_->is_trained;

    for (size_t l = 0; l < (size_t)cpuIndex_->nlist; ++l) {
        size_t ls = cpuIndex_->invlists->list_size(l);
        if (ls == 0) {
            continue;
        }
        ScopedCodes codes(cpuIndex_->invlists, l);
        ScopedIds ids(cpuIndex_->invlists, l);
        dst->invlists->add_entries(l, ls, ids.ptr, codes.ptr);
    }
    dst->ntotal = cpuIndex_->ntotal;
}

} // namespace gpu_metal
} // namespace faiss
