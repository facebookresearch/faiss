// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexIVFPQ.h"

#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/MetalDistance.h>
#include <faiss/gpu_metal/impl/MetalIVFPQ.h>
#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace faiss {
namespace gpu_metal {

namespace {
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
} // namespace

// ============================================================
//  Constructors
// ============================================================

void MetalIndexIVFPQ::verifyPQSettings_() const {
    int M = cpuIndex_ ? (int)cpuIndex_->pq.M : 0;
    int nbits = cpuIndex_ ? (int)cpuIndex_->pq.nbits : 0;
    idx_t nl = cpuIndex_ ? cpuIndex_->nlist : 0;

    FAISS_THROW_IF_NOT_MSG(nl > 0, "MetalIndexIVFPQ: nlist must be > 0");
    FAISS_THROW_IF_NOT_FMT(
            nbits == 8,
            "MetalIndexIVFPQ: only 8-bit PQ codes are supported (got %d)",
            nbits);
    FAISS_THROW_IF_NOT_FMT(
            d % M == 0,
            "MetalIndexIVFPQ: sub-quantizers (%d) must evenly divide "
            "dimensions (%d)",
            M,
            d);
    FAISS_THROW_IF_NOT_FMT(
            M > 0 && M <= 64,
            "MetalIndexIVFPQ: sub-quantizers must be in [1, 64] (got %d)",
            M);
}

MetalIndexIVFPQ::MetalIndexIVFPQ(
        std::shared_ptr<MetalResources> resources,
        int dims,
        idx_t nlist,
        int M,
        int nbitsPerIdx,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, metricArg, config) {
    FAISS_THROW_IF_NOT(nbitsPerIdx == 8);
    FAISS_THROW_IF_NOT(dims % M == 0);

    faiss::IndexFlat* quantizer = (metric == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dims)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dims);
    cpuIndex_ = std::make_unique<faiss::IndexIVFPQ>(
            quantizer,
            (size_t)dims,
            (size_t)nlist,
            (size_t)M,
            (size_t)nbitsPerIdx);
    cpuIndex_->own_fields = true;
    verifyPQSettings_();
    gpuIvf_ = std::make_unique<MetalIVFPQImpl>(
            resources, dims, nlist, M, nbitsPerIdx, metric, metricArg);
}

MetalIndexIVFPQ::MetalIndexIVFPQ(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexIVFPQ* cpuIndex,
        MetalIndexConfig config)
        : MetalIndex(
                  resources,
                  (int)cpuIndex->d,
                  cpuIndex->metric_type,
                  cpuIndex->metric_arg,
                  config) {
    FAISS_THROW_IF_NOT(cpuIndex->pq.nbits == 8);

    int M = (int)cpuIndex->pq.M;
    faiss::IndexFlat* quantizer =
            (cpuIndex->metric_type == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP((int)cpuIndex->d)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2((int)cpuIndex->d);
    cpuIndex_ = std::make_unique<faiss::IndexIVFPQ>(
            quantizer,
            cpuIndex->d,
            cpuIndex->nlist,
            (size_t)M,
            cpuIndex->pq.nbits);
    cpuIndex_->own_fields = true;
    verifyPQSettings_();
    gpuIvf_ = std::make_unique<MetalIVFPQImpl>(
            resources,
            (int)cpuIndex->d,
            cpuIndex->nlist,
            M,
            (int)cpuIndex->pq.nbits,
            cpuIndex->metric_type,
            cpuIndex->metric_arg);
    copyFrom(cpuIndex);
}

MetalIndexIVFPQ::~MetalIndexIVFPQ() = default;

// ============================================================
//  GPU buffer helpers
// ============================================================

void MetalIndexIVFPQ::ensureSearchBuf_(
        id<MTLBuffer>& buf,
        size_t& cap,
        size_t needed) const {
    if (buf != nil && cap >= needed)
        return;
    size_t newCap = std::max(needed, cap * 2);
    id<MTLDevice> device = resources_->getDevice();
    buf = [device newBufferWithLength:newCap
                              options:MTLResourceStorageModeShared];
    cap = buf ? newCap : 0;
}

void MetalIndexIVFPQ::uploadCentroids_() const {
    if (!cpuIndex_ || !cpuIndex_->quantizer || !resources_)
        return;
    auto* flatQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    if (!flatQ || flatQ->ntotal == 0) {
        centroidBuf_ = nil;
        return;
    }
    size_t bytes = (size_t)flatQ->ntotal * (size_t)d * sizeof(float);
    id<MTLDevice> device = resources_->getDevice();
    if (!device)
        return;
    centroidBuf_ = [device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
    if (centroidBuf_) {
        std::memcpy([centroidBuf_ contents], flatQ->get_xb(), bytes);
    }
}

void MetalIndexIVFPQ::uploadPQCentroids_() const {
    if (!cpuIndex_ || !gpuIvf_)
        return;
    const auto& pq = cpuIndex_->pq;
    if (pq.centroids.empty())
        return;
    gpuIvf_->setPQCentroids(pq.centroids.data());
}

void MetalIndexIVFPQ::precomputeTerm2_() const {
    // GPU analogue of faiss::IndexIVFPQ::precompute_table(): the
    // query-independent term ||pq||^2 + 2<c, pq> of the L2 by_residual
    // decomposition, built once per trained index.
    term2Buf_ = nil;
    if (!cpuIndex_ || !resources_ || !gpuIvf_)
        return;
    if (cpuIndex_->metric_type != METRIC_L2 || !cpuIndex_->by_residual)
        return;
    if (!centroidBuf_ || !gpuIvf_->pqCentroidsBuffer())
        return;

    const int M = (int)cpuIndex_->pq.M;
    const idx_t nl = cpuIndex_->nlist;
    if (M <= 0 || M > 16 || (d % M) != 0 || (d / M) > 256 || nl <= 0)
        return;

    // Mirror the CPU's precomputed_table_max_bytes default (2 GiB).
    const size_t bytes = (size_t)nl * (size_t)M * 256 * sizeof(float);
    if (bytes > ((size_t)1 << 31))
        return;

    id<MTLDevice> device = resources_->getDevice();
    id<MTLCommandQueue> queue = resources_->getCommandQueue();
    if (!device || !queue)
        return;

    term2Buf_ = [device newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
    if (!term2Buf_)
        return;
    if (!runMetalIVFPQPrecomputeTerm2(
                device,
                queue,
                centroidBuf_,
                gpuIvf_->pqCentroidsBuffer(),
                term2Buf_,
                (int)nl,
                d,
                M)) {
        term2Buf_ = nil;
    }
}

// ============================================================
//  Train / add / reset
// ============================================================

void MetalIndexIVFPQ::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    verifyPQSettings_();
    cpuIndex_->train(n, x);
    if (cpuIndex_->metric_type == METRIC_L2 && cpuIndex_->by_residual) {
        cpuIndex_->precompute_table();
    }
    is_trained = cpuIndex_->is_trained;

    // Mirror coarse and PQ centroids onto the GPU for the lookup-table build.
    uploadCentroids_();
    uploadPQCentroids_();
    precomputeTerm2_();
}

void MetalIndexIVFPQ::encodeResidualAndAppend_(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        const idx_t* xids) {
    if (!gpuIvf_)
        return;
    auto* flatQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    size_t code_size = cpuIndex_->pq.code_size;
    std::vector<uint8_t> encoded(n * code_size);

    if (cpuIndex_->by_residual && flatQ) {
        std::vector<float> residuals((size_t)n * d);
        for (idx_t i = 0; i < n; ++i) {
            idx_t li = list_nos[i];
            if (li < 0) {
                std::memcpy(
                        residuals.data() + i * d,
                        x + i * d,
                        (size_t)d * sizeof(float));
                continue;
            }
            const float* centroid = flatQ->get_xb() + li * d;
            for (int j = 0; j < d; ++j) {
                residuals[i * d + j] = x[i * d + j] - centroid[j];
            }
        }
        cpuIndex_->pq.compute_codes(residuals.data(), encoded.data(), n);
    } else {
        cpuIndex_->pq.compute_codes(x, encoded.data(), n);
    }

    gpuIvf_->appendCodes(n, encoded.data(), list_nos, xids);
}

void MetalIndexIVFPQ::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    if (n == 0)
        return;

    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    idx_t oldNt = cpuIndex_->ntotal;
    cpuIndex_->add_core(n, x, nullptr, list_nos.data());
    ntotal = cpuIndex_->ntotal;

    std::vector<idx_t> ids(n);
    for (idx_t i = 0; i < n; ++i)
        ids[i] = oldNt + i;
    encodeResidualAndAppend_(n, x, list_nos.data(), ids.data());
}

void MetalIndexIVFPQ::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(xids != nullptr);
    if (n == 0)
        return;

    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    cpuIndex_->add_core(n, x, xids, list_nos.data());
    ntotal = cpuIndex_->ntotal;

    encodeResidualAndAppend_(n, x, list_nos.data(), xids);
}

void MetalIndexIVFPQ::reset() {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->reset();
    ntotal = 0;
    if (gpuIvf_)
        gpuIvf_->reset();
}

// ============================================================
//  Search
// ============================================================

void MetalIndexIVFPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(k > 0);

    const float inf = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();
    const bool isL2 = (metric_type == METRIC_L2);

    if (cpuIndex_->ntotal == 0 || n == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i] = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    id<MTLDevice> device = resources_ ? resources_->getDevice() : nil;
    id<MTLCommandQueue> queue =
            resources_ ? resources_->getCommandQueue() : nil;

    // Fall back to CPU when the GPU-resident storage or device is unavailable,
    // when k exceeds what the Metal top-k kernels support, or when the PQ
    // centroids have not been mirrored onto the GPU.
    const int maxK = getMetalDistanceMaxK();
    if (!device || !queue || !gpuIvf_ || !gpuIvf_->codesBuffer() ||
        !gpuIvf_->idsBuffer() || !gpuIvf_->listOffsetGpuBuffer() ||
        !gpuIvf_->listLengthGpuBuffer() || !gpuIvf_->pqCentroidsBuffer() ||
        k > maxK) {
        cpuIndex_->search(n, x, k, distances, labels, params);
        return;
    }

    size_t nprobe = cpuIndex_->nprobe;
    if (auto* ivfParams = dynamic_cast<const IVFSearchParameters*>(params)) {
        if (ivfParams->nprobe > 0)
            nprobe = ivfParams->nprobe;
    }
    nprobe = std::min(nprobe, (size_t)cpuIndex_->nlist);
    if (nprobe == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i] = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    // Coarse quantization on the CPU.
    std::vector<float> coarseDistVec((size_t)n * nprobe);
    std::vector<idx_t> coarseAssignVec((size_t)n * nprobe);
    cpuIndex_->quantizer->search(
            n, x, (idx_t)nprobe, coarseDistVec.data(), coarseAssignVec.data());

    const int M = (int)cpuIndex_->pq.M;
    const int ksub = (int)cpuIndex_->pq.ksub;
    const bool useFp16Lut = config_.useFloat16;

    const size_t outDistBytes = (size_t)n * (size_t)k * sizeof(float);
    const size_t outIdxBytes = (size_t)n * (size_t)k * sizeof(int64_t);
    const size_t queryBytes = (size_t)n * (size_t)d * sizeof(float);
    const size_t perListBytes = (size_t)n * nprobe * (size_t)k * sizeof(float);
    const size_t perListIdxB = (size_t)n * nprobe * (size_t)k * sizeof(int64_t);
    const size_t coarseBytes = (size_t)n * nprobe * sizeof(int32_t);

    ensureSearchBuf_(searchQueriesBuf_, searchQueriesCap_, queryBytes);
    ensureSearchBuf_(searchOutDistBuf_, searchOutDistCap_, outDistBytes);
    ensureSearchBuf_(searchOutIdxBuf_, searchOutIdxCap_, outIdxBytes);
    ensureSearchBuf_(
            searchPerListDistBuf_, searchPerListDistCap_, perListBytes);
    ensureSearchBuf_(searchPerListIdxBuf_, searchPerListIdxCap_, perListIdxB);
    ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseBytes);

    if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
        !searchPerListDistBuf_ || !searchPerListIdxBuf_ || !searchCoarseBuf_) {
        cpuIndex_->search(n, x, k, distances, labels, params);
        return;
    }
    std::memcpy([searchQueriesBuf_ contents], x, queryBytes);

    // Upload coarse assignments as int32.
    auto* coarseDst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
    for (size_t i = 0; i < (size_t)n * nprobe; ++i) {
        FAISS_THROW_IF_NOT_MSG(
                coarseAssignVec[i] >=
                                (idx_t)std::numeric_limits<int32_t>::min() &&
                        coarseAssignVec[i] <=
                                (idx_t)std::numeric_limits<int32_t>::max(),
                "MetalIndexIVFPQ: coarse assignment exceeds int32 range");
        coarseDst[i] = (int32_t)coarseAssignVec[i];
    }

    const int nI = checkedIdxToInt(n, "MetalIndexIVFPQ: n exceeds int range");
    const int kI = checkedIdxToInt(k, "MetalIndexIVFPQ: k exceeds int range");
    const int nprobeI = checkedSizeToInt(
            nprobe, "MetalIndexIVFPQ: nprobe exceeds int range");
    const int avgListLen = cpuIndex_->nlist > 0
            ? (int)(cpuIndex_->ntotal / cpuIndex_->nlist)
            : 0;

    bool ok = false;

    // Preferred path: precomputed-table decomposition. The per-query term is
    // built once per batch (no per-(query, probe) LUT), the scan is exact for
    // any list length, and the merge runs in rounds for any nprobe * k.
    const int dsub = d / M;
    const bool byResidual = cpuIndex_->by_residual;
    const bool canPrecomp = M <= 16 && dsub <= 256 && kI <= 512 &&
            (!isL2 || (byResidual && term2Buf_ != nil)) &&
            gpuIvf_->totalVecs() <= (size_t)std::numeric_limits<int32_t>::max();
    if (canPrecomp) {
        const size_t coarseDistBytes = (size_t)n * nprobe * sizeof(float);
        const size_t qtermBytes =
                (size_t)n * (size_t)M * (size_t)ksub * sizeof(float);
        ensureSearchBuf_(
                searchCoarseDistBuf_, searchCoarseDistCap_, coarseDistBytes);
        ensureSearchBuf_(searchQTermBuf_, searchQTermCap_, qtermBytes);
        if (nprobeI > 1) {
            ensureSearchBuf_(
                    searchMergeDistBuf_, searchMergeDistCap_, perListBytes);
            ensureSearchBuf_(
                    searchMergeIdxBuf_, searchMergeIdxCap_, perListIdxB);
        }
        if (searchCoarseDistBuf_ && searchQTermBuf_ &&
            (nprobeI == 1 || (searchMergeDistBuf_ && searchMergeIdxBuf_))) {
            std::memcpy(
                    [searchCoarseDistBuf_ contents],
                    coarseDistVec.data(),
                    coarseDistBytes);
            ok = runMetalIVFPQPrecompSearch(
                    device,
                    queue,
                    searchQueriesBuf_,
                    searchCoarseBuf_,
                    searchCoarseDistBuf_,
                    isL2 ? term2Buf_ : nil,
                    searchQTermBuf_,
                    gpuIvf_->pqCentroidsBuffer(),
                    gpuIvf_->codesBuffer(),
                    gpuIvf_->idsBuffer(),
                    gpuIvf_->listOffsetGpuBuffer(),
                    gpuIvf_->listLengthGpuBuffer(),
                    nI,
                    d,
                    M,
                    kI,
                    nprobeI,
                    isL2,
                    byResidual,
                    searchOutDistBuf_,
                    searchOutIdxBuf_,
                    searchPerListDistBuf_,
                    searchPerListIdxBuf_,
                    searchMergeDistBuf_,
                    searchMergeIdxBuf_);
        }
    }

    // Legacy path: per-(query, probe) lookup tables. Only allocate the large
    // LUT buffer when this path is actually taken.
    if (!ok) {
        const size_t tableSize = (size_t)n * nprobe * M * ksub;
        const size_t tableBytes = tableSize * sizeof(float);
        const size_t tableHalfBytes = tableSize * sizeof(uint16_t);
        ensureSearchBuf_(lookupTableBuf_, lookupTableCap_, tableBytes);
        if (useFp16Lut) {
            ensureSearchBuf_(
                    lookupTableHalfBuf_, lookupTableHalfCap_, tableHalfBytes);
        }
        if (lookupTableBuf_ && (!useFp16Lut || lookupTableHalfBuf_)) {
            ok = runMetalIVFPQFullSearch(
                    device,
                    queue,
                    searchQueriesBuf_,
                    searchCoarseBuf_,
                    centroidBuf_,
                    gpuIvf_->pqCentroidsBuffer(),
                    useFp16Lut ? lookupTableHalfBuf_ : lookupTableBuf_,
                    gpuIvf_->codesBuffer(),
                    gpuIvf_->idsBuffer(),
                    gpuIvf_->listOffsetGpuBuffer(),
                    gpuIvf_->listLengthGpuBuffer(),
                    nI,
                    d,
                    M,
                    kI,
                    nprobeI,
                    (int)cpuIndex_->nlist,
                    avgListLen,
                    useFp16Lut,
                    isL2,
                    searchOutDistBuf_,
                    searchOutIdxBuf_,
                    searchPerListDistBuf_,
                    searchPerListIdxBuf_);
        }
    }

    if (!ok) {
        cpuIndex_->search(n, x, k, distances, labels, params);
        return;
    }

    const float* outDistPtr =
            reinterpret_cast<const float*>([searchOutDistBuf_ contents]);
    const int64_t* outIdxPtr =
            reinterpret_cast<const int64_t*>([searchOutIdxBuf_ contents]);
    for (idx_t qi = 0; qi < n; ++qi) {
        for (idx_t j = 0; j < k; ++j) {
            const size_t pos = (size_t)qi * (size_t)k + (size_t)j;
            const int64_t globalId = outIdxPtr[pos];
            labels[pos] = (globalId < 0) ? -1 : (idx_t)globalId;
            distances[pos] = outDistPtr[pos];
        }
    }
}

// ============================================================
//  Copy from / to
// ============================================================

void MetalIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* src) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(src);
    FAISS_THROW_IF_NOT_FMT(
            src->pq.nbits == 8,
            "copyFrom: only 8-bit PQ codes are supported (got %d)",
            (int)src->pq.nbits);
    FAISS_THROW_IF_NOT_MSG(
            src->by_residual, "copyFrom: only by_residual = true is supported");
    FAISS_THROW_IF_NOT_MSG(
            src->polysemous_ht == 0,
            "copyFrom: polysemous codes are not supported");
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
    FAISS_THROW_IF_NOT_MSG(
            src->quantizer, "copyFrom: source quantizer is null");
    auto* ourQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    FAISS_THROW_IF_NOT_MSG(ourQ, "copyFrom: internal quantizer not IndexFlat");
    ourQ->reset();
    if (src->nlist > 0) {
        std::vector<float> coarse((size_t)src->nlist * d);
        src->quantizer->reconstruct_n(0, src->nlist, coarse.data());
        ourQ->add(src->nlist, coarse.data());
    }

    // Copy PQ centroids.
    cpuIndex_->pq = src->pq;
    cpuIndex_->metric_type = src->metric_type;
    cpuIndex_->metric_arg = src->metric_arg;
    cpuIndex_->is_trained = true;
    cpuIndex_->nprobe = src->nprobe;
    cpuIndex_->by_residual = src->by_residual;
    cpuIndex_->use_precomputed_table = src->use_precomputed_table;
    if (cpuIndex_->metric_type == METRIC_L2 && cpuIndex_->by_residual) {
        cpuIndex_->precompute_table();
    }
    is_trained = true;

    // Mirror coarse and PQ centroids onto the GPU for the lookup-table build.
    uploadCentroids_();
    uploadPQCentroids_();
    precomputeTerm2_();

    // Gather IVF list data.
    size_t totalN = 0;
    for (size_t l = 0; l < (size_t)src->nlist; ++l)
        totalN += src->invlists->list_size(l);

    if (totalN > 0) {
        size_t code_size = src->pq.code_size;
        std::vector<uint8_t> allCodes(totalN * code_size);
        std::vector<idx_t> allListNos(totalN);
        std::vector<idx_t> allIds(totalN);
        size_t pos = 0;

        for (size_t l = 0; l < (size_t)src->nlist; ++l) {
            size_t ls = src->invlists->list_size(l);
            if (ls == 0)
                continue;
            const uint8_t* codes = src->invlists->get_codes(l);
            const idx_t* ids = src->invlists->get_ids(l);

            cpuIndex_->invlists->add_entries(l, ls, ids, codes);

            std::memcpy(
                    allCodes.data() + pos * code_size, codes, ls * code_size);
            std::memcpy(allIds.data() + pos, ids, ls * sizeof(idx_t));
            for (size_t i = 0; i < ls; ++i)
                allListNos[pos + i] = (idx_t)l;
            pos += ls;
        }

        cpuIndex_->ntotal = (idx_t)totalN;
        ntotal = (idx_t)totalN;

        if (gpuIvf_)
            gpuIvf_->appendCodes(
                    (idx_t)totalN,
                    allCodes.data(),
                    allListNos.data(),
                    allIds.data());
    }
}

void MetalIndexIVFPQ::copyTo(faiss::IndexIVFPQ* dst) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(dst);

    auto* srcQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    auto* dstQ = dynamic_cast<faiss::IndexFlat*>(dst->quantizer);
    FAISS_THROW_IF_NOT_MSG(srcQ, "copyTo: internal quantizer not IndexFlat");
    FAISS_THROW_IF_NOT_MSG(dstQ, "copyTo: destination quantizer not IndexFlat");

    dstQ->reset();
    if (srcQ->ntotal > 0)
        dstQ->add(srcQ->ntotal, srcQ->get_xb());

    dst->pq = cpuIndex_->pq;
    dst->metric_type = cpuIndex_->metric_type;
    dst->metric_arg = cpuIndex_->metric_arg;
    dst->d = cpuIndex_->d;
    dst->nlist = cpuIndex_->nlist;
    dst->nprobe = cpuIndex_->nprobe;
    dst->is_trained = cpuIndex_->is_trained;
    dst->by_residual = cpuIndex_->by_residual;
    dst->use_precomputed_table = cpuIndex_->use_precomputed_table;
    if (dst->use_precomputed_table && dst->metric_type == METRIC_L2 &&
        dst->by_residual) {
        dst->precompute_table();
    }

    for (size_t l = 0; l < (size_t)cpuIndex_->nlist; ++l) {
        size_t ls = cpuIndex_->invlists->list_size(l);
        if (ls == 0)
            continue;
        const uint8_t* codes = cpuIndex_->invlists->get_codes(l);
        const idx_t* ids = cpuIndex_->invlists->get_ids(l);
        dst->invlists->add_entries(l, ls, ids, codes);
    }
    dst->ntotal = cpuIndex_->ntotal;
}

// ============================================================
//  Accessors
// ============================================================

void MetalIndexIVFPQ::updateQuantizer() {}

std::vector<idx_t> MetalIndexIVFPQ::getListIndices(idx_t listId) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(listId >= 0 && listId < cpuIndex_->nlist);
    size_t ls = cpuIndex_->invlists->list_size(listId);
    if (ls == 0)
        return {};
    const idx_t* ids = cpuIndex_->invlists->get_ids(listId);
    return std::vector<idx_t>(ids, ids + ls);
}

void MetalIndexIVFPQ::reclaimMemory() {
    // No-op: Metal unified memory doesn't require explicit reclaim.
}

void MetalIndexIVFPQ::reserveMemory(idx_t numVecs) {
    if (gpuIvf_) {
        gpuIvf_->reserveMemory(numVecs);
    }
}

idx_t MetalIndexIVFPQ::nlist() const {
    return cpuIndex_ ? cpuIndex_->nlist : 0;
}

size_t MetalIndexIVFPQ::nprobe() const {
    return cpuIndex_ ? cpuIndex_->nprobe : 1;
}

int MetalIndexIVFPQ::getNumSubQuantizers() const {
    return cpuIndex_ ? (int)cpuIndex_->pq.M : 0;
}

void MetalIndexIVFPQ::setUsePrecomputedTables(bool enable) {
    if (!cpuIndex_) {
        return;
    }
    cpuIndex_->use_precomputed_table = enable ? 1 : 0;
    if (enable && cpuIndex_->metric_type == METRIC_L2 &&
        cpuIndex_->by_residual) {
        cpuIndex_->precompute_table();
    } else if (!enable) {
        cpuIndex_->precomputed_table.clear();
    }
}

bool MetalIndexIVFPQ::getUsePrecomputedTables() const {
    return cpuIndex_ && cpuIndex_->use_precomputed_table != 0;
}

} // namespace gpu_metal
} // namespace faiss
