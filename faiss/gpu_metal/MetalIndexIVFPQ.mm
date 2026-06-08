// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexIVFPQ.h"

#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/impl/MetalIVFPQ.h>
#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace faiss {
namespace gpu_metal {

// ============================================================
//  Constructors
// ============================================================

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
//  Train / add / reset
// ============================================================

void MetalIndexIVFPQ::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->train(n, x);
    if (cpuIndex_->metric_type == METRIC_L2 && cpuIndex_->by_residual) {
        cpuIndex_->precompute_table();
    }
    is_trained = cpuIndex_->is_trained;
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

    if (auto* ivfParams = dynamic_cast<const IVFSearchParameters*>(params)) {
        if (ivfParams->nprobe > 0)
            cpuIndex_->nprobe = ivfParams->nprobe;
    }

    cpuIndex_->search(n, x, k, distances, labels);
}

// ============================================================
//  Copy from / to
// ============================================================

void MetalIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* src) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(src);
    FAISS_THROW_IF_NOT(src->pq.nbits == 8);
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
