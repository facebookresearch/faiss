// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/clone_index.h>
#include <faiss/gpu_metal/MetalIVFSQScan.h>
#include <faiss/gpu_metal/MetalIndexIVFScalarQuantizer.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/InvertedLists.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <typeinfo>
#include <vector>

namespace faiss::gpu_metal {
namespace {
size_t bytes(size_t a, size_t b) {
    FAISS_THROW_IF_NOT_MSG(
            !a || b <= SIZE_MAX / a, "Metal IVFSQ size overflow");
    return a * b;
}
id<MTLBuffer> allocate(id<MTLDevice> device, size_t size)
        __attribute__((ns_returns_retained));
id<MTLBuffer> allocate(id<MTLDevice> device, size_t size) {
    size = std::max(size, size_t(8));
    FAISS_THROW_IF_NOT_MSG(
            size <= device.maxBufferLength,
            "Metal IVFSQ buffer exceeds maxBufferLength");
    id<MTLBuffer> buffer =
            [device newBufferWithLength:size
                                options:MTLResourceStorageModeShared];
    FAISS_THROW_IF_NOT_MSG(buffer, "Metal IVFSQ buffer allocation failed");
    return buffer;
}
// The generic CPU cloner can preserve borrowed MaybeOwnedVector views. Walk
// every supported coarse-quantizer node, including HNSW's graph and leaf.
// Reject unknown graph structures rather than silently retaining dependencies.
void makeQuantizerOwning(Index* index) {
    if (auto* flat = dynamic_cast<IndexFlatCodes*>(index)) {
        MaybeOwnedVector<uint8_t> owned(flat->codes.size());
        if (flat->codes.size())
            memcpy(owned.data(), flat->codes.data(), flat->codes.size());
        flat->codes = std::move(owned);
    } else if (auto* transform = dynamic_cast<IndexPreTransform*>(index)) {
        makeQuantizerOwning(transform->index);
    } else if (auto* mapped = dynamic_cast<IndexIDMap*>(index)) {
        makeQuantizerOwning(mapped->index);
    } else if (auto* hnsw = dynamic_cast<IndexHNSW*>(index)) {
        auto& neighbors = hnsw->hnsw.neighbors;
        MaybeOwnedVector<IndexHNSW::storage_idx_t> owned(neighbors.size());
        if (neighbors.size())
            memcpy(owned.data(),
                   neighbors.data(),
                   bytes(neighbors.size(), sizeof(IndexHNSW::storage_idx_t)));
        neighbors = std::move(owned);
        makeQuantizerOwning(hnsw->storage);
    } else {
        FAISS_THROW_MSG(
                "Metal IVFSQ coarse quantizers require FlatCodes or HNSW storage, optionally wrapped in IndexPreTransform/IndexIDMap");
    }
}
std::unique_ptr<IndexIVFScalarQuantizer> clone(
        const IndexIVFScalarQuantizer* index) {
    FAISS_THROW_IF_NOT(index);
    auto result = std::make_unique<IndexIVFScalarQuantizer>();
    // Copy configuration without ever making the shallow CPU pointers owned.
    static_cast<Index&>(*result) = static_cast<const Index&>(*index);
    try {
        static_cast<IndexIVFInterface&>(*result) =
                static_cast<const IndexIVFInterface&>(*index);
    } catch (...) {
        result->own_fields = false;
        throw;
    }
    result->own_fields = false;
    result->code_size = index->code_size;
    result->parallel_mode = index->parallel_mode;
    result->direct_map = index->direct_map;
    result->by_residual = index->by_residual;
    result->sq = index->sq;
    std::unique_ptr<Index> quantizer(clone_index(index->quantizer));
    makeQuantizerOwning(quantizer.get());
    auto lists = std::make_unique<ArrayInvertedLists>(
            index->nlist, index->code_size);
    if (index->invlists) {
        for (size_t l = 0; l < index->nlist; ++l) {
            size_t count = index->invlists->list_size(l);
            if (!count)
                continue;
            InvertedLists::ScopedCodes codes(index->invlists, l);
            InvertedLists::ScopedIds ids(index->invlists, l);
            lists->add_entries(l, count, ids.get(), codes.get());
        }
    }
    result->quantizer = quantizer.release();
    result->own_fields = true;
    result->invlists = lists.release();
    result->own_invlists = true;
    return result;
}
} // namespace

struct MetalIndexIVFScalarQuantizer::Impl {
    std::unique_ptr<IndexIVFScalarQuantizer> cpu;
    id<MTLDevice> device;
    id<MTLBuffer> codes, ids, offsets, lengths, centroids, ranges;
    std::vector<uint32_t> offset, length, capacity;
    size_t tail = 0, allocated = 0;
    bool valid = true, nonnegative = true;
    mutable id<MTLBuffer> queries, assignments, distances, labels, coarseScores;
    mutable size_t dispatches = 0;

    Impl(id<MTLDevice> dev, std::unique_ptr<IndexIVFScalarQuantizer> index)
            : cpu(std::move(index)), device(dev) {
        offset.resize(cpu->nlist);
        length.resize(cpu->nlist);
        capacity.resize(cpu->nlist);
    }

    bool codecSupported() const {
        return cpu->sq.qtype == ScalarQuantizer::QT_8bit ||
                cpu->sq.qtype == ScalarQuantizer::QT_fp16;
    }
    bool representable() const {
        if (cpu->d <= 0 || cpu->nlist > INT32_MAX || cpu->ntotal < 0 ||
            uint64_t(cpu->ntotal) > UINT32_MAX)
            return false;
        size_t limit = device.maxBufferLength;
        if (cpu->nlist > limit / sizeof(uint32_t) ||
            uint64_t(cpu->ntotal) >
                    limit / std::max(cpu->code_size, sizeof(idx_t)) ||
            cpu->sq.trained.size() > limit / sizeof(float) ||
            (cpu->by_residual &&
             cpu->nlist > limit / sizeof(float) / size_t(cpu->d)))
            return false;
        for (size_t l = 0; l < cpu->nlist; ++l)
            if (cpu->invlists->list_size(l) > UINT32_MAX)
                return false;
        return true;
    }

    // Append only newly encoded list tails. Relocated lists receive geometric
    // capacity; replaced buffers and metadata are published together. Existing
    // bytes are untouched until all allocations/access guards have succeeded.
    void upload() {
        if (!codecSupported() || !representable())
            return;
        auto newOffset = offset, newLength = length, newCapacity = capacity;
        size_t newTail = tail;
        struct Pending {
            size_t list, old, count;
            std::vector<uint8_t> code;
            std::vector<idx_t> id;
        };
        std::vector<Pending> pending;
        bool positive = nonnegative;
        for (size_t l = 0; l < cpu->nlist; ++l) {
            size_t count = cpu->invlists->list_size(l);
            FAISS_THROW_IF_NOT(count >= length[l]);
            if (count == length[l])
                continue;
            bool moved = count > capacity[l];
            if (moved) {
                size_t cap = std::max(
                        count,
                        std::max(size_t(16), bytes(size_t(capacity[l]), 2)));
                FAISS_THROW_IF_NOT(
                        cap <= UINT32_MAX && newTail <= UINT32_MAX - cap);
                newOffset[l] = uint32_t(newTail);
                newCapacity[l] = uint32_t(cap);
                newTail += cap;
            }
            size_t old = moved ? 0 : length[l];
            Pending part{l, old, count - old, {}, {}};
            InvertedLists::ScopedCodes code(cpu->invlists, l);
            InvertedLists::ScopedIds id(cpu->invlists, l);
            part.code.assign(
                    code.get() + bytes(old, cpu->code_size),
                    code.get() + bytes(count, cpu->code_size));
            part.id.assign(id.get() + old, id.get() + count);
            for (auto label : part.id)
                positive &= label >= 0;
            pending.push_back(std::move(part));
            newLength[l] = uint32_t(count);
        }
        size_t newAllocated = allocated;
        id<MTLBuffer> newCodes = codes, newIds = ids;
        if (!codes || newTail > allocated) {
            newAllocated = std::max(
                    newTail, std::max(size_t(16), bytes(allocated, 2)));
            newCodes = allocate(device, bytes(newAllocated, cpu->code_size));
            newIds = allocate(device, bytes(newAllocated, sizeof(idx_t)));
            if (codes) {
                memcpy(newCodes.contents,
                       codes.contents,
                       bytes(tail, cpu->code_size));
                memcpy(newIds.contents,
                       ids.contents,
                       bytes(tail, sizeof(idx_t)));
            }
        }
        auto newOffsets = allocate(device, bytes(cpu->nlist, sizeof(uint32_t)));
        auto newLengths = allocate(device, bytes(cpu->nlist, sizeof(uint32_t)));
        // Codec parameters and reconstructed coarse centroids are immutable
        // between training/replacement operations. Appends only change lists.
        id<MTLBuffer> newRanges = ranges, newCentroids = centroids;
        if (!newRanges) {
            newRanges = allocate(
                    device, bytes(cpu->sq.trained.size(), sizeof(float)));
            if (!cpu->sq.trained.empty())
                memcpy(newRanges.contents,
                       cpu->sq.trained.data(),
                       bytes(cpu->sq.trained.size(), sizeof(float)));
        }
        if (!newCentroids) {
            size_t centroidCount = cpu->is_trained && cpu->by_residual
                    ? bytes(cpu->nlist, size_t(cpu->d))
                    : 0;
            newCentroids =
                    allocate(device, bytes(centroidCount, sizeof(float)));
            if (centroidCount)
                cpu->quantizer->reconstruct_n(
                        0, cpu->nlist, (float*)newCentroids.contents);
        }
        memcpy(newOffsets.contents, newOffset.data(), bytes(cpu->nlist, 4));
        memcpy(newLengths.contents, newLength.data(), bytes(cpu->nlist, 4));
        for (const auto& part : pending) {
            size_t start = size_t(newOffset[part.list]) + part.old;
            memcpy((uint8_t*)newCodes.contents + bytes(start, cpu->code_size),
                   part.code.data(),
                   part.code.size());
            memcpy((idx_t*)newIds.contents + start,
                   part.id.data(),
                   bytes(part.count, sizeof(idx_t)));
        }
        codes = newCodes;
        ids = newIds;
        offsets = newOffsets;
        lengths = newLengths;
        ranges = newRanges;
        centroids = newCentroids;
        offset.swap(newOffset);
        length.swap(newLength);
        capacity.swap(newCapacity);
        tail = newTail;
        allocated = newAllocated;
        nonnegative = positive;
    }
    void scratch(size_t nq, size_t probes, size_t k) const {
        // Allocate all replacements before changing any scratch field.
        auto grow = [&](id<MTLBuffer> old, size_t needed) {
            if (old && old.length >= needed)
                return old;
            size_t grown = old ? std::min(
                                         size_t(device.maxBufferLength),
                                         bytes(size_t(old.length), 2))
                               : 1;
            return allocate(device, std::max(needed, grown));
        };
        auto q = grow(queries, bytes(bytes(nq, cpu->d), 4));
        auto a = grow(assignments, bytes(bytes(nq, probes), 4));
        auto d = grow(distances, bytes(bytes(nq, k), 4));
        auto i = grow(labels, bytes(bytes(nq, k), 8));
        auto scores = grow(coarseScores, bytes(bytes(nq, probes), 4));
        coarseScores = scores;
        queries = q;
        assignments = a;
        distances = d;
        labels = i;
    }
};

MetalIndexIVFScalarQuantizer::MetalIndexIVFScalarQuantizer(
        std::shared_ptr<MetalResources> resources,
        const IndexIVFScalarQuantizer* index,
        MetalIndexConfig config)
        : MetalIndex(
                  resources,
                  index ? index->d : 0,
                  index ? index->metric_type : METRIC_L2,
                  index ? index->metric_arg : 0,
                  config) {
    copyFrom(index);
}
MetalIndexIVFScalarQuantizer::MetalIndexIVFScalarQuantizer(
        std::shared_ptr<MetalResources> resources,
        int dims,
        size_t lists,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool residual,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, 0, config) {
    auto quantizer = std::make_unique<IndexFlat>(dims, metric);
    auto cpu = std::make_unique<IndexIVFScalarQuantizer>(
            quantizer.get(), dims, lists, qtype, metric, residual);
    cpu->own_fields = true;
    quantizer.release();
    impl_ = std::make_unique<Impl>(resources_->getDevice(), std::move(cpu));
    syncState_();
}
MetalIndexIVFScalarQuantizer::~MetalIndexIVFScalarQuantizer() = default;
void MetalIndexIVFScalarQuantizer::syncState_() {
    d = impl_->cpu->d;
    ntotal = impl_->cpu->ntotal;
    metric_type = impl_->cpu->metric_type;
    metric_arg = impl_->cpu->metric_arg;
    is_trained = impl_->cpu->is_trained;
    verbose = impl_->cpu->verbose;
}
void MetalIndexIVFScalarQuantizer::copyFrom(
        const IndexIVFScalarQuantizer* index) {
    auto next = std::make_unique<Impl>(resources_->getDevice(), clone(index));
    if (next->cpu->is_trained)
        next->upload();
    impl_.swap(next);
    syncState_();
}
IndexIVFScalarQuantizer* MetalIndexIVFScalarQuantizer::copyTo() const {
    return clone(impl_->cpu.get()).release();
}
void MetalIndexIVFScalarQuantizer::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(impl_->valid);
    impl_->valid = false;
    try {
        impl_->cpu->train(n, x);
        impl_->ranges = nil;
        impl_->centroids = nil;
        impl_->upload();
        impl_->valid = true;
    } catch (...) {
        syncState_();
        throw;
    }
    syncState_();
}
void MetalIndexIVFScalarQuantizer::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}
void MetalIndexIVFScalarQuantizer::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* ids) {
    FAISS_THROW_IF_NOT_MSG(
            impl_->valid,
            "Metal IVFSQ invalid after failed mutation; reset or copyFrom required");
    impl_->valid = false;
    try {
        impl_->cpu->add_with_ids(n, x, ids);
        impl_->upload();
        impl_->valid = true;
    } catch (...) {
        syncState_();
        throw;
    }
    syncState_();
}
void MetalIndexIVFScalarQuantizer::reset() {
    auto cpu = clone(impl_->cpu.get());
    cpu->reset();
    auto next = std::make_unique<Impl>(resources_->getDevice(), std::move(cpu));
    impl_.swap(next);
    syncState_();
}
void MetalIndexIVFScalarQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* outD,
        idx_t* outI,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            impl_->valid,
            "Metal IVFSQ invalid after failed mutation; reset or copyFrom required");
    auto& cpu = *impl_->cpu;
    const auto* ivf = dynamic_cast<const SearchParametersIVF*>(params);
    size_t probes = ivf ? ivf->nprobe : cpu.nprobe;
    bool control = params &&
            (!ivf || typeid(*params) != typeid(SearchParametersIVF) ||
             ivf->sel || ivf->max_codes || ivf->max_lists_num ||
             ivf->ensure_topk_full || ivf->max_empty_result_buckets ||
             ivf->quantizer_params || ivf->inverted_list_context);
    probes = std::min(probes, cpu.nlist);
    bool gpu = !control && impl_->codecSupported() && impl_->representable() &&
            impl_->nonnegative &&
            (metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT) &&
            is_trained && n > 0 && uint64_t(n) <= UINT32_MAX && k > 0 &&
            k <= 512 && probes && probes <= UINT32_MAX && !cpu.max_codes &&
            !(cpu.parallel_mode & cpu.PARALLEL_MODE_NO_HEAP_INIT);
    // These checked products also define the allocation envelope before any
    // dispatch.
    auto fits = [&](size_t a, size_t b, size_t width) {
        return !a ||
                (b <= SIZE_MAX / a &&
                 a * b <= size_t(resources_->getDevice().maxBufferLength) /
                                 width);
    };
    if (gpu)
        gpu = fits(n, d, 4) && fits(n, probes, 4) && fits(n, k, 8);
    if (!gpu) {
        cpu.search(n, x, k, outD, outI, params);
        return;
    }
    FAISS_THROW_IF_NOT(x && outD && outI);
    if (!impl_->offsets)
        impl_->upload();
    impl_->scratch(n, probes, k);
    std::vector<idx_t> coarse(bytes(size_t(n), probes));
    std::vector<float> coarseD(coarse.size());
    cpu.quantizer->search(n, x, probes, coarseD.data(), coarse.data());
    memcpy(impl_->coarseScores.contents,
           coarseD.data(),
           bytes(coarseD.size(), 4));
    auto* assignment = (int32_t*)impl_->assignments.contents;
    for (size_t i = 0; i < coarse.size(); ++i) {
        FAISS_THROW_IF_NOT(
                coarse[i] >= -1 &&
                (coarse[i] < 0 || uint64_t(coarse[i]) < cpu.nlist));
        assignment[i] = int32_t(coarse[i]);
    }
    memcpy(impl_->queries.contents, x, bytes(bytes(size_t(n), size_t(d)), 4));
    uint32_t p[] = {
            uint32_t(n),
            uint32_t(d),
            uint32_t(k),
            uint32_t(probes),
            uint32_t(cpu.nlist),
            cpu.sq.qtype == ScalarQuantizer::QT_fp16,
            cpu.by_residual,
            metric_type == METRIC_L2,
            uint32_t(cpu.code_size)};
    runMetalIVFSQScan(
            resources_->getDevice(),
            resources_->getCommandQueue(),
            impl_->queries,
            impl_->codes,
            impl_->ids,
            impl_->offsets,
            impl_->lengths,
            impl_->assignments,
            impl_->centroids,
            impl_->ranges,
            impl_->distances,
            impl_->labels,
            impl_->coarseScores,
            p);
    ++impl_->dispatches;
    memcpy(outD,
           impl_->distances.contents,
           bytes(bytes(size_t(n), size_t(k)), 4));
    memcpy(outI, impl_->labels.contents, bytes(bytes(size_t(n), size_t(k)), 8));
}
void MetalIndexIVFScalarQuantizer::reconstruct(idx_t key, float* out) const {
    impl_->cpu->reconstruct(key, out);
}
void MetalIndexIVFScalarQuantizer::reconstruct_n(
        idx_t first,
        idx_t n,
        float* out) const {
    impl_->cpu->reconstruct_n(first, n, out);
}
void MetalIndexIVFScalarQuantizer::reserveMemory(idx_t count) {
    FAISS_THROW_IF_NOT(count >= 0 && uint64_t(count) <= UINT32_MAX);
    FAISS_THROW_IF_NOT(impl_->valid);
    if (!impl_->codecSupported() || size_t(count) <= impl_->allocated)
        return;
    auto codes = allocate(impl_->device, bytes(count, impl_->cpu->code_size));
    auto ids = allocate(impl_->device, bytes(count, sizeof(idx_t)));
    if (impl_->tail) {
        memcpy(codes.contents,
               impl_->codes.contents,
               bytes(impl_->tail, impl_->cpu->code_size));
        memcpy(ids.contents,
               impl_->ids.contents,
               bytes(impl_->tail, sizeof(idx_t)));
    }
    impl_->codes = codes;
    impl_->ids = ids;
    impl_->allocated = count;
}
void MetalIndexIVFScalarQuantizer::reclaimMemory() {
    FAISS_THROW_IF_NOT(impl_->valid);
    copyFrom(impl_->cpu.get());
}
size_t MetalIndexIVFScalarQuantizer::nlist() const {
    return impl_->cpu->nlist;
}
size_t MetalIndexIVFScalarQuantizer::nprobe() const {
    return impl_->cpu->nprobe;
}
void MetalIndexIVFScalarQuantizer::setNumProbes(size_t probes) {
    impl_->cpu->nprobe = probes;
}
size_t MetalIndexIVFScalarQuantizer::gpuSearchCount() const {
    return impl_->dispatches;
}
} // namespace faiss::gpu_metal
