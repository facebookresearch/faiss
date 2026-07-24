// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal IVF-PQ index: 8-bit product quantization with precomputed
 * per-query lookup tables and GPU IVF list scanning.
 */

#pragma once

#import <Metal/Metal.h>

#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu_metal/MetalIndex.h>

#include <memory>

namespace faiss {
namespace gpu_metal {

class MetalIVFPQImpl;

class MetalIndexIVFPQ : public MetalIndex {
   public:
    MetalIndexIVFPQ(
            std::shared_ptr<MetalResources> resources,
            int dims,
            idx_t nlist,
            int M,
            int nbitsPerIdx,
            faiss::MetricType metric,
            float metricArg = 0.0f,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexIVFPQ(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexIVFPQ* cpuIndex,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndexIVFPQ() override;

    void train(idx_t n, const float* x) override;
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

    void copyFrom(const faiss::IndexIVFPQ* index);
    void copyTo(faiss::IndexIVFPQ* index) const;

    void updateQuantizer();
    std::vector<idx_t> getListIndices(idx_t listId) const;
    void reclaimMemory();

    /// Pre-allocate GPU storage for the given total number of vectors.
    void reserveMemory(idx_t numVecs);

    idx_t nlist() const;
    size_t nprobe() const;
    int getNumSubQuantizers() const;
    void setUsePrecomputedTables(bool enable);
    bool getUsePrecomputedTables() const;

   private:
    std::unique_ptr<faiss::IndexIVFPQ> cpuIndex_;
    std::unique_ptr<MetalIVFPQImpl> gpuIvf_;

    void verifyPQSettings_() const;

    void encodeResidualAndAppend_(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            const idx_t* xids);
};

} // namespace gpu_metal
} // namespace faiss
