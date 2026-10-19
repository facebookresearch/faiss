// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu_metal/MetalIndex.h>
#include <memory>

namespace faiss::gpu_metal {

/** GPU-resident IVF scalar codes with CPU training, encoding and assignment.
 *
 * GPU search supports QT_8bit and QT_fp16, L2/IP, either residual mode,
 * 1 <= k <= 512, positive dimensions and counts fitting uint32_t (nlist <=
 * INT32_MAX). Buffer sizes must fit device.maxBufferLength. Labels must be
 * nonnegative. Other valid requests use CPU search with the original
 * parameters, including selectors, max_codes, quantizer_params, and derived
 * parameter types, and PARALLEL_MODE_NO_HEAP_INIT. Coarse quantizers must use
 * FlatCodes or HNSW storage, optionally wrapped in PreTransform/IDMap; cloning
 * rejects other structures rather than retaining borrowed storage.
 * Selected GPU execution failures throw; a failed mutation
 * invalidates GPU state until reset/copyFrom. Search and mutation must be
 * serialized by the caller.
 */
class MetalIndexIVFScalarQuantizer : public MetalIndex {
   public:
    MetalIndexIVFScalarQuantizer(
            std::shared_ptr<MetalResources> resources,
            const IndexIVFScalarQuantizer* index,
            MetalIndexConfig config = MetalIndexConfig());
    MetalIndexIVFScalarQuantizer(
            std::shared_ptr<MetalResources> resources,
            int dims,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool byResidual = true,
            MetalIndexConfig config = MetalIndexConfig());
    ~MetalIndexIVFScalarQuantizer() override;

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* ids) override;
    void reset() override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    void reconstruct(idx_t key, float* out) const override;
    void reconstruct_n(idx_t first, idx_t count, float* out) const override;
    void copyFrom(const IndexIVFScalarQuantizer* index);
    /// Return a complete owning CPU copy (caller owns the result).
    IndexIVFScalarQuantizer* copyTo() const;
    /// Reserve total code/ID capacity; list segments are assigned during add.
    void reserveMemory(idx_t numVectors);
    /// Compact list storage and release reusable search scratch.
    void reclaimMemory();
    size_t nlist() const;
    size_t nprobe() const;
    void setNumProbes(size_t nprobe);
    /// Successful GPU dispatch count, for tests and benchmark visibility.
    size_t gpuSearchCount() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void syncState_();
};
} // namespace faiss::gpu_metal
