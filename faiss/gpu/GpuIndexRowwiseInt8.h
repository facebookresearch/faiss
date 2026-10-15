/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndex.h>
#include <memory>

namespace faiss {
struct IndexRowwiseInt8;
}

namespace faiss {
namespace gpu {

struct RowwiseInt8Data;

struct GpuIndexRowwiseInt8Config : public GpuIndexConfig {
    /// queries per GEMM tile
    int tileQueries = 4096;

    /// bank vectors per GEMM tile. Scratch memory per tile is
    /// tileQueries * tileBank * 4 bytes (int32 dot products, turned into
    /// float distances in place)
    int tileBank = 32768;
};

/** GPU version of IndexRowwiseInt8: the bank is kept on the GPU as int8
 * codes + one float scale per vector. Queries are row-wise int8 encoded on
 * the GPU, dot products are exact int8 x int8 -> int32 GEMMs (cuBLAS), and
 * the k nearest are chosen with the FAISS k-selection kernels.
 */
class GpuIndexRowwiseInt8 : public GpuIndex {
   public:
    GpuIndexRowwiseInt8(
            GpuResourcesProvider* provider,
            int dims,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexRowwiseInt8Config config = GpuIndexRowwiseInt8Config());

    GpuIndexRowwiseInt8(
            GpuResourcesProvider* provider,
            const faiss::IndexRowwiseInt8* index,
            GpuIndexRowwiseInt8Config config = GpuIndexRowwiseInt8Config());

    ~GpuIndexRowwiseInt8() override;

    /// Replace our contents with those of the CPU index
    void copyFrom(const faiss::IndexRowwiseInt8* index);

    /// Copy our contents to the CPU index
    void copyTo(faiss::IndexRowwiseInt8* index) const;

    void reset() override;

    /// add vectors that are already row-wise int8: codes (n, d), scales (n),
    /// on the host or on this GPU
    void add_int8(idx_t n, const int8_t* codes, const float* scales);

    /// copy out the codes (n, d) and scales (n) of vectors i0 .. i0 + n - 1
    /// to host memory
    void get_int8(idx_t i0, idx_t n, int8_t* codes, float* scales) const;

    /// search with queries that are already row-wise int8: codes (n, d),
    /// scales (n); inputs and outputs on the host or on this GPU
    void search_int8(
            idx_t n,
            const int8_t* codes,
            const float* scales,
            int k,
            float* distances,
            idx_t* labels) const;

    void reconstruct_n(idx_t i0, idx_t n, float* out) const override;

    void reconstruct(idx_t key, float* out) const override;

    /// GPU memory held by the bank (codes, scales, norms), in bytes
    size_t getBankBytes() const;

   protected:
    bool addImplRequiresIDs_() const override;

    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

    void addCodes_(idx_t n, const int8_t* codesDev, const float* scalesDev);

    /// search core, on device data: queries are either float (x) or
    /// row-wise int8 (qc, qs)
    void searchTiles_(
            idx_t n,
            const float* x,
            const int8_t* qc,
            const float* qs,
            int k,
            float* distances,
            idx_t* labels) const;

    const GpuIndexRowwiseInt8Config rwConfig_;

    std::unique_ptr<RowwiseInt8Data> data_;
};

} // namespace gpu
} // namespace faiss
