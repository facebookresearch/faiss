/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/impl/ProductQuantizer.h>
#include <memory>
#include <vector>

namespace faiss {
struct IndexIVFPQ;
}

namespace faiss {
namespace gpu {

class GpuIndexFlat;
class IVFPQ;

struct GpuIndexIVFPQConfig : public GpuIndexIVFConfig {
    inline GpuIndexIVFPQConfig()
            : useFloat16LookupTables(false),
              usePrecomputedTables(false),
              interleavedLayout(false),
              useMMCodeDistance(false) {}

    /// Whether or not float16 residual distance tables are used in the
    /// list scanning kernels. When subQuantizers * 2^bitsPerCode >
    /// 16384, this is required.
    bool useFloat16LookupTables;

    /// Whether or not we enable the precomputed table option for
    /// search, which can substantially increase the memory requirement.
    bool usePrecomputedTables;

    /// Use the alternative memory layout for the IVF lists
    /// WARNING: this is a feature under development, do not use!
    bool interleavedLayout;

    /// Use GEMM-backed computation of PQ code distances for the no precomputed
    /// table version of IVFPQ.
    /// This is for debugging purposes, it should not substantially affect the
    /// results one way for another.
    ///
    /// Note that MM code distance is enabled automatically if one uses a number
    /// of dimensions per sub-quantizer that is not natively specialized (an odd
    /// number like 7 or so).
    bool useMMCodeDistance;
};

/// IVFPQ index for the GPU
class GpuIndexIVFPQ : public GpuIndexIVF {
   public:
    /// Construct from a pre-existing faiss::IndexIVFPQ instance, copying
    /// data over to the given GPU, if the input index is trained.
    GpuIndexIVFPQ(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFPQ* index,
            GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());

    /// Construct an empty index
    GpuIndexIVFPQ(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            int subQuantizers,
            int bitsPerCode,
            faiss::MetricType metric,
            GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());

    ~GpuIndexIVFPQ() override;

    /// Reserve space on the GPU for the inverted lists for `num`
    /// vectors, assumed equally distributed among

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::IndexIVFPQ* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexIVFPQ* index) const;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(size_t numVecs);

    /// Enable or disable pre-computed codes
    void setPrecomputedCodes(bool enable);

    /// Are pre-computed codes enabled?
    bool getPrecomputedCodes() const;

    /// Return the number of sub-quantizers we are using
    int getNumSubQuantizers() const;

    /// Return the number of bits per PQ code
    int getBitsPerCode() const;

    /// Return the number of centroids per PQ code (2^bits per code)
    int getCentroidsPerSubQuantizer() const;

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory();

    /// Clears out all inverted lists, but retains the coarse and
    /// product centroid information
    void reset() override;

    /// Trains the coarse and product quantizer based on the given vector data
    void train(Index::idx_t n, const float* x) override;

    /// Returns the number of vectors present in a particular inverted list
    int getListLength(int listId) const override;

    /// Return the encoded vector data contained in a particular inverted list,
    /// for debugging purposes.
    /// If gpuFormat is true, the data is returned as it is encoded in the
    /// GPU-side representation.
    /// Otherwise, it is converted to the CPU format.
    /// compliant format, while the native GPU format may differ.
    std::vector<uint8_t> getListVectorData(int listId, bool gpuFormat = false)
            const override;

    /// Return the vector indices contained in a particular inverted list, for
    /// debugging purposes.
    std::vector<Index::idx_t> getListIndices(int listId) const override;

   public:
    /// Like the CPU version, we expose a publically-visible ProductQuantizer
    /// for manipulation
    ProductQuantizer pq;

   protected:
    /// Called from GpuIndex for add/add_with_ids
    void addImpl_(int n, const float* x, const Index::idx_t* ids) override;

    /// Called from GpuIndex for search
    void searchImpl_(
            int n,
            const float* x,
            int k,
            float* distances,
            Index::idx_t* labels) const override;

    /// Throws errors if configuration settings are improper
    void verifySettings_() const;

    /// Trains the PQ quantizer based on the given vector data
    void trainResidualQuantizer_(Index::idx_t n, const float* x);

   protected:
    /// Our configuration options that we were initialized with
    const GpuIndexIVFPQConfig ivfpqConfig_;

    /// Runtime override: whether or not we use precomputed tables
    bool usePrecomputedTables_;

    /// Number of sub-quantizers per encoded vector
    int subQuantizers_;

    /// Bits per sub-quantizer code
    int bitsPerCode_;

    /// Desired inverted list memory reservation
    size_t reserveMemoryVecs_;

    /// The product quantizer instance that we own; contains the
    /// inverted lists
    std::unique_ptr<IVFPQ> index_;
};

} // namespace gpu
} // namespace faiss
