/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/impl/AdditiveQuantizer.h>
#include <memory>
#include <vector>

namespace faiss {
namespace gpu {

class GpuIndexFlat;
class IVFAQ;

struct GpuIndexIVFAQConfig : public GpuIndexIVFConfig {
    inline GpuIndexIVFAQConfig()
            : useFloat16LookupTables(false),
              usePrecomputedTables(false),
              interleavedLayout(false),
              useMMCodeDistance(false) {}
    /// When subQuantizers * 2^(bitsPerCode) > 2^14,
    /// this is required.
    bool useFloat16LookupTables;

    /// this can substantially increase the memory requirement
    bool usePrecomputedTables;

    // bool interleavedLayout

    bool useMMCodeDistance;
}

/// IVFAQ index for the GPU
class GpuIndexIVFAQ : public GpuIndexIVF {
   public:
    /// Construct from a pre-existing faiss::IndexIVFAQ instance, copying
    /// data over to the given GPU, if the input index is trained.
    GpuIndexIVFAQ(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFAQ* index,
            GpuIndexIVFAQConfig config = GpuIndexIVFAQConfig());

    /// Construct an empty index
    GpuIndexIVFAQ(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            faiss::MetricType metric,
            GpuIndexIVFAQConfig config = GpuIndexIVFAQConfig());
    ~GpuIndexIVFPQ() override;

    void copyFrom(const faiss::IndexIVFAQ* index);

    void copyTo(faiss::IndexIVFAQ* index) const;

    void reserveMemory(size_t numVecs);

    void setPrecomputedCodes(bool enable);

    bool getPrecomputedCodes() const;

    int getNumSubQuantizers() const;

    int getBitsPerCode() const;

    /// Return the number of bits per AQ code
    /// (2^bits per code)
    int getCentroidsPerSubQuantizer() const;

    size_t reclaimMemory();

    void reset() override;

    void train(Index::idx_t n, const float* x) override;

    /// for debugging purposes
    std::vector<uint8_t> getListVectorData(int listId, bool gpuFormat = false)
            const override;

    std::vector<Index::idx_t> getListIndices(int listId) const override;

   public:
    /// expose a publically-visible AQ for manipulation
    AdditiveQuantizer aq;

   protected:
    /// called from GpuIndex for add/add_with_ids
    void addImpl_(int n, const float* x, const Index::idx_t* ids) override;

    void searchImpl_(
            int n,
            const float* x,
            int k,
            float* distances,
            Index::idx_t* labels) const override;

    void verifySettings_() const;

    void trainResidualQuantizer_(Index::idx_t n, const float* x);

   protected:
    const GpuIndexIVFAQConfig ivfaqConfig_;

    bool usePrecomputedTables_;

    int subQuantizers_;

    int bitsPerCode_;

    size_t reserveMemoryVecs_;

    std::unique_ptr<IVFAQ> index_;
};

} // namespace gpu
} // namespace faiss