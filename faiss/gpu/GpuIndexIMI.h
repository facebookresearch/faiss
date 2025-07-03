/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Clustering.h>
#include <faiss/IndexIVF.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuMultiIndex2.h>
#include <faiss/gpu/GpuResources.h>
#include <memory>

namespace faiss {
namespace gpu {

struct GpuIndexIMIConfig : public GpuIndexConfig {
    inline GpuIndexIMIConfig()
            : indicesOptions(INDICES_64_BIT), forcePinnedMemory(false) {}

    /// Index storage options for the GPU
    IndicesOptions indicesOptions;

    bool forcePinnedMemory;

    /// Configuration for the coarse quantizer object
    GpuMultiIndex2Config multiIndexConfig;
};

class GpuIndexIMI : public GpuIndex {
   public:
    GpuIndexIMI(
            GpuResourcesProvider* provider,
            int dims,
            int coarseCodebookSize,
            GpuIndexIMIConfig config = GpuIndexIMIConfig());

    GpuIndexIMI(
            std::shared_ptr<GpuResources> resources,
            int dims,
            int coarseCodebookSize,
            GpuIndexIMIConfig config = GpuIndexIMIConfig());

    ~GpuIndexIMI() override;

    /// Calculate the memory space in bytes for storing in the index a given
    /// number of vectors according to given dimension and the stored format
    static size_t calcMemorySpaceSizeCoarseQuantizer(
            int numVecsTotal,
            int dimPerCodebook,
            bool useFloat16);

    /// Copy what we need from the CPU equivalent
    void copyFrom(const faiss::IndexIVF* index);

    /// Copy what we have to the CPU equivalent
    void copyTo(faiss::IndexIVF* index) const;

    /// Returns the number of inverted lists we're managing
    int getNumLists() const;

    /// Returns the number of vectors present in a particular inverted list
    virtual int getListLength(int listId) const = 0;

    /// Return the encoded vector data contained in a particular inverted list,
    /// for debugging purposes.
    /// If gpuFormat is true, the data is returned as it is encoded in the
    /// GPU-side representation.
    /// Otherwise, it is converted to the CPU format.
    /// compliant format, while the native GPU format may differ.
    virtual std::vector<uint8_t> getListVectorData(
            int listId,
            bool gpuFormat = false) const = 0;

    /// Return the vector indices contained in a particular inverted list, for
    /// debugging purposes.
    virtual std::vector<idx_t> getListIndices(int listId) const = 0;

    /// Return the quantizer we're using
    GpuMultiIndex2* getQuantizer();

    /// Sets the number of list probes per query
    void setNumProbes(int nprobe);

    /// Returns our current number of list probes per query
    int getNumProbes() const;

    /// `x` and `ids` can be resident on the CPU or any GPU; copies are
    /// performed as needed
    /// Handles paged adds if the add set is too large; calls addInternal_
    void add_with_ids(idx_t n, const float* x, const idx_t* ids) override;

    /// `x`, `distances` and `labels` can be resident on the CPU or any
    /// GPU; copies are performed as needed
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

   protected:
    /// Flat index does not require IDs as there is no storage available for
    /// them
    bool addImplRequiresIDs_() const override;

    void trainQuantizer_(faiss::idx_t n, const float* x);

   private:
    /// Handles paged adds if the add set is too large, passes to
    /// addImpl_ to actually perform the add for the current page
    void addPaged_(int n, const float* x, const idx_t* ids);

    /// Calls addImpl_ for a single page of GPU-resident data
    void addPage_(int n, const float* x, const idx_t* ids);

    /// Calls searchImpl_ for a single page of GPU-resident data
    void searchNonPaged_(
            int n,
            const float* x,
            int k,
            float* outDistancesData,
            idx_t* outIndicesData,
            const SearchParameters* params) const;

    /// Calls searchImpl_ for a single page of GPU-resident data,
    /// handling paging of the data and copies from the CPU
    void searchFromCpuPaged_(
            int n,
            const float* x,
            int k,
            float* outDistancesData,
            idx_t* outIndicesData,
            const SearchParameters* params) const;

   public:
    /// Exposing this like the CPU version for manipulation
    ClusteringParameters cp;

    /// Exposing this like the CPU version for query
    int nlist;

    /// Exposing this like the CPU version for manipulation
    int nprobe;

    /// Exposing this like the CPU version for query
    GpuMultiIndex2* quantizer;

   protected:
    /// Our configuration options
    const GpuIndexIMIConfig imiConfig_;
};

} // namespace gpu
} // namespace faiss
