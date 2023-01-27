/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuResources.h>

namespace faiss {
namespace gpu {

struct GpuIndexConfig {
    inline GpuIndexConfig() : device(0), memorySpace(MemorySpace::Device) {}

    /// GPU device on which the index is resident
    int device;

    /// What memory space to use for primary storage.
    /// On Pascal and above (CC 6+) architectures, allows GPUs to use
    /// more memory than is available on the GPU.
    MemorySpace memorySpace;
};

class GpuIndex : public faiss::Index {
   public:
    GpuIndex(
            std::shared_ptr<GpuResources> resources,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            GpuIndexConfig config);

    /// Returns the device that this index is resident on
    int getDevice() const;

    /// Returns a reference to our GpuResources object that manages memory,
    /// stream and handle resources on the GPU
    std::shared_ptr<GpuResources> getResources();

    /// Set the minimum data size for searches (in MiB) for which we use
    /// CPU -> GPU paging
    void setMinPagingSize(size_t size);

    /// Returns the current minimum data size for paged searches
    size_t getMinPagingSize() const;

    /// `x` can be resident on the CPU or any GPU; copies are performed
    /// as needed
    /// Handles paged adds if the add set is too large; calls addInternal_
    void add(idx_t, const float* x) override;

    /// `x` and `ids` can be resident on the CPU or any GPU; copies are
    /// performed as needed
    /// Handles paged adds if the add set is too large; calls addInternal_
    void add_with_ids(idx_t n, const float* x, const idx_t* ids) override;

    /// `x` and `labels` can be resident on the CPU or any GPU; copies are
    /// performed as needed
    void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1)
            const override;

    /// `x`, `distances` and `labels` can be resident on the CPU or any
    /// GPU; copies are performed as needed
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// `x`, `distances` and `labels` and `recons` can be resident on the CPU or
    /// any GPU; copies are performed as needed
    void search_and_reconstruct(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* recons,
            const SearchParameters* params = nullptr) const override;

    /// Overridden to force GPU indices to provide their own GPU-friendly
    /// implementation
    void compute_residual(const float* x, float* residual, idx_t key)
            const override;

    /// Overridden to force GPU indices to provide their own GPU-friendly
    /// implementation
    void compute_residual_n(
            idx_t n,
            const float* xs,
            float* residuals,
            const idx_t* keys) const override;

   protected:
    /// Copy what we need from the CPU equivalent
    void copyFrom(const faiss::Index* index);

    /// Copy what we have to the CPU equivalent
    void copyTo(faiss::Index* index) const;

    /// Does addImpl_ require IDs? If so, and no IDs are provided, we will
    /// generate them sequentially based on the order in which the IDs are added
    virtual bool addImplRequiresIDs_() const = 0;

    /// Overridden to actually perform the add
    /// All data is guaranteed to be resident on our device
    virtual void addImpl_(int n, const float* x, const idx_t* ids) = 0;

    /// Overridden to actually perform the search
    /// All data is guaranteed to be resident on our device
    virtual void searchImpl_(
            int n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const = 0;

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

   protected:
    /// Manages streams, cuBLAS handles and scratch memory for devices
    std::shared_ptr<GpuResources> resources_;

    /// Our configuration options
    const GpuIndexConfig config_;

    /// Size above which we page copies from the CPU to GPU
    size_t minPagedSize_;
};

/// If the given index is a GPU index, this returns the index instance
GpuIndex* tryCastGpuIndex(faiss::Index* index);

/// Is the given index instance a GPU index?
bool isGpuIndex(faiss::Index* index);

/// Does the given CPU index instance have a corresponding GPU implementation?
bool isGpuIndexImplemented(faiss::Index* index);

} // namespace gpu
} // namespace faiss
