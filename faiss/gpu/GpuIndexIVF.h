/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Clustering.h>
#include <faiss/IndexIVF.h> // for SearchParametersIVF
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <memory>

namespace faiss {
namespace gpu {

class GpuIndexFlat;
class IVFBase;

struct GpuIndexIVFConfig : public GpuIndexConfig {
    /// Index storage options for the GPU
    IndicesOptions indicesOptions = INDICES_64_BIT;

    /// Configuration for the coarse quantizer object
    GpuIndexFlatConfig flatConfig;

    /// This flag controls the CPU fallback logic for coarse quantizer
    /// component of the index. When set to false (default), the cloner will
    /// throw an exception for indices not implemented on GPU. When set to
    /// true, it will fallback to a CPU implementation.
    bool allowCpuCoarseQuantizer = false;
};

/// Base class of all GPU IVF index types. This (for now) deliberately does not
/// inherit from IndexIVF, as many of the public data members and functionality
/// in IndexIVF is not supported in the same manner on the GPU.
class GpuIndexIVF : public GpuIndex, public IndexIVFInterface {
   public:
    /// Version that auto-constructs a flat coarse quantizer based on the
    /// desired metric
    GpuIndexIVF(
            GpuResourcesProvider* provider,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            idx_t nlist,
            GpuIndexIVFConfig config = GpuIndexIVFConfig());

    /// Version that takes a coarse quantizer instance. The GpuIndexIVF does not
    /// own the coarseQuantizer instance by default (functions like IndexIVF).
    GpuIndexIVF(
            GpuResourcesProvider* provider,
            Index* coarseQuantizer,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            idx_t nlist,
            GpuIndexIVFConfig config = GpuIndexIVFConfig());

    ~GpuIndexIVF() override;

   private:
    /// Shared initialization functions
    void init_();

   public:
    /// Copy what we need from the CPU equivalent
    void copyFrom(const faiss::IndexIVF* index);

    /// Copy what we have to the CPU equivalent
    void copyTo(faiss::IndexIVF* index) const;

    /// Should be called if the user ever changes the state of the IVF coarse
    /// quantizer manually (e.g., substitutes a new instance or changes vectors
    /// in the coarse quantizer outside the scope of training)
    virtual void updateQuantizer() = 0;

    /// Returns the number of inverted lists we're managing
    virtual idx_t getNumLists() const;

    /// Returns the number of vectors present in a particular inverted list
    virtual idx_t getListLength(idx_t listId) const;

    /// Return the encoded vector data contained in a particular inverted list,
    /// for debugging purposes.
    /// If gpuFormat is true, the data is returned as it is encoded in the
    /// GPU-side representation.
    /// Otherwise, it is converted to the CPU format.
    /// compliant format, while the native GPU format may differ.
    virtual std::vector<uint8_t> getListVectorData(
            idx_t listId,
            bool gpuFormat = false) const;

    /// Return the vector indices contained in a particular inverted list, for
    /// debugging purposes.
    virtual std::vector<idx_t> getListIndices(idx_t listId) const;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const SearchParametersIVF* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    // not implemented for GPU
    void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            const idx_t* keys,
            const float* coarse_dis,
            RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

   protected:
    /// From either the current set nprobe or the SearchParameters if available,
    /// return the nprobe that we should use for the current search
    int getCurrentNProbe_(const SearchParameters* params) const;
    void verifyIVFSettings_() const;
    bool addImplRequiresIDs_() const override;
    virtual void trainQuantizer_(idx_t n, const float* x);

    /// Called from GpuIndex for add/add_with_ids
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;
    void addImpl_(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            const idx_t* ids) override;

    /// Called from GpuIndex for search
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

    void searchImpl_(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   protected:
    /// Our configuration options
    const GpuIndexIVFConfig ivfConfig_;

    /// For a trained/initialized index, this is a reference to the base class
    std::shared_ptr<IVFBase> baseIndex_;
};

} // namespace gpu
} // namespace faiss
