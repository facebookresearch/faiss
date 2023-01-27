/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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
    inline GpuIndexIVFConfig() : indicesOptions(INDICES_64_BIT) {}

    /// Index storage options for the GPU
    IndicesOptions indicesOptions;

    /// Configuration for the coarse quantizer object
    GpuIndexFlatConfig flatConfig;
};

/// Base class of all GPU IVF index types. This (for now) deliberately does not
/// inherit from IndexIVF, as many of the public data members and functionality
/// in IndexIVF is not supported in the same manner on the GPU.
class GpuIndexIVF : public GpuIndex {
   public:
    /// Version that auto-constructs a flat coarse quantizer based on the
    /// desired metric
    GpuIndexIVF(
            GpuResourcesProvider* provider,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            int nlist,
            GpuIndexIVFConfig config = GpuIndexIVFConfig());

    /// Version that takes a coarse quantizer instance. The GpuIndexIVF does not
    /// own the coarseQuantizer instance by default (functions like IndexIVF).
    GpuIndexIVF(
            GpuResourcesProvider* provider,
            Index* coarseQuantizer,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            int nlist,
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
    int getNumLists() const;

    /// Returns the number of vectors present in a particular inverted list
    int getListLength(int listId) const;

    /// Return the encoded vector data contained in a particular inverted list,
    /// for debugging purposes.
    /// If gpuFormat is true, the data is returned as it is encoded in the
    /// GPU-side representation.
    /// Otherwise, it is converted to the CPU format.
    /// compliant format, while the native GPU format may differ.
    std::vector<uint8_t> getListVectorData(int listId, bool gpuFormat = false)
            const;

    /// Return the vector indices contained in a particular inverted list, for
    /// debugging purposes.
    std::vector<idx_t> getListIndices(int listId) const;

    /// Sets the number of list probes per query
    void setNumProbes(int nprobe);

    /// Returns our current number of list probes per query
    int getNumProbes() const;

    /// Same interface as faiss::IndexIVF, in order to search a set of vectors
    /// pre-quantized by the IVF quantizer. Does not include IndexIVFStats as
    /// that can only be obtained on the host via a GPU d2h copy.
    /// @param n      nb of vectors to query
    /// @param x      query vectors, size nx * d
    /// @param assign coarse quantization indices, size nx * nprobe
    /// @param centroid_dis
    ///             distances to coarse centroids, size nx * nprobe
    /// @param distance
    ///             output distances, size n * k
    /// @param labels output labels, size n * k
    /// @param store_pairs store inv list index + inv list offset
    ///                   instead in upper/lower 32 bit of result,
    ///                   instead of ids (used for reranking).
    /// @param params used to override the object's search parameters
    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const SearchParametersIVF* params = nullptr) const;

   protected:
    void verifyIVFSettings_() const;
    bool addImplRequiresIDs_() const override;
    void trainQuantizer_(idx_t n, const float* x);

    /// Called from GpuIndex for add/add_with_ids
    void addImpl_(int n, const float* x, const idx_t* ids) override;

    /// Called from GpuIndex for search
    void searchImpl_(
            int n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   public:
    /// Exposing this like the CPU version for manipulation
    ClusteringParameters cp;

    /// Exposing this like the CPU version for query
    int nlist;

    /// Exposing this like the CPU version for manipulation
    int nprobe;

    /// A user-pluggable coarse quantizer
    Index* quantizer;

    /// Whether or not we own the coarse quantizer
    bool own_fields;

   protected:
    /// Our configuration options
    const GpuIndexIVFConfig ivfConfig_;

    /// For a trained/initialized index, this is a reference to the base class
    std::shared_ptr<IVFBase> baseIndex_;
};

} // namespace gpu
} // namespace faiss
