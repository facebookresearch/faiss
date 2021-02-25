/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Clustering.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndicesOptions.h>

namespace faiss {
struct IndexIVF;
}

namespace faiss {
namespace gpu {

class GpuIndexFlat;

struct GpuIndexIVFConfig : public GpuIndexConfig {
    inline GpuIndexIVFConfig() : indicesOptions(INDICES_64_BIT) {}

    /// Index storage options for the GPU
    IndicesOptions indicesOptions;

    /// Configuration for the coarse quantizer object
    GpuIndexFlatConfig flatConfig;
};

class GpuIndexIVF : public GpuIndex {
   public:
    GpuIndexIVF(
            GpuResourcesProvider* provider,
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
    virtual std::vector<Index::idx_t> getListIndices(int listId) const = 0;

    /// Return the quantizer we're using
    GpuIndexFlat* getQuantizer();

    /// Sets the number of list probes per query
    void setNumProbes(int nprobe);

    /// Returns our current number of list probes per query
    int getNumProbes() const;

   protected:
    bool addImplRequiresIDs_() const override;
    void trainQuantizer_(Index::idx_t n, const float* x);

   public:
    /// Exposing this like the CPU version for manipulation
    ClusteringParameters cp;

    /// Exposing this like the CPU version for query
    int nlist;

    /// Exposing this like the CPU version for manipulation
    int nprobe;

    /// Exposeing this like the CPU version for query
    GpuIndexFlat* quantizer;

   protected:
    /// Our configuration options
    const GpuIndexIVFConfig ivfConfig_;
};

} // namespace gpu
} // namespace faiss
