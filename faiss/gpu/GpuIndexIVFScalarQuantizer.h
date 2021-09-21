/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <memory>

namespace faiss {
namespace gpu {

class IVFFlat;
class GpuIndexFlat;

struct GpuIndexIVFScalarQuantizerConfig : public GpuIndexIVFConfig {
    inline GpuIndexIVFScalarQuantizerConfig() : interleavedLayout(true) {}

    /// Use the alternative memory layout for the IVF lists
    /// (currently the default)
    bool interleavedLayout;
};

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexIVFScalarQuantizer
class GpuIndexIVFScalarQuantizer : public GpuIndexIVF {
   public:
    /// Construct from a pre-existing faiss::IndexIVFScalarQuantizer instance,
    /// copying data over to the given GPU, if the input index is trained.
    GpuIndexIVFScalarQuantizer(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFScalarQuantizer* index,
            GpuIndexIVFScalarQuantizerConfig config =
                    GpuIndexIVFScalarQuantizerConfig());

    /// Constructs a new instance with an empty flat quantizer; the user
    /// provides the number of lists desired.
    GpuIndexIVFScalarQuantizer(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            faiss::ScalarQuantizer::QuantizerType qtype,
            faiss::MetricType metric = MetricType::METRIC_L2,
            bool encodeResidual = true,
            GpuIndexIVFScalarQuantizerConfig config =
                    GpuIndexIVFScalarQuantizerConfig());

    ~GpuIndexIVFScalarQuantizer() override;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(size_t numVecs);

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::IndexIVFScalarQuantizer* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexIVFScalarQuantizer* index) const;

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory();

    /// Clears out all inverted lists, but retains the coarse and scalar
    /// quantizer information
    void reset() override;

    /// Trains the coarse and scalar quantizer based on the given vector data
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

    /// Called from train to handle SQ residual training
    void trainResiduals_(Index::idx_t n, const float* x);

   public:
    /// Exposed like the CPU version
    faiss::ScalarQuantizer sq;

    /// Exposed like the CPU version
    bool by_residual;

   protected:
    /// Our configuration options
    const GpuIndexIVFScalarQuantizerConfig ivfSQConfig_;

    /// Desired inverted list memory reservation
    size_t reserveMemoryVecs_;

    /// Instance that we own; contains the inverted list
    std::unique_ptr<IVFFlat> index_;
};

} // namespace gpu
} // namespace faiss
