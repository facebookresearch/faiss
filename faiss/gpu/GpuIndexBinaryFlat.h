/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexBinaryFlat.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <memory>

namespace faiss {
namespace gpu {

class BinaryFlatIndex;

struct GpuIndexBinaryFlatConfig : public GpuIndexConfig {};

/// A GPU version of IndexBinaryFlat for brute-force comparison of bit vectors
/// via Hamming distance
class GpuIndexBinaryFlat : public IndexBinary {
   public:
    /// Construct from a pre-existing faiss::IndexBinaryFlat instance, copying
    /// data over to the given GPU
    GpuIndexBinaryFlat(
            GpuResourcesProvider* resources,
            const faiss::IndexBinaryFlat* index,
            GpuIndexBinaryFlatConfig config = GpuIndexBinaryFlatConfig());

    /// Construct an empty instance that can be added to
    GpuIndexBinaryFlat(
            GpuResourcesProvider* resources,
            int dims,
            GpuIndexBinaryFlatConfig config = GpuIndexBinaryFlatConfig());

    ~GpuIndexBinaryFlat() override;

    /// Returns the device that this index is resident on
    int getDevice() const;

    /// Returns a reference to our GpuResources object that manages memory,
    /// stream and handle resources on the GPU
    std::shared_ptr<GpuResources> getResources();

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::IndexBinaryFlat* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexBinaryFlat* index) const;

    void add(faiss::idx_t n, const uint8_t* x) override;
    void add(idx_t n, const void* x, NumericType numeric_type) override;

    void reset() override;

    void search(
            idx_t n,
            const uint8_t* x,
            // faiss::IndexBinary has idx_t for k
            idx_t k,
            int32_t* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override;
    void search(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(faiss::idx_t key, uint8_t* recons) const override;

   protected:
    /// Called from search when the input data is on the CPU;
    /// potentially allows for pinned memory usage
    void searchFromCpuPaged_(
            idx_t n,
            const uint8_t* x,
            int k,
            int32_t* outDistancesData,
            idx_t* outIndicesData) const;

    void searchNonPaged_(
            idx_t n,
            const uint8_t* x,
            int k,
            int32_t* outDistancesData,
            idx_t* outIndicesData) const;

   protected:
    /// Manages streans, cuBLAS handles and scratch memory for devices
    std::shared_ptr<GpuResources> resources_;

    /// Configuration options
    const GpuIndexBinaryFlatConfig binaryFlatConfig_;

    /// Holds our GPU data containing the list of vectors
    std::unique_ptr<BinaryFlatIndex> data_;
};

} // namespace gpu
} // namespace faiss
