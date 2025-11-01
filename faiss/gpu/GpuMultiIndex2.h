/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexPQ.h>
#include <faiss/gpu/GpuIndex.h>
#include <memory>
#include <utility>
#include <vector>

namespace faiss {
namespace gpu {

struct MultiIndex2;

struct GpuMultiIndex2Config : GpuIndexConfig {
    inline GpuMultiIndex2Config() {}
};

class GpuMultiIndex2 : public GpuIndex {
   public:
    /// Construct from a pre-existing faiss::IndexFlat instance, copying
    /// data over to the given GPU
    GpuMultiIndex2(
            GpuResourcesProvider* provider,
            const faiss::MultiIndexQuantizer* index,
            GpuMultiIndex2Config config = GpuMultiIndex2Config());

    GpuMultiIndex2(
            std::shared_ptr<GpuResources> resources,
            const faiss::MultiIndexQuantizer* index,
            GpuMultiIndex2Config config = GpuMultiIndex2Config());

    GpuMultiIndex2(
            GpuResourcesProvider* provider,
            int dims,
            int numCentroidsPerCodebook,
            GpuMultiIndex2Config config = GpuMultiIndex2Config());

    GpuMultiIndex2(
            std::shared_ptr<GpuResources> resources,
            int dims,
            int numCentroidsPerCodebook,
            GpuMultiIndex2Config config = GpuMultiIndex2Config());

    ~GpuMultiIndex2() override;

   public:
    static size_t calcMemorySpaceSize(
            int numVecsTotal,
            int dimPerCodebook,
            bool useFloat16);

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::MultiIndexQuantizer* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::MultiIndexQuantizer* index) const;

    /// Calculate the inverted list id from the centroid id pair from the
    /// codebooks
    int toMultiIndex(std::pair<ushort, ushort> indexPair) const;

    /// Return the size of each codebook
    int getCodebookSize();

    /// Return the number of codebooks
    int getNumCodebooks();

    /// Returns the number of vectors we address
    int getNumVecs();

    int getSubDim();

    /// Returns centrois (numCodebooks * codebookSize * subDim) for debugging
    /// purpose
    std::vector<float> getCentroids();

    // centroids (numCodebooks * codebookSize, subDim)
    void load(int codebookSize, const float* centroids);

    /// Clears all vectors from this index
    void reset() override;

    /// Trains the codebooks based on the given vector data
    void train(idx_t n, const float* x) override;

    /// `x` can be resident on the CPU or any GPU; copies are performed
    /// as needed
    /// Handles paged adds if the add set is too large; calls addInternal_
    void add(faiss::idx_t, const float* x) override;

    /// `x` and `ids` can be resident on the CPU or any GPU; copies are
    /// performed as needed
    /// Handles paged adds if the add set is too large; calls addInternal_
    void add_with_ids(idx_t n, const float* x, const idx_t* ids) override;

    /// `x` and `labels` can be resident on the CPU or any GPU; copies are
    /// performed as needed
    void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1)
            const override;

    /// `x` and `labels` can be resident on the CPU or any GPU; copies are
    /// performed as needed. The `labels` vector is assigned with the pair
    /// of subcentroid ids, one of each codebok
    void assign_pair(
            idx_t n,
            const float* x,
            std::pair<ushort, ushort>* labels,
            idx_t k = 1) const;

    /// `x`, `distances` and `labels` can be resident on the CPU or any
    /// GPU; copies are performed as needed
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// `x` and `labels` can be resident on the CPU or any GPU; copies are
    /// performed as needed. The `labels` vector is assigned with the pair
    /// of subcentroid ids, one of each codebok
    void search_pair(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            std::pair<ushort, ushort>* labels) const;

    /// Calculate the residual vector between the a single input vector and the
    /// virtual centroid of the related subcentroid id pair
    void compute_residual_pair(
            const float* xs,
            float* residuals,
            std::pair<ushort, ushort> key) const;

    /// Calculate the residual vector between the n input vectors and the
    /// virtual centroid of the related subcentroid id pair
    void compute_residual_n_pair(
            faiss::idx_t n,
            const float* xs,
            float* residuals,
            const std::pair<ushort, ushort>* keys) const;

    /// Search the subcentroid id pair for each input vector and
    /// calculate the residual vector between the n input vectors and the
    /// virtual centroid of the related subcentroid id pair
    void compute_nearest_residual_n(
            faiss::idx_t n,
            const float* xs,
            float* residuals) const;

    /// For internal access
    inline MultiIndex2* getGpuData() {
        return data_.get();
    }

    static const int NUM_CODEBOOKS;

    /// Exposing this for manipulation
    ClusteringParameters cp;

   protected:
    bool addImplRequiresIDs_() const override;

    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void searchPairImpl_(
            int n,
            const float* x,
            int k,
            float* distances,
            std::pair<ushort, ushort>* labels) const;

    int numVecsPerCodebook_, subDim_;

    /// Our configuration options
    const GpuMultiIndex2Config config_;

    /// Holds our GPU data containing the list of vectors
    std::unique_ptr<MultiIndex2> data_;

   private:
    /// Shared initialization functions
    void init_();
    /// Calls searchImpl_ for a single page of GPU-resident data
    void searchNonPaged_(
            int n,
            const float* x,
            int k,
            float* outDistancesData,
            idx_t* outIndicesData) const;

    /// Calls searchImpl_ for a single page of GPU-resident data,
    /// handling paging of the data and copies from the CPU
    void searchFromCpuPaged_(
            int n,
            const float* x,
            int k,
            float* outDistancesData,
            idx_t* outIndicesData) const;

    /// Calls searchImpl_ for a single page of GPU-resident data
    void searchNonPaged_(
            int n,
            const float* x,
            int k,
            float* outDistancesData,
            std::pair<ushort, ushort>* outIndicesData) const;

    /// Calls searchImpl_ for a single page of GPU-resident data,
    /// handling paging of the data and copies from the CPU
    void searchFromCpuPaged_(
            int n,
            const float* x,
            int k,
            float* outDistancesData,
            std::pair<ushort, ushort>* outIndicesData) const;
};

} // namespace gpu
} // namespace faiss
