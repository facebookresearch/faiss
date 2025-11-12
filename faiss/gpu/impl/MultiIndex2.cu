/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/MetricType.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <algorithm>
#include <ctime>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/MultiIndex2.cuh>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/MultiSequence.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <iostream>
#include <vector>

namespace faiss {
namespace gpu {

MultiIndex2::MultiIndex2(GpuResources* res, int dim, MemorySpace space)
        : resources_(res),
          space_(space),
          numCodebooks_(2),
          dimPerCodebook_(dim / numCodebooks_),
          numCentroidsPerCodebook_(0),
          rawData_(
                  res,
                  AllocInfo(
                          AllocType::CoarseQuantizer,
                          getCurrentDevice(),
                          space,
                          res->getDefaultStreamCurrentDevice())) {
    FAISS_ASSERT(dim % numCodebooks_ == 0);
}

size_t MultiIndex2::calcMemorySpaceSize(
        int numVecsTotal,
        int dimPerCodebook,
        bool useFloat16) {
    const size_t normMemorySpaceSize = (size_t)numVecsTotal * sizeof(float);
    if (!useFloat16) {
        return (size_t)numVecsTotal * dimPerCodebook * sizeof(float) +
                normMemorySpaceSize;
    }

    return (size_t)numVecsTotal * dimPerCodebook * sizeof(half) +
            normMemorySpaceSize;
}

std::unordered_map<AllocType, size_t> MultiIndex2::getAllocSizePerTypeInfo(
        int numVecsTotal,
        int dimPerCodebook,
        bool useFloat16) {
    std::unordered_map<AllocType, size_t> allocSizePerType;
    allocSizePerType[AllocType::CoarseQuantizer] =
            calcMemorySpaceSize(numVecsTotal, dimPerCodebook, useFloat16);
    return allocSizePerType;
}

bool MultiIndex2::getUseFloat16() const {
    return false;
}

int MultiIndex2::getSize() const {
    return numCentroidsPerCodebook_ * numCentroidsPerCodebook_;
}

int MultiIndex2::getCodebookSize() const {
    return numCentroidsPerCodebook_;
}

int MultiIndex2::toMultiIndex(ushort2 indexPair) const {
    return indexPair.x + numCentroidsPerCodebook_ * indexPair.y;
}

int MultiIndex2::getDim() const {
    return numCodebooks_ * dimPerCodebook_;
}

int MultiIndex2::getSubDim() const {
    return dimPerCodebook_;
}

int MultiIndex2::getNumCodebooks() const {
    return numCodebooks_;
}

void MultiIndex2::reserve(int numVecsTotal, cudaStream_t stream) {
    rawData_.reserve(
            (unsigned)numVecsTotal * dimPerCodebook_ * sizeof(float), stream);
}

Tensor<float, 2, true>& MultiIndex2::getVectorsFloat32Ref() {
    return vectors_;
}

template <typename IndexT>
int calculateNumQueriesTilePerCodebook(
        const size_t sizeAvailable,
        const int n,
        const int d,
        const int numCodebooks,
        const int numCentroidsPerCodebook,
        const int subK) {
    constexpr int minNumQueries = 1;

    if (n <= minNumQueries) {
        return n;
    }

    idx_t distanceKernelTileRows = 0;
    idx_t distanceKernelTileCols = 0;
    chooseTileSize(
            n,
            numCentroidsPerCodebook,
            d,
            sizeof(float),
            sizeAvailable,
            distanceKernelTileRows,
            distanceKernelTileCols);
    int distanceKernelNumColTiles =
            utils::divUp(numCentroidsPerCodebook, (int)distanceKernelTileCols);

    size_t distanceBufSize = 2 * (size_t)distanceKernelTileRows *
            distanceKernelTileCols * sizeof(float);
    size_t outDistanceBufSize = 2 * (size_t)subK * distanceKernelTileRows *
            distanceKernelNumColTiles * sizeof(float);
    size_t outIndiceBufSize = 2 * (size_t)subK * distanceKernelTileRows *
            distanceKernelNumColTiles * sizeof(IndexT);
    size_t distanceKernelSize =
            distanceBufSize + outDistanceBufSize + outIndiceBufSize;

    const int sizePerQuery =
            numCodebooks * subK * (sizeof(float) + sizeof(IndexT));

    size_t multiSequenceSize = (size_t)n * sizePerQuery;

    size_t requestedSize = distanceKernelSize + multiSequenceSize;

    if (requestedSize <= sizeAvailable) {
        return n;
    }

    size_t adjustableSize = 0;
    if (distanceKernelSize > sizeAvailable) {
        if (multiSequenceSize <= sizeAvailable) {
            return n;
        }
        adjustableSize = sizeAvailable;
    } else {
        adjustableSize = sizeAvailable - distanceKernelSize;
    }

    int maxNumQueriesTile = std::min(
            n, std::max((int)(adjustableSize / sizePerQuery), minNumQueries));

    int minNumTiles = utils::divUp(n, maxNumQueriesTile);
    int numQueriesTile = utils::divUp(n, minNumTiles);

    // try to align with distance computation kernel
    constexpr size_t numQueriesAlignment = 512;
    size_t adjNumQueriesTile =
            utils::roundUp((size_t)numQueriesTile, numQueriesAlignment);

    if (adjNumQueriesTile <= maxNumQueriesTile) {
        return adjNumQueriesTile;
    }

    return numQueriesTile;
}

template <typename IndexT, typename IndexTVec2>
void MultiIndex2::queryImpl(
        Tensor<float, 2, true>& subQueries,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<IndexTVec2, 2, true>& outIndices,
        bool exactDistance) {
    if (subQueries.getSize(0) == 0 || k == 0) {
        return;
    }

    FAISS_ASSERT(subQueries.getSize(0) % numCodebooks_ == 0);
    FAISS_ASSERT(subQueries.getSize(1) == dimPerCodebook_);

    size_t sizeAvailable = resources_->getTempMemoryAvailableCurrentDevice();
    int numSubQueries = subQueries.getSize(0);
    int numSubQueriesPerCodebook = numSubQueries / numCodebooks_;
    int subK = std::min(k, numCentroidsPerCodebook_);
    subK = std::min(subK, (int)GPU_MAX_SELECTION_K);
    int numQueriesTilePerCodebook = calculateNumQueriesTilePerCodebook<IndexT>(
            sizeAvailable,
            numSubQueriesPerCodebook,
            dimPerCodebook_,
            numCodebooks_,
            numCentroidsPerCodebook_,
            subK);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    DeviceTensor<float, 3, true> outSubDistances(
            resources_,
            makeTempAlloc(AllocType::MultiSequenceInput, stream),
            {numCodebooks_, numQueriesTilePerCodebook, subK});

    DeviceTensor<IndexT, 3, true> outSubIndices(
            resources_,
            makeTempAlloc(AllocType::MultiSequenceInput, stream),
            {numCodebooks_, numQueriesTilePerCodebook, subK});

    auto allStreams = resources_->getAlternateStreamsCurrentDevice();
    // 2 streams for the first codebook and 2 streams for the second codebook
    std::vector<cudaStream_t> streams = {
            allStreams[0], allStreams[1], allStreams[2], allStreams[3]};

    for (int currentTile = 0; currentTile < numSubQueriesPerCodebook;
         currentTile += numQueriesTilePerCodebook) {
        int currentTileSize = std::min(
                numQueriesTilePerCodebook,
                numSubQueriesPerCodebook - currentTile);

        for (int i = 0; i < numCodebooks_; i++) {
            auto subQueriesView = subQueries.narrowOutermost(
                    i * numSubQueriesPerCodebook + currentTile,
                    currentTileSize);
            auto vectorsView = vectors_.narrowOutermost(
                    i * numCentroidsPerCodebook_, numCentroidsPerCodebook_);
            auto normsView = norms_.narrowOutermost(
                    i * numCentroidsPerCodebook_, numCentroidsPerCodebook_);
            auto outSubDistancesView =
                    outSubDistances[i].view().narrowOutermost(
                            0, currentTileSize);
            auto outSubIndicesView =
                    outSubIndices[i].view().narrowOutermost(0, currentTileSize);

            runL2Distance(
                    resources_,
                    stream,
                    vectorsView,
                    true, // vectors is row major
                    &normsView,
                    subQueriesView,
                    true, // input is row major
                    subK,
                    outSubDistancesView,
                    outSubIndicesView,
                    {streams[0], streams[1]},
                    !exactDistance);
        }

        auto outDistancesView =
                outDistances.narrowOutermost(currentTile, currentTileSize);
        auto outIndicesView =
                outIndices.narrowOutermost(currentTile, currentTileSize);

        // use the first stream from current tile to compute multi-sequence
        runMultiSequence2(
                currentTileSize,
                subK,
                k,
                outSubDistances,
                outSubIndices,
                outDistancesView,
                outIndicesView,
                resources_);
    }
}

template <typename IndexT>
void MultiIndex2::queryImpl(
        Tensor<float, 2, true>& subQueries,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool exactDistance) {
    if (subQueries.getSize(0) == 0 || k == 0) {
        return;
    }

    FAISS_ASSERT(subQueries.getSize(0) % numCodebooks_ == 0);
    FAISS_ASSERT(subQueries.getSize(1) == dimPerCodebook_);

    size_t sizeAvailable = resources_->getTempMemoryAvailableCurrentDevice();
    int numSubQueries = subQueries.getSize(0);
    int numSubQueriesPerCodebook = numSubQueries / numCodebooks_;
    int subK = std::min(k, numCentroidsPerCodebook_);
    subK = std::min(subK, (int)GPU_MAX_SELECTION_K);
    int numQueriesTilePerCodebook = calculateNumQueriesTilePerCodebook<IndexT>(
            sizeAvailable,
            numSubQueriesPerCodebook,
            dimPerCodebook_,
            numCodebooks_,
            numCentroidsPerCodebook_,
            subK);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    DeviceTensor<float, 3, true> outSubDistances(
            resources_,
            makeTempAlloc(AllocType::MultiSequenceInput, stream),
            {numCodebooks_, numQueriesTilePerCodebook, subK});

    DeviceTensor<IndexT, 3, true> outSubIndices(
            resources_,
            makeTempAlloc(AllocType::MultiSequenceInput, stream),
            {numCodebooks_, numQueriesTilePerCodebook, subK});

    auto allStreams = resources_->getAlternateStreamsCurrentDevice();
    // 2 streams for the first codebook and 2 streams for the second codebook
    std::vector<cudaStream_t> streams = {
            allStreams[0], allStreams[1], allStreams[2], allStreams[3]};

    for (int currentTile = 0; currentTile < numSubQueriesPerCodebook;
         currentTile += numQueriesTilePerCodebook) {
        int currentTileSize = std::min(
                numQueriesTilePerCodebook,
                numSubQueriesPerCodebook - currentTile);

        for (int i = 0; i < numCodebooks_; i++) {
            auto subQueriesView = subQueries.narrowOutermost(
                    i * numSubQueriesPerCodebook + currentTile,
                    currentTileSize);
            auto vectorsView = vectors_.narrowOutermost(
                    i * numCentroidsPerCodebook_, numCentroidsPerCodebook_);
            auto normsView = norms_.narrowOutermost(
                    i * numCentroidsPerCodebook_, numCentroidsPerCodebook_);
            auto outSubDistancesView =
                    outSubDistances[i].view().narrowOutermost(
                            0, currentTileSize);
            auto outSubIndicesView =
                    outSubIndices[i].view().narrowOutermost(0, currentTileSize);

            runL2Distance(
                    resources_,
                    stream,
                    vectorsView,
                    true, // vectors is row major
                    &normsView,
                    subQueriesView,
                    true, // input is row major
                    subK,
                    outSubDistancesView,
                    outSubIndicesView,
                    {streams[0], streams[1]},
                    !exactDistance);
        }

        auto outDistancesView =
                outDistances.narrowOutermost(currentTile, currentTileSize);
        auto outIndicesView =
                outIndices.narrowOutermost(currentTile, currentTileSize);

        // use the first stream from current tile to compute multi-sequence
        runMultiSequence2(
                currentTileSize,
                subK,
                k,
                outSubDistances,
                outSubIndices,
                outDistancesView,
                numCentroidsPerCodebook_,
                outIndicesView,
                resources_);
    }
}

void MultiIndex2::query(
        Tensor<float, 2, true>& subQueries,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<int2, 2, true>& outIndices,
        bool exactDistance) {
    queryImpl<idx_t, int2>(
            subQueries, k, outDistances, outIndices, exactDistance);
}

void MultiIndex2::query(
        Tensor<float, 2, true>& subQueries,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<ushort2, 2, true>& outIndices,
        bool exactDistance) {
    queryImpl<unsigned short, ushort2>(
            subQueries, k, outDistances, outIndices, exactDistance);
}

void MultiIndex2::query(
        Tensor<float, 2, true>& subQueries,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool exactDistance) {
    queryImpl<unsigned short>(
            subQueries, k, outDistances, outIndices, exactDistance);
}

void MultiIndex2::computeResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<float, 2, true>& residuals) {
    runCalcResidual(
            vecs,
            vectors_,
            listIds,
            residuals,
            resources_->getDefaultStreamCurrentDevice());
}

void MultiIndex2::computeResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<int2, 1, true>& listIds,
        Tensor<float, 2, true>& residuals) {
    runCalcResidual(
            vecs,
            vectors_,
            listIds,
            residuals,
            resources_->getDefaultStreamCurrentDevice());
}

void MultiIndex2::add(
        const float* data,
        int numVecsTotal,
        cudaStream_t stream) {
    FAISS_ASSERT(numVecsTotal % numCodebooks_ == 0);
    FAISS_ASSERT(numCentroidsPerCodebook_ == 0);
    if (numVecsTotal == 0) {
        return;
    }

    rawData_.append(
            (char*)data,
            (unsigned)numVecsTotal * dimPerCodebook_ * sizeof(float),
            stream,
            true /* reserve exactly */);

    numCentroidsPerCodebook_ += numVecsTotal / numCodebooks_;

    DeviceTensor<float, 2, true> vectors(
            (float*)rawData_.data(), {numVecsTotal, dimPerCodebook_});
    vectors_ = std::move(vectors);

    DeviceTensor<float, 1, true> norms(
            resources_,
            makeSpaceAlloc(AllocType::FlatData, space_, stream),
            {numVecsTotal});
    runL2Norm(vectors_, true, norms, true, stream);
    norms_ = std::move(norms);
}

void MultiIndex2::reset() {
    rawData_.clear();
    vectors_ = DeviceTensor<float, 2, true>();
    norms_ = DeviceTensor<float, 1, true>();
    numCentroidsPerCodebook_ = 0;
}

} // namespace gpu
} // namespace faiss
