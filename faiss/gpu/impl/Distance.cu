/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/BroadcastSum.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/L2Select.cuh>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <memory>

namespace faiss {
namespace gpu {

template <typename T>
void runAllPairwiseDistance(
        bool computeL2,
        GpuResources* res,
        cudaStream_t stream,
        Tensor<T, 2, true>& centroids,
        bool centroidsRowMajor,
        Tensor<float, 1, true>* centroidNorms,
        Tensor<T, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances) {
    // The # of centroids in `centroids` based on memory layout
    auto numCentroids = centroids.getSize(centroidsRowMajor ? 0 : 1);

    // The # of queries in `queries` based on memory layout
    auto numQueries = queries.getSize(queriesRowMajor ? 0 : 1);

    // The dimensions of the vectors to consider
    auto dim = queries.getSize(queriesRowMajor ? 1 : 0);
    FAISS_ASSERT(
            (numQueries == 0 || numCentroids == 0) ||
            dim == centroids.getSize(centroidsRowMajor ? 1 : 0));

    FAISS_ASSERT(outDistances.getSize(0) == numQueries);
    FAISS_ASSERT(outDistances.getSize(1) == numCentroids);

    // If we're querying against a 0 sized set, just return empty results
    if (centroids.numElements() == 0) {
        thrust::fill(
                thrust::cuda::par.on(stream),
                outDistances.data(),
                outDistances.end(),
                Limits<float>::getMax());

        return;
    }

    // L2: If ||c||^2 is not pre-computed, calculate it
    DeviceTensor<float, 1, true> cNorms;
    if (computeL2 && !centroidNorms) {
        cNorms = DeviceTensor<float, 1, true>(
                res, makeTempAlloc(AllocType::Other, stream), {numCentroids});
        runL2Norm(centroids, centroidsRowMajor, cNorms, true, stream);
        centroidNorms = &cNorms;
    }

    //
    // Prepare norm vector ||q||^2; ||c||^2 is already pre-computed
    //
    DeviceTensor<float, 1, true> queryNorms(
            res, makeTempAlloc(AllocType::Other, stream), {numQueries});

    // ||q||^2
    if (computeL2) {
        runL2Norm(queries, queriesRowMajor, queryNorms, true, stream);
    }

    // L2: distance is ||c||^2 - 2qc + ||q||^2, we compute -2qc
    // IP: just compute qc
    // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
    runMatrixMult(
            outDistances,
            false, // not transposed
            queries,
            !queriesRowMajor, // transposed MM if col major
            centroids,
            centroidsRowMajor, // transposed MM if row major
            computeL2 ? -2.0f : 1.0f,
            0.0f,
            res->getBlasHandleCurrentDevice(),
            stream);

    if (computeL2) {
        // Need to add ||q||^2 along rows
        // Need to add ||c||^2 along columns
        // FIXME: optimize with a dedicated kernel
        runSumAlongColumns(*centroidNorms, outDistances, stream);

        runSumAlongRows(
                queryNorms,
                outDistances,
                true, // L2 distances should not go below zero
                // due to roundoff error
                stream);
    }
}

template <typename T>
void runDistance(
        bool computeL2,
        GpuResources* res,
        cudaStream_t stream,
        Tensor<T, 2, true>& centroids,
        bool centroidsRowMajor,
        Tensor<float, 1, true>* centroidNorms,
        Tensor<T, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool ignoreOutDistances) {
    // The # of centroids in `centroids` based on memory layout
    auto numCentroids = centroids.getSize(centroidsRowMajor ? 0 : 1);

    // The # of queries in `queries` based on memory layout
    auto numQueries = queries.getSize(queriesRowMajor ? 0 : 1);

    // The dimensions of the vectors to consider
    auto dim = queries.getSize(queriesRowMajor ? 1 : 0);
    FAISS_ASSERT(
            (numQueries == 0 || numCentroids == 0) ||
            dim == centroids.getSize(centroidsRowMajor ? 1 : 0));

    FAISS_ASSERT(outDistances.getSize(0) == numQueries);
    FAISS_ASSERT(outIndices.getSize(0) == numQueries);
    FAISS_ASSERT(outDistances.getSize(1) == k);
    FAISS_ASSERT(outIndices.getSize(1) == k);

    // If we're querying against a 0 sized set, just return empty results
    if (centroids.numElements() == 0) {
        thrust::fill(
                thrust::cuda::par.on(stream),
                outDistances.data(),
                outDistances.end(),
                Limits<float>::getMax());

        thrust::fill(
                thrust::cuda::par.on(stream),
                outIndices.data(),
                outIndices.end(),
                -1);

        return;
    }

    // L2: If ||c||^2 is not pre-computed, calculate it
    DeviceTensor<float, 1, true> cNorms;
    if (computeL2 && !centroidNorms) {
        cNorms = DeviceTensor<float, 1, true>(
                res, makeTempAlloc(AllocType::Other, stream), {numCentroids});
        runL2Norm(centroids, centroidsRowMajor, cNorms, true, stream);
        centroidNorms = &cNorms;
    }

    //
    // Prepare norm vector ||q||^2; ||c||^2 is already pre-computed
    //
    DeviceTensor<float, 1, true> queryNorms(
            res, makeTempAlloc(AllocType::Other, stream), {numQueries});

    // ||q||^2
    if (computeL2) {
        runL2Norm(queries, queriesRowMajor, queryNorms, true, stream);
    }

    // By default, aim to use up to 512 MB of memory for the processing, with
    // both number of queries and number of centroids being at least 512.
    idx_t tileRows = 0;
    idx_t tileCols = 0;
    chooseTileSize(
            numQueries,
            numCentroids,
            dim,
            sizeof(T),
            res->getTempMemoryAvailableCurrentDevice(),
            tileRows,
            tileCols);

    idx_t numColTiles = utils::divUp(numCentroids, tileCols);

    // We can have any number of vectors to query against, even less than k, in
    // which case we'll return -1 for the index
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation

    // Temporary output memory space we'll use
    DeviceTensor<float, 2, true> distanceBuf1(
            res, makeTempAlloc(AllocType::Other, stream), {tileRows, tileCols});
    DeviceTensor<float, 2, true> distanceBuf2(
            res, makeTempAlloc(AllocType::Other, stream), {tileRows, tileCols});
    DeviceTensor<float, 2, true>* distanceBufs[2] = {
            &distanceBuf1, &distanceBuf2};

    DeviceTensor<float, 2, true> outDistanceBuf1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {tileRows, numColTiles * k});
    DeviceTensor<float, 2, true> outDistanceBuf2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {tileRows, numColTiles * k});
    DeviceTensor<float, 2, true>* outDistanceBufs[2] = {
            &outDistanceBuf1, &outDistanceBuf2};

    DeviceTensor<idx_t, 2, true> outIndexBuf1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {tileRows, numColTiles * k});
    DeviceTensor<idx_t, 2, true> outIndexBuf2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {tileRows, numColTiles * k});
    DeviceTensor<idx_t, 2, true>* outIndexBufs[2] = {
            &outIndexBuf1, &outIndexBuf2};

    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    int curStream = 0;
    bool interrupt = false;

    // Tile over the input queries
    for (idx_t i = 0; i < numQueries; i += tileRows) {
        if (interrupt || InterruptCallback::is_interrupted()) {
            interrupt = true;
            break;
        }

        idx_t curQuerySize = std::min(tileRows, numQueries - i);

        auto outDistanceView = outDistances.narrow(0, i, curQuerySize);
        auto outIndexView = outIndices.narrow(0, i, curQuerySize);

        auto queryView =
                queries.narrow(queriesRowMajor ? 0 : 1, i, curQuerySize);
        auto queryNormNiew = queryNorms.narrow(0, i, curQuerySize);

        auto outDistanceBufRowView =
                outDistanceBufs[curStream]->narrow(0, 0, curQuerySize);
        auto outIndexBufRowView =
                outIndexBufs[curStream]->narrow(0, 0, curQuerySize);

        // Tile over the centroids
        for (idx_t j = 0; j < numCentroids; j += tileCols) {
            if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
            }

            auto curCentroidSize = std::min(tileCols, numCentroids - j);
            auto curColTile = j / tileCols;

            auto centroidsView = sliceCentroids(
                    centroids, centroidsRowMajor, j, curCentroidSize);

            auto distanceBufView = distanceBufs[curStream]
                                           ->narrow(0, 0, curQuerySize)
                                           .narrow(1, 0, curCentroidSize);

            auto outDistanceBufColView =
                    outDistanceBufRowView.narrow(1, k * curColTile, k);
            auto outIndexBufColView =
                    outIndexBufRowView.narrow(1, k * curColTile, k);

            // L2: distance is ||c||^2 - 2qc + ||q||^2, we compute -2qc
            // IP: just compute qc
            // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
            runMatrixMult(
                    distanceBufView,
                    false, // not transposed
                    queryView,
                    !queriesRowMajor, // transposed MM if col major
                    centroidsView,
                    centroidsRowMajor, // transposed MM if row major
                    computeL2 ? -2.0f : 1.0f,
                    0.0f,
                    res->getBlasHandleCurrentDevice(),
                    streams[curStream]);

            if (computeL2) {
                // For L2 distance, we use this fused kernel that performs both
                // adding ||c||^2 to -2qc and k-selection, so we only need two
                // passes (one write by the gemm, one read here) over the huge
                // region of output memory
                //
                // If we aren't tiling along the number of centroids, we can
                // perform the output work directly
                if (tileCols == numCentroids) {
                    // Write into the final output
                    runL2SelectMin(
                            distanceBufView,
                            *centroidNorms,
                            outDistanceView,
                            outIndexView,
                            k,
                            streams[curStream]);

                    if (!ignoreOutDistances) {
                        // expand (query id) to (query id, k) by duplicating
                        // along rows top-k ||c||^2 - 2qc + ||q||^2 in the form
                        // (query id, k)
                        runSumAlongRows(
                                queryNormNiew,
                                outDistanceView,
                                true, // L2 distances should not go below zero
                                      // due to roundoff error
                                streams[curStream]);
                    }
                } else {
                    auto centroidNormsView =
                            centroidNorms->narrow(0, j, curCentroidSize);

                    // Write into our intermediate output
                    runL2SelectMin(
                            distanceBufView,
                            centroidNormsView,
                            outDistanceBufColView,
                            outIndexBufColView,
                            k,
                            streams[curStream]);

                    if (!ignoreOutDistances) {
                        // expand (query id) to (query id, k) by duplicating
                        // along rows top-k ||c||^2 - 2qc + ||q||^2 in the form
                        // (query id, k)
                        runSumAlongRows(
                                queryNormNiew,
                                outDistanceBufColView,
                                true, // L2 distances should not go below zero
                                      // due to roundoff error
                                streams[curStream]);
                    }
                }
            } else {
                // For IP, just k-select the output for this tile
                if (tileCols == numCentroids) {
                    // Write into the final output
                    runBlockSelect(
                            distanceBufView,
                            outDistanceView,
                            outIndexView,
                            true,
                            k,
                            streams[curStream]);
                } else {
                    // Write into the intermediate output
                    runBlockSelect(
                            distanceBufView,
                            outDistanceBufColView,
                            outIndexBufColView,
                            true,
                            k,
                            streams[curStream]);
                }
            }
        }

        // As we're finished with processing a full set of centroids, perform
        // the final k-selection
        if (tileCols != numCentroids) {
            // The indices are tile-relative; for each tile of k, we need to add
            // tileCols to the index
            runIncrementIndex(
                    outIndexBufRowView, k, tileCols, streams[curStream]);

            runBlockSelectPair(
                    outDistanceBufRowView,
                    outIndexBufRowView,
                    outDistanceView,
                    outIndexView,
                    computeL2 ? false : true,
                    k,
                    streams[curStream]);
        }

        curStream = (curStream + 1) % 2;
    }

    // Have the desired ordering stream wait on the multi-stream
    streamWait({stream}, streams);

    if (interrupt) {
        FAISS_THROW_MSG("interrupted");
    }
}

template <typename T>
void runL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<T, 2, true>& centroids,
        bool centroidsRowMajor,
        Tensor<float, 1, true>* centroidNorms,
        Tensor<T, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool ignoreOutDistances = false) {
    runDistance<T>(
            true, // L2
            res,
            stream,
            centroids,
            centroidsRowMajor,
            centroidNorms,
            queries,
            queriesRowMajor,
            k,
            outDistances,
            outIndices,
            ignoreOutDistances);
}

template <typename T>
void runIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<T, 2, true>& centroids,
        bool centroidsRowMajor,
        Tensor<T, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    runDistance<T>(
            false, // IP
            res,
            stream,
            centroids,
            centroidsRowMajor,
            nullptr, // no centroid norms provided
            queries,
            queriesRowMajor,
            k,
            outDistances,
            outIndices,
            false);
}

//
// Instantiations of the distance templates
//

void runAllPairwiseL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<float, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances) {
    runAllPairwiseDistance<float>(
            true,
            res,
            stream,
            vectors,
            vectorsRowMajor,
            vectorNorms,
            queries,
            queriesRowMajor,
            outDistances);
}

void runAllPairwiseL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<half, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances) {
    runAllPairwiseDistance<half>(
            true,
            res,
            stream,
            vectors,
            vectorsRowMajor,
            vectorNorms,
            queries,
            queriesRowMajor,
            outDistances);
}

void runAllPairwiseL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<__nv_bfloat16, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<__nv_bfloat16, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances) {
    runAllPairwiseDistance<__nv_bfloat16>(
            true,
            res,
            stream,
            vectors,
            vectorsRowMajor,
            vectorNorms,
            queries,
            queriesRowMajor,
            outDistances);
}

void runAllPairwiseIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances) {
    runAllPairwiseDistance<float>(
            false,
            res,
            stream,
            vectors,
            vectorsRowMajor,
            nullptr,
            queries,
            queriesRowMajor,
            outDistances);
}

void runAllPairwiseIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<half, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances) {
    runAllPairwiseDistance<half>(
            false,
            res,
            stream,
            vectors,
            vectorsRowMajor,
            nullptr,
            queries,
            queriesRowMajor,
            outDistances);
}

void runAllPairwiseIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<__nv_bfloat16, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<__nv_bfloat16, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances) {
    runAllPairwiseDistance<__nv_bfloat16>(
            false,
            res,
            stream,
            vectors,
            vectorsRowMajor,
            nullptr,
            queries,
            queriesRowMajor,
            outDistances);
}

void runL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<float, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool ignoreOutDistances) {
    runL2Distance<float>(
            res,
            stream,
            vectors,
            vectorsRowMajor,
            vectorNorms,
            queries,
            queriesRowMajor,
            k,
            outDistances,
            outIndices,
            ignoreOutDistances);
}

void runL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<half, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool ignoreOutDistances) {
    runL2Distance<half>(
            res,
            stream,
            vectors,
            vectorsRowMajor,
            vectorNorms,
            queries,
            queriesRowMajor,
            k,
            outDistances,
            outIndices,
            ignoreOutDistances);
}

void runL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<__nv_bfloat16, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<__nv_bfloat16, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool ignoreOutDistances) {
    runL2Distance<__nv_bfloat16>(
            res,
            stream,
            vectors,
            vectorsRowMajor,
            vectorNorms,
            queries,
            queriesRowMajor,
            k,
            outDistances,
            outIndices,
            ignoreOutDistances);
}

void runIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    runIPDistance<float>(
            res,
            stream,
            vectors,
            vectorsRowMajor,
            queries,
            queriesRowMajor,
            k,
            outDistances,
            outIndices);
}

void runIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<half, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    runIPDistance<half>(
            res,
            stream,
            vectors,
            vectorsRowMajor,
            queries,
            queriesRowMajor,
            k,
            outDistances,
            outIndices);
}

void runIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<__nv_bfloat16, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<__nv_bfloat16, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    runIPDistance<__nv_bfloat16>(
            res,
            stream,
            vectors,
            vectorsRowMajor,
            queries,
            queriesRowMajor,
            k,
            outDistances,
            outIndices);
}

} // namespace gpu
} // namespace faiss
