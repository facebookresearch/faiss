/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/gpu/utils/ThrustUtils.cuh>

#include <algorithm>

namespace faiss {
namespace gpu {

size_t getIVFKSelectionPass2Chunks(size_t nprobe) {
    // We run two passes of heap selection
    // This is the size of the second-level heap passes
    constexpr size_t kNProbeSplit = 8;
    return std::min(nprobe, kNProbeSplit);
}

size_t getIVFPerQueryTempMemory(size_t k, size_t nprobe, size_t maxListLength) {
    size_t pass2Chunks = getIVFKSelectionPass2Chunks(nprobe);

    size_t sizeForFirstSelectPass =
            pass2Chunks * k * (sizeof(float) + sizeof(idx_t));

    // Each IVF list being scanned concurrently needs a separate array to
    // indicate where the per-IVF list distances are being stored via prefix
    // sum. There is one per each nprobe, plus 1 more entry at the end
    size_t prefixSumOffsets = nprobe * sizeof(idx_t) + sizeof(idx_t);

    // Storage for all distances from all the IVF lists we are processing
    size_t allDistances = nprobe * maxListLength * sizeof(float);

    // There are 2 streams on which computations is performed (hence the 2 *)
    return 2 * (prefixSumOffsets + allDistances + sizeForFirstSelectPass);
}

size_t getIVFPQPerQueryTempMemory(
        size_t k,
        size_t nprobe,
        size_t maxListLength,
        bool usePrecomputedCodes,
        size_t numSubQuantizers,
        size_t numSubQuantizerCodes) {
    // Residual PQ distances per each IVF partition (in case we are not using
    // precomputed codes;
    size_t residualDistances = usePrecomputedCodes
            ? 0
            : (nprobe * numSubQuantizers * numSubQuantizerCodes *
               sizeof(float));

    // There are 2 streams on which computations is performed (hence the 2 *)
    // The IVF-generic temp memory allocation already takes this multi-streaming
    // into account, but we need to do so for the PQ residual distances too
    return (2 * residualDistances) +
            getIVFPerQueryTempMemory(k, nprobe, maxListLength);
}

size_t getIVFQueryTileSize(
        size_t numQueries,
        size_t tempMemoryAvailable,
        size_t sizePerQuery) {
    // Our ideal minimum number of queries that we'd like to run concurrently
    constexpr size_t kMinQueryTileSize = 8;

    // Our absolute maximum number of queries that we can run concurrently
    // (based on max Y grid dimension)
    constexpr size_t kMaxQueryTileSize = 65536;

    // First, see how many queries we can run within the limit of our available
    // temporary memory. If all queries can run within the temporary memory
    // limit, we'll just use that.
    size_t withinTempMemoryNumQueries =
            std::min(tempMemoryAvailable / sizePerQuery, numQueries);

    // However, there is a maximum cap on the number of queries that we can run
    // at once, even if memory were unlimited (due to max Y grid dimension)
    withinTempMemoryNumQueries =
            std::min(withinTempMemoryNumQueries, kMaxQueryTileSize);

    // However. withinTempMemoryNumQueries could be really small, or even zero
    // (in the case where there is no temporary memory available, or the memory
    // resources for a single query required are really large). If we are below
    // the ideal minimum number of queries to run concurrently, then we will
    // ignore the temporary memory limit and fall back to a general device
    // allocation.
    // Note that if we only had a single query, then this is ok to run as-is
    if (withinTempMemoryNumQueries < numQueries &&
        withinTempMemoryNumQueries < kMinQueryTileSize) {
        // Either the amount of temporary memory available is too low, or the
        // amount of memory needed to run a single query is really high. Ignore
        // the temporary memory available, and always attempt to use this amount
        // of memory for temporary results
        //
        // FIXME: could look at amount of memory available on the current
        // device, but there is no guarantee that all that memory available
        // could be done in a single allocation, so we just pick a suitably
        // large allocation that can yield enough efficiency but something that
        // the GPU can likely allocate.
        constexpr size_t kMinMemoryAllocation = 512 * 1024 * 1024; // 512 MiB

        size_t withinMemoryNumQueries =
                std::min(kMinMemoryAllocation / sizePerQuery, numQueries);

        // It is possible that the per-query size is incredibly huge, in which
        // case even the 512 MiB allocation will not fit it. In this case, we
        // have no option except to try running a single one.
        return std::max(withinMemoryNumQueries, size_t(1));
    } else {
        // withinTempMemoryNumQueries cannot be > numQueries.
        // Either:
        // 1. == numQueries, >= kMinQueryTileSize (i.e., we can satisfy all
        // queries in one go, or are limited by max query tile size)
        // 2. < numQueries, >= kMinQueryTileSize (i.e., we can't satisfy all
        // queries in one go, but we have a large enough batch to run which is
        // ok
        return withinTempMemoryNumQueries;
    }
}

// Calculates the total number of intermediate distances to consider
// for all queries
__global__ void getResultLengths(
        Tensor<idx_t, 2, true> ivfListIds,
        idx_t* listLengths,
        idx_t totalSize,
        Tensor<idx_t, 2, true> length) {
    idx_t linearThreadId = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearThreadId >= totalSize) {
        return;
    }

    auto nprobe = ivfListIds.getSize(1);
    auto queryId = linearThreadId / nprobe;
    auto listId = linearThreadId % nprobe;

    idx_t centroidId = ivfListIds[queryId][listId];

    // Safety guard in case NaNs in input cause no list ID to be generated
    length[queryId][listId] = (centroidId != -1) ? listLengths[centroidId] : 0;
}

void runCalcListOffsets(
        GpuResources* res,
        Tensor<idx_t, 2, true>& ivfListIds,
        DeviceVector<idx_t>& listLengths,
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<char, 1, true>& thrustMem,
        cudaStream_t stream) {
    FAISS_ASSERT(ivfListIds.getSize(0) == prefixSumOffsets.getSize(0));
    FAISS_ASSERT(ivfListIds.getSize(1) == prefixSumOffsets.getSize(1));

    idx_t totalSize = ivfListIds.numElements();

    idx_t numThreads = std::min(totalSize, (idx_t)getMaxThreadsCurrentDevice());
    idx_t numBlocks = utils::divUp(totalSize, numThreads);

    auto grid = dim3(numBlocks);
    auto block = dim3(numThreads);

    getResultLengths<<<grid, block, 0, stream>>>(
            ivfListIds, listLengths.data(), totalSize, prefixSumOffsets);
    CUDA_TEST_ERROR();

    // Prefix sum of the indices, so we know where the intermediate
    // results should be maintained
    // Thrust wants a place for its temporary allocations, so provide
    // one, so it won't call cudaMalloc/Free if we size it sufficiently
    ThrustAllocator alloc(
            res, stream, thrustMem.data(), thrustMem.getSizeInBytes());

    thrust::inclusive_scan(
            thrust::cuda::par(alloc).on(stream),
            prefixSumOffsets.data(),
            prefixSumOffsets.data() + totalSize,
            prefixSumOffsets.data());
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
