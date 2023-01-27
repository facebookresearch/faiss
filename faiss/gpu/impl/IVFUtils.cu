/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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

// Calculates the total number of intermediate distances to consider
// for all queries
__global__ void getResultLengths(
        Tensor<idx_t, 2, true> ivfListIds,
        int* listLengths,
        int totalSize,
        Tensor<int, 2, true> length) {
    int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearThreadId >= totalSize) {
        return;
    }

    int nprobe = ivfListIds.getSize(1);
    int queryId = linearThreadId / nprobe;
    int listId = linearThreadId % nprobe;

    idx_t centroidId = ivfListIds[queryId][listId];

    // Safety guard in case NaNs in input cause no list ID to be generated
    length[queryId][listId] = (centroidId != -1) ? listLengths[centroidId] : 0;
}

void runCalcListOffsets(
        GpuResources* res,
        Tensor<idx_t, 2, true>& ivfListIds,
        DeviceVector<int>& listLengths,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<char, 1, true>& thrustMem,
        cudaStream_t stream) {
    FAISS_ASSERT(ivfListIds.getSize(0) == prefixSumOffsets.getSize(0));
    FAISS_ASSERT(ivfListIds.getSize(1) == prefixSumOffsets.getSize(1));

    int totalSize = ivfListIds.numElements();

    int numThreads = std::min(totalSize, getMaxThreadsCurrentDevice());
    int numBlocks = utils::divUp(totalSize, numThreads);

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
