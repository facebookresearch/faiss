/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

//
// This kernel is split into a separate compilation unit to cut down
// on compile time
//

namespace faiss {
namespace gpu {

// This is warp divergence central, but this is really a final step
// and happening a small number of times
template <typename T>
__device__ int binarySearchForBucket(T* prefixSumOffsets, T size, T val) {
    T start = 0;
    T end = size;

    while (end - start > 0) {
        T mid = start + (end - start) / 2;
        T midVal = prefixSumOffsets[mid];

        // Find the first bucket that we are <=
        if (midVal <= val) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    // We must find the bucket that it is in
    assert(start != size);

    return start;
}

template <
        typename IndexT,
        int ThreadsPerBlock,
        int NumWarpQ,
        int NumThreadQ,
        bool Dir>
__global__ void pass2SelectLists(
        Tensor<float, 2, true> heapDistances,
        Tensor<idx_t, 2, true> heapIndices,
        void** listIndices,
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<idx_t, 2, true> ivfListIds,
        int k,
        IndicesOptions opt,
        Tensor<float, 2, true> outDistances,
        Tensor<idx_t, 2, true> outIndices) {
    if constexpr ((NumWarpQ == 1 && NumThreadQ == 1) || NumWarpQ >= kWarpSize) {
        constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

        __shared__ float smemK[kNumWarps * NumWarpQ];
        __shared__ IndexT smemV[kNumWarps * NumWarpQ];

        constexpr auto kInit = Dir ? kFloatMin : kFloatMax;
        BlockSelect<
                float,
                IndexT,
                Dir,
                Comparator<float>,
                NumWarpQ,
                NumThreadQ,
                ThreadsPerBlock>
                heap(kInit, -1, smemK, smemV, k);

        auto queryId = blockIdx.x;
        idx_t num = heapDistances.getSize(1);
        idx_t limit = utils::roundDown(num, kWarpSize);

        idx_t i = threadIdx.x;
        auto heapDistanceStart = heapDistances[queryId];

        // BlockSelect add cannot be used in a warp divergent circumstance; we
        // handle the remainder warp below
        for (; i < limit; i += blockDim.x) {
            heap.add(heapDistanceStart[i], IndexT(i));
        }

        // Handle warp divergence separately
        if (i < num) {
            heap.addThreadQ(heapDistanceStart[i], IndexT(i));
        }

        // Merge all final results
        heap.reduce();

        for (auto i = threadIdx.x; i < k; i += blockDim.x) {
            outDistances[queryId][i] = smemK[i];

            // `v` is the index in `heapIndices`
            // We need to translate this into an original user index. The
            // reason why we don't maintain intermediate results in terms of
            // user indices is to substantially reduce temporary memory
            // requirements and global memory write traffic for the list
            // scanning.
            // This code is highly divergent, but it's probably ok, since this
            // is the very last step and it is happening a small number of
            // times (#queries x k).
            idx_t v = smemV[i];
            idx_t index = -1;

            if (v != -1) {
                // `offset` is the offset of the intermediate result, as
                // calculated by the original scan.
                idx_t offset = heapIndices[queryId][v];

                // In order to determine the actual user index, we need to first
                // determine what list it was in.
                // We do this by binary search in the prefix sum list.
                idx_t probe = binarySearchForBucket(
                        prefixSumOffsets[queryId].data(),
                        prefixSumOffsets.getSize(1),
                        offset);

                // This is then the probe for the query; we can find the actual
                // list ID from this
                idx_t listId = ivfListIds[queryId][probe];

                // Now, we need to know the offset within the list
                // We ensure that before the array (at offset -1), there is a 0
                // value
                idx_t listStart =
                        *(prefixSumOffsets[queryId][probe].data() - 1);
                idx_t listOffset = offset - listStart;

                // This gives us our final index
                if (opt == INDICES_32_BIT) {
                    index = (idx_t)((int*)listIndices[listId])[listOffset];
                } else if (opt == INDICES_64_BIT) {
                    index = ((idx_t*)listIndices[listId])[listOffset];
                } else {
                    index = (listId << 32 | (idx_t)listOffset);
                }
            }

            outIndices[queryId][i] = index;
        }
    }
}

void runPass2SelectLists(
        Tensor<float, 2, true>& heapDistances,
        Tensor<idx_t, 2, true>& heapIndices,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<idx_t, 2, true>& ivfListIds,
        int k,
        bool use64BitSelection,
        bool chooseLargest,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        cudaStream_t stream) {
    // This is also caught at a higher level
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    auto grid = dim3(ivfListIds.getSize(0));

#define RUN_PASS(INDEX_T, BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR)         \
    do {                                                                \
        pass2SelectLists<INDEX_T, BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR> \
                <<<grid, BLOCK, 0, stream>>>(                           \
                        heapDistances,                                  \
                        heapIndices,                                    \
                        listIndices.data(),                             \
                        prefixSumOffsets,                               \
                        ivfListIds,                                     \
                        k,                                              \
                        indicesOptions,                                 \
                        outDistances,                                   \
                        outIndices);                                    \
    } while (0)

#if GPU_MAX_SELECTION_K >= 2048

    // block size 128 for k <= 1024, 64 for k = 2048
#define RUN_PASS_DIR(INDEX_T, DIR)                                \
    do {                                                          \
        if (k == 1) {                                             \
            RUN_PASS(INDEX_T, 128, 1, 1, DIR);                    \
        } else if (k <= 32 && getWarpSizeCurrentDevice() == 32) { \
            RUN_PASS(INDEX_T, 128, 32, 2, DIR);                   \
        } else if (k <= 64) {                                     \
            RUN_PASS(INDEX_T, 128, 64, 3, DIR);                   \
        } else if (k <= 128) {                                    \
            RUN_PASS(INDEX_T, 128, 128, 3, DIR);                  \
        } else if (k <= 256) {                                    \
            RUN_PASS(INDEX_T, 128, 256, 4, DIR);                  \
        } else if (k <= 512) {                                    \
            RUN_PASS(INDEX_T, 128, 512, 8, DIR);                  \
        } else if (k <= 1024) {                                   \
            RUN_PASS(INDEX_T, 128, 1024, 8, DIR);                 \
        } else if (k <= 2048) {                                   \
            RUN_PASS(INDEX_T, 64, 2048, 8, DIR);                  \
        }                                                         \
    } while (0)

#else

#define RUN_PASS_DIR(INDEX_T, DIR)                                \
    do {                                                          \
        if (k == 1) {                                             \
            RUN_PASS(INDEX_T, 128, 1, 1, DIR);                    \
        } else if (k <= 32 && getWarpSizeCurrentDevice() == 32) { \
            RUN_PASS(INDEX_T, 128, 32, 2, DIR);                   \
        } else if (k <= 64) {                                     \
            RUN_PASS(INDEX_T, 128, 64, 3, DIR);                   \
        } else if (k <= 128) {                                    \
            RUN_PASS(INDEX_T, 128, 128, 3, DIR);                  \
        } else if (k <= 256) {                                    \
            RUN_PASS(INDEX_T, 128, 256, 4, DIR);                  \
        } else if (k <= 512) {                                    \
            RUN_PASS(INDEX_T, 128, 512, 8, DIR);                  \
        } else if (k <= 1024) {                                   \
            RUN_PASS(INDEX_T, 128, 1024, 8, DIR);                 \
        }                                                         \
    } while (0)

#endif // GPU_MAX_SELECTION_K

    if (use64BitSelection) {
        if (chooseLargest) {
            RUN_PASS_DIR(idx_t, true);
        } else {
            RUN_PASS_DIR(idx_t, false);
        }
    } else {
        if (chooseLargest) {
            RUN_PASS_DIR(int32_t, true);
        } else {
            RUN_PASS_DIR(int32_t, false);
        }
    }

#undef RUN_PASS_DIR
#undef RUN_PASS

    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
