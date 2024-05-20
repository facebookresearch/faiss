/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/impl/scan/IVFInterleavedImpl.cuh>

namespace faiss {
namespace gpu {

constexpr uint32_t kMaxUInt32 = std::numeric_limits<uint32_t>::max();

// Second-pass kernel to further k-select the results from the first pass across
// IVF lists and produce the final results
template <int ThreadsPerBlock, int NumWarpQ, int NumThreadQ>
__global__ void ivfInterleavedScan2(
        Tensor<float, 3, true> distanceIn,
        Tensor<idx_t, 3, true> indicesIn,
        Tensor<idx_t, 2, true> listIds,
        int k,
        void** listIndices,
        IndicesOptions opt,
        bool dir,
        Tensor<float, 2, true> distanceOut,
        Tensor<idx_t, 2, true> indicesOut) {
    if constexpr ((NumWarpQ == 1 && NumThreadQ == 1) || NumWarpQ >= kWarpSize) {
        int queryId = blockIdx.x;

        constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

        __shared__ float smemK[kNumWarps * NumWarpQ];
        // The BlockSelect value type is uint32_t, as we pack together which
        // probe (up to nprobe - 1) and which k (up to k - 1) from each
        // individual list together, and both nprobe and k are limited to
        // GPU_MAX_SELECTION_K.
        __shared__ uint32_t smemV[kNumWarps * NumWarpQ];

        // To avoid creating excessive specializations, we combine direction
        // kernels, selecting for the smallest element. If `dir` is true, we
        // negate all values being selected (so that we are selecting the
        // largest element).
        BlockSelect<
                float,
                uint32_t,
                false,
                Comparator<float>,
                NumWarpQ,
                NumThreadQ,
                ThreadsPerBlock>
                heap(kFloatMax, kMaxUInt32, smemK, smemV, k);

        // nprobe x k
        idx_t num = distanceIn.getSize(1) * distanceIn.getSize(2);

        const float* distanceBase = distanceIn[queryId].data();
        idx_t limit = utils::roundDown(num, kWarpSize);

        // This will keep our negation factor
        float adj = dir ? -1 : 1;

        idx_t i = threadIdx.x;
        for (; i < limit; i += blockDim.x) {
            // We represent the index as (probe id)(k)
            // Right now, both are limited to a maximum of 2048, but we will
            // dedicate each to the high and low words of a uint32_t
            static_assert(GPU_MAX_SELECTION_K <= 65536, "");

            uint32_t curProbe = i / k;
            uint32_t curK = i % k;
            // Since nprobe and k are limited, we can pack both of these
            // together into a uint32_t
            uint32_t index = (curProbe << 16) | (curK & (uint32_t)0xffff);

            // The IDs reported from the list may be -1, if a particular IVF
            // list doesn't even have k entries in it
            if (listIds[queryId][curProbe] != -1) {
                // Adjust the value we are selecting based on the sorting order
                heap.addThreadQ(distanceBase[i] * adj, index);
            }

            heap.checkThreadQ();
        }

        // Handle warp divergence separately
        if (i < num) {
            uint32_t curProbe = i / k;
            uint32_t curK = i % k;
            uint32_t index = (curProbe << 16) | (curK & (uint32_t)0xffff);

            idx_t listId = listIds[queryId][curProbe];
            if (listId != -1) {
                heap.addThreadQ(distanceBase[i] * adj, index);
            }
        }

        // Merge all final results
        heap.reduce();

        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            // Re-adjust the value we are selecting based on the sorting order
            distanceOut[queryId][i] = smemK[i] * adj;
            auto packedIndex = smemV[i];

            // We need to remap to the user-provided indices
            idx_t index = -1;

            // We may not have at least k values to return; in this function,
            // max uint32 is our sentinel value
            if (packedIndex != kMaxUInt32) {
                uint32_t curProbe = packedIndex >> 16;
                uint32_t curK = packedIndex & 0xffff;

                idx_t listId = listIds[queryId][curProbe];
                idx_t listOffset = indicesIn[queryId][curProbe][curK];

                if (opt == INDICES_32_BIT) {
                    index = (idx_t)((int*)listIndices[listId])[listOffset];
                } else if (opt == INDICES_64_BIT) {
                    index = ((idx_t*)listIndices[listId])[listOffset];
                } else {
                    index = (listId << 32 | (idx_t)listOffset);
                }
            }

            indicesOut[queryId][i] = index;
        }
    }
}

void runIVFInterleavedScan2(
        Tensor<float, 3, true>& distanceIn,
        Tensor<idx_t, 3, true>& indicesIn,
        Tensor<idx_t, 2, true>& listIds,
        int k,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        bool dir,
        Tensor<float, 2, true>& distanceOut,
        Tensor<idx_t, 2, true>& indicesOut,
        cudaStream_t stream) {
#define IVF_SCAN_2(THREADS, NUM_WARP_Q, NUM_THREAD_Q)        \
    ivfInterleavedScan2<THREADS, NUM_WARP_Q, NUM_THREAD_Q>   \
            <<<distanceIn.getSize(0), THREADS, 0, stream>>>( \
                    distanceIn,                              \
                    indicesIn,                               \
                    listIds,                                 \
                    k,                                       \
                    listIndices.data(),                      \
                    indicesOptions,                          \
                    dir,                                     \
                    distanceOut,                             \
                    indicesOut)

    if (k == 1) {
        IVF_SCAN_2(128, 1, 1);
    } else if (k <= 32 && getWarpSizeCurrentDevice() == 32) {
        IVF_SCAN_2(128, 32, 2);
    } else if (k <= 64) {
        IVF_SCAN_2(128, 64, 3);
    } else if (k <= 128) {
        IVF_SCAN_2(128, 128, 3);
    } else if (k <= 256) {
        IVF_SCAN_2(128, 256, 4);
    } else if (k <= 512) {
        IVF_SCAN_2(128, 512, 8);
    } else if (k <= 1024) {
        IVF_SCAN_2(128, 1024, 8);
    }
#if GPU_MAX_SELECTION_K >= 2048
    else if (k <= 2048) {
        IVF_SCAN_2(64, 2048, 8);
    }
#endif
}

void runIVFInterleavedScan(
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        int k,
        faiss::MetricType metric,
        bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res) {
    // caught for exceptions at a higher level
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    const auto ivf_interleaved_call = [&](const auto func) {
        func(queries,
             listIds,
             listData,
             listIndices,
             indicesOptions,
             listLengths,
             k,
             metric,
             useResidual,
             residualBase,
             scalarQ,
             outDistances,
             outIndices,
             res);
    };

    if (k == 1) {
        ivf_interleaved_call(ivfInterleavedScanImpl<128, 1, 1>);
    } else if (k <= 32 && getWarpSizeCurrentDevice() == 32) {
        ivf_interleaved_call(ivfInterleavedScanImpl<128, 32, 2>);
    } else if (k <= 64) {
        ivf_interleaved_call(ivfInterleavedScanImpl<128, 64, 3>);
    } else if (k <= 128) {
        ivf_interleaved_call(ivfInterleavedScanImpl<128, 128, 3>);
    } else if (k <= 256) {
        ivf_interleaved_call(ivfInterleavedScanImpl<128, 256, 4>);
    } else if (k <= 512) {
        ivf_interleaved_call(ivfInterleavedScanImpl<128, 512, 8>);
    } else if (k <= 1024) {
        ivf_interleaved_call(ivfInterleavedScanImpl<128, 1024, 8>);
    }
#if GPU_MAX_SELECTION_K >= 2048
    else if (k <= 2048) {
        ivf_interleaved_call(ivfInterleavedScanImpl<64, 2048, 8>);
    }
#endif
}

} // namespace gpu
} // namespace faiss
