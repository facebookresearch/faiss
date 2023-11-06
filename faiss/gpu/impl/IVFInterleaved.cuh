/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/utils/Comparators.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/WarpPackedBits.cuh>

namespace faiss {
namespace gpu {

/// First pass kernel to perform scanning of IVF lists to produce top-k
/// candidates
template <
        typename Codec,
        typename Metric,
        int ThreadsPerBlock,
        int NumWarpQ,
        int NumThreadQ>
__global__ void ivfInterleavedScan(
        Tensor<float, 2, true> queries,
        Tensor<float, 3, true> residualBase,
        Tensor<idx_t, 2, true> listIds,
        void** allListData,
        idx_t* listLengths,
        Codec codec,
        Metric metric,
        int k,
        // [query][probe][k]
        Tensor<float, 3, true> distanceOut,
        Tensor<idx_t, 3, true> indicesOut,
        const bool Residual) {
    if constexpr ((NumWarpQ == 1 && NumThreadQ == 1) || NumWarpQ >= kWarpSize) {
        extern __shared__ float smem[];

        constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

        for (idx_t queryId = blockIdx.y; queryId < queries.getSize(0);
             queryId += gridDim.y) {
            int probeId = blockIdx.x;
            idx_t listId = listIds[queryId][probeId];

            // Safety guard in case NaNs in input cause no list ID to be
            // generated, or we have more nprobe than nlist
            if (listId == -1) {
                return;
            }

            // Vector dimension is currently limited to 32 bit
            int dim = queries.getSize(1);

            // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
            int laneId = threadIdx.x % kWarpSize;
            int warpId = threadIdx.x / kWarpSize;

            using EncodeT = typename Codec::EncodeT;

            auto query = queries[queryId].data();
            auto vecsBase = (EncodeT*)allListData[listId];
            int numVecs = listLengths[listId];
            auto residualBaseSlice = residualBase[queryId][probeId].data();

            constexpr auto kInit = Metric::kDirection ? kFloatMin : kFloatMax;

            __shared__ float smemK[kNumWarps * NumWarpQ];
            __shared__ idx_t smemV[kNumWarps * NumWarpQ];

            BlockSelect<
                    float,
                    idx_t,
                    Metric::kDirection,
                    Comparator<float>,
                    NumWarpQ,
                    NumThreadQ,
                    ThreadsPerBlock>
                    heap(kInit, -1, smemK, smemV, k);

            // The codec might be dependent upon data that we need to reference
            // or store in shared memory
            codec.initKernel(smem, dim);
            __syncthreads();

            // How many vector blocks of kWarpSize are in this list?
            idx_t numBlocks = utils::divUp(numVecs, (idx_t)kWarpSize);

            // Number of EncodeT words per each dimension of block of kWarpSize
            // vecs
            constexpr int bytesPerVectorBlockDim =
                    Codec::kEncodeBits * kWarpSize / 8;
            constexpr int wordsPerVectorBlockDim =
                    bytesPerVectorBlockDim / sizeof(EncodeT);
            int wordsPerVectorBlock = wordsPerVectorBlockDim * dim;

            int dimBlocks = utils::roundDown(dim, kWarpSize);

            for (idx_t block = warpId; block < numBlocks; block += kNumWarps) {
                // We're handling a new vector
                Metric dist = metric.zero();

                // This is the vector a given lane/thread handles
                idx_t vec = block * kWarpSize + laneId;
                bool valid = vec < numVecs;

                // This is where this warp begins reading data
                EncodeT* data = vecsBase + block * wordsPerVectorBlock;

                // whole blocks
                for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
                    const int loadDim = dBase + laneId;
                    const float queryReg = query[loadDim];
                    const float residualReg =
                            Residual ? residualBaseSlice[loadDim] : 0;

                    constexpr int kUnroll = 4;

#pragma unroll
                    for (int i = 0; i < kWarpSize / kUnroll;
                         ++i, data += kUnroll * wordsPerVectorBlockDim) {
                        EncodeT encV[kUnroll];
#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            encV[j] = WarpPackedBits<
                                    EncodeT,
                                    Codec::kEncodeBits>::
                                    read(laneId,
                                         data + j * wordsPerVectorBlockDim);
                        }

#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            encV[j] = WarpPackedBits<
                                    EncodeT,
                                    Codec::kEncodeBits>::
                                    postRead(laneId, encV[j]);
                        }

                        float decV[kUnroll];
#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            int d = i * kUnroll + j;
                            decV[j] = codec.decodeNew(dBase + d, encV[j]);
                        }

                        if (Residual) {
#pragma unroll
                            for (int j = 0; j < kUnroll; ++j) {
                                int d = i * kUnroll + j;
                                decV[j] += SHFL_SYNC(residualReg, d, kWarpSize);
                            }
                        }

#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            int d = i * kUnroll + j;
                            float q = SHFL_SYNC(queryReg, d, kWarpSize);
                            dist.handle(q, decV[j]);
                        }
                    }
                }

                // remainder
                const int loadDim = dimBlocks + laneId;
                const bool loadDimInBounds = loadDim < dim;

                const float queryReg = loadDimInBounds ? query[loadDim] : 0;
                const float residualReg = Residual && loadDimInBounds
                        ? residualBaseSlice[loadDim]
                        : 0;

                for (int d = 0; d < dim - dimBlocks;
                     ++d, data += wordsPerVectorBlockDim) {
                    float q = SHFL_SYNC(queryReg, d, kWarpSize);

                    EncodeT enc =
                            WarpPackedBits<EncodeT, Codec::kEncodeBits>::read(
                                    laneId, data);
                    enc = WarpPackedBits<EncodeT, Codec::kEncodeBits>::postRead(
                            laneId, enc);
                    float dec = codec.decodeNew(dimBlocks + d, enc);
                    if (Residual) {
                        dec += SHFL_SYNC(residualReg, d, kWarpSize);
                    }

                    dist.handle(q, dec);
                }

                if (valid) {
                    heap.addThreadQ(dist.reduce(), vec);
                }

                heap.checkThreadQ();
            }

            heap.reduce();

            auto distanceOutBase = distanceOut[queryId][probeId].data();
            auto indicesOutBase = indicesOut[queryId][probeId].data();

            for (int i = threadIdx.x; i < k; i += blockDim.x) {
                distanceOutBase[i] = smemK[i];
                indicesOutBase[i] = smemV[i];
            }
        }
    }
}

//
// We split up the scan function into multiple compilation units to cut down on
// compile time using these macros to define the function body
//

// Top-level IVF scan function for the interleaved by kWarpSize layout
// with all implementations
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
        GpuResources* res);

// Second pass of IVF list scanning to perform final k-selection and look up the
// user indices
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
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
