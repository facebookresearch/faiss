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
#include <thrust/device_vector.h>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/utils/Comparators.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
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
        int NumThreadQ,
        bool Residual>
__global__ void ivfInterleavedScan(
        Tensor<float, 2, true> queries,
        Tensor<float, 3, true> residualBase,
        Tensor<int, 2, true> listIds,
        void** allListData,
        int* listLengths,
        Codec codec,
        Metric metric,
        int k,
        // [query][probe][k]
        Tensor<float, 3, true> distanceOut,
        Tensor<int, 3, true> indicesOut) {
    extern __shared__ float smem[];

    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    int queryId = blockIdx.y;
    int probeId = blockIdx.x;
    int listId = listIds[queryId][probeId];

    // Safety guard in case NaNs in input cause no list ID to be generated, or
    // we have more nprobe than nlist
    if (listId == -1) {
        return;
    }

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
    __shared__ int smemV[kNumWarps * NumWarpQ];

    BlockSelect<
            float,
            int,
            Metric::kDirection,
            Comparator<float>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(kInit, -1, smemK, smemV, k);

    // The codec might be dependent upon data that we need to reference or store
    // in shared memory
    codec.initKernel(smem, dim);
    __syncthreads();

    // How many vector blocks of 32 are in this list?
    int numBlocks = utils::divUp(numVecs, 32);

    // Number of EncodeT words per each dimension of block of 32 vecs
    constexpr int bytesPerVectorBlockDim = Codec::kEncodeBits * 32 / 8;
    constexpr int wordsPerVectorBlockDim =
            bytesPerVectorBlockDim / sizeof(EncodeT);
    int wordsPerVectorBlock = wordsPerVectorBlockDim * dim;

    int dimBlocks = utils::roundDown(dim, kWarpSize);

    for (int block = warpId; block < numBlocks; block += kNumWarps) {
        // We're handling a new vector
        Metric dist = metric.zero();

        // This is the vector a given lane/thread handles
        int vec = block * kWarpSize + laneId;
        bool valid = vec < numVecs;

        // This is where this warp begins reading data
        EncodeT* data = vecsBase + block * wordsPerVectorBlock;

        // whole blocks
        for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
            int loadDim = dBase + laneId;
            float queryReg = query[loadDim];
            float residualReg = Residual ? residualBaseSlice[loadDim] : 0;

            constexpr int kUnroll = 4;

#pragma unroll
            for (int i = 0; i < kWarpSize / kUnroll;
                 ++i, data += kUnroll * wordsPerVectorBlockDim) {
                EncodeT encV[kUnroll];
#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    encV[j] = WarpPackedBits<EncodeT, Codec::kEncodeBits>::read(
                            laneId, data + j * wordsPerVectorBlockDim);
                }

#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    encV[j] = WarpPackedBits<EncodeT, Codec::kEncodeBits>::
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
        int loadDim = dimBlocks + laneId;
        bool loadDimInBounds = loadDim < dim;

        float queryReg = loadDimInBounds ? query[loadDim] : 0;
        float residualReg =
                Residual && loadDimInBounds ? residualBaseSlice[loadDim] : 0;

        for (int d = 0; d < dim - dimBlocks;
             ++d, data += wordsPerVectorBlockDim) {
            float q = SHFL_SYNC(queryReg, d, kWarpSize);

            EncodeT enc = WarpPackedBits<EncodeT, Codec::kEncodeBits>::read(
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

//
// We split up the scan function into multiple compilation units to cut down on
// compile time using these macros to define the function body
//

#define IVFINT_RUN(CODEC_TYPE, METRIC_TYPE, THREADS, NUM_WARP_Q, NUM_THREAD_Q) \
    do {                                                                       \
        dim3 grid(nprobe, nq);                                                 \
        if (useResidual) {                                                     \
            ivfInterleavedScan<                                                \
                    CODEC_TYPE,                                                \
                    METRIC_TYPE,                                               \
                    THREADS,                                                   \
                    NUM_WARP_Q,                                                \
                    NUM_THREAD_Q,                                              \
                    true><<<grid, THREADS, codec.getSmemSize(dim), stream>>>(  \
                    queries,                                                   \
                    residualBase,                                              \
                    listIds,                                                   \
                    listData.data().get(),                                     \
                    listLengths.data().get(),                                  \
                    codec,                                                     \
                    metric,                                                    \
                    k,                                                         \
                    distanceTemp,                                              \
                    indicesTemp);                                              \
        } else {                                                               \
            ivfInterleavedScan<                                                \
                    CODEC_TYPE,                                                \
                    METRIC_TYPE,                                               \
                    THREADS,                                                   \
                    NUM_WARP_Q,                                                \
                    NUM_THREAD_Q,                                              \
                    false><<<grid, THREADS, codec.getSmemSize(dim), stream>>>( \
                    queries,                                                   \
                    residualBase,                                              \
                    listIds,                                                   \
                    listData.data().get(),                                     \
                    listLengths.data().get(),                                  \
                    codec,                                                     \
                    metric,                                                    \
                    k,                                                         \
                    distanceTemp,                                              \
                    indicesTemp);                                              \
        }                                                                      \
                                                                               \
        runIVFInterleavedScan2(                                                \
                distanceTemp,                                                  \
                indicesTemp,                                                   \
                listIds,                                                       \
                k,                                                             \
                listIndices,                                                   \
                indicesOptions,                                                \
                METRIC_TYPE::kDirection,                                       \
                outDistances,                                                  \
                outIndices,                                                    \
                stream);                                                       \
    } while (0);

#define IVFINT_CODECS(METRIC_TYPE, THREADS, NUM_WARP_Q, NUM_THREAD_Q)          \
    do {                                                                       \
        if (!scalarQ) {                                                        \
            using CodecT = CodecFloat;                                         \
            CodecT codec(dim * sizeof(float));                                 \
            IVFINT_RUN(                                                        \
                    CodecT, METRIC_TYPE, THREADS, NUM_WARP_Q, NUM_THREAD_Q);   \
        } else {                                                               \
            switch (scalarQ->qtype) {                                          \
                case ScalarQuantizer::QuantizerType::QT_8bit: {                \
                    using CodecT =                                             \
                            Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1>; \
                    CodecT codec(                                              \
                            scalarQ->code_size,                                \
                            scalarQ->gpuTrained.data(),                        \
                            scalarQ->gpuTrained.data() + dim);                 \
                    IVFINT_RUN(                                                \
                            CodecT,                                            \
                            METRIC_TYPE,                                       \
                            THREADS,                                           \
                            NUM_WARP_Q,                                        \
                            NUM_THREAD_Q);                                     \
                } break;                                                       \
                case ScalarQuantizer::QuantizerType::QT_8bit_uniform: {        \
                    using CodecT = Codec<                                      \
                            ScalarQuantizer::QuantizerType::QT_8bit_uniform,   \
                            1>;                                                \
                    CodecT codec(                                              \
                            scalarQ->code_size,                                \
                            scalarQ->trained[0],                               \
                            scalarQ->trained[1]);                              \
                    IVFINT_RUN(                                                \
                            CodecT,                                            \
                            METRIC_TYPE,                                       \
                            THREADS,                                           \
                            NUM_WARP_Q,                                        \
                            NUM_THREAD_Q);                                     \
                } break;                                                       \
                case ScalarQuantizer::QuantizerType::QT_fp16: {                \
                    using CodecT =                                             \
                            Codec<ScalarQuantizer::QuantizerType::QT_fp16, 1>; \
                    CodecT codec(scalarQ->code_size);                          \
                    IVFINT_RUN(                                                \
                            CodecT,                                            \
                            METRIC_TYPE,                                       \
                            THREADS,                                           \
                            NUM_WARP_Q,                                        \
                            NUM_THREAD_Q);                                     \
                } break;                                                       \
                case ScalarQuantizer::QuantizerType::QT_8bit_direct: {         \
                    using CodecT = Codec<                                      \
                            ScalarQuantizer::QuantizerType::QT_8bit_direct,    \
                            1>;                                                \
                    Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct, 1>   \
                            codec(scalarQ->code_size);                         \
                    IVFINT_RUN(                                                \
                            CodecT,                                            \
                            METRIC_TYPE,                                       \
                            THREADS,                                           \
                            NUM_WARP_Q,                                        \
                            NUM_THREAD_Q);                                     \
                } break;                                                       \
                case ScalarQuantizer::QuantizerType::QT_6bit: {                \
                    using CodecT =                                             \
                            Codec<ScalarQuantizer::QuantizerType::QT_6bit, 1>; \
                    Codec<ScalarQuantizer::QuantizerType::QT_6bit, 1> codec(   \
                            scalarQ->code_size,                                \
                            scalarQ->gpuTrained.data(),                        \
                            scalarQ->gpuTrained.data() + dim);                 \
                    IVFINT_RUN(                                                \
                            CodecT,                                            \
                            METRIC_TYPE,                                       \
                            THREADS,                                           \
                            NUM_WARP_Q,                                        \
                            NUM_THREAD_Q);                                     \
                } break;                                                       \
                case ScalarQuantizer::QuantizerType::QT_4bit: {                \
                    using CodecT =                                             \
                            Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1>; \
                    Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1> codec(   \
                            scalarQ->code_size,                                \
                            scalarQ->gpuTrained.data(),                        \
                            scalarQ->gpuTrained.data() + dim);                 \
                    IVFINT_RUN(                                                \
                            CodecT,                                            \
                            METRIC_TYPE,                                       \
                            THREADS,                                           \
                            NUM_WARP_Q,                                        \
                            NUM_THREAD_Q);                                     \
                } break;                                                       \
                case ScalarQuantizer::QuantizerType::QT_4bit_uniform: {        \
                    using CodecT = Codec<                                      \
                            ScalarQuantizer::QuantizerType::QT_4bit_uniform,   \
                            1>;                                                \
                    Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform, 1>  \
                            codec(scalarQ->code_size,                          \
                                  scalarQ->trained[0],                         \
                                  scalarQ->trained[1]);                        \
                    IVFINT_RUN(                                                \
                            CodecT,                                            \
                            METRIC_TYPE,                                       \
                            THREADS,                                           \
                            NUM_WARP_Q,                                        \
                            NUM_THREAD_Q);                                     \
                } break;                                                       \
                default:                                                       \
                    FAISS_ASSERT(false);                                       \
            }                                                                  \
        }                                                                      \
    } while (0)

#define IVFINT_METRICS(THREADS, NUM_WARP_Q, NUM_THREAD_Q)                 \
    do {                                                                  \
        auto stream = res->getDefaultStreamCurrentDevice();               \
        auto nq = queries.getSize(0);                                     \
        auto dim = queries.getSize(1);                                    \
        auto nprobe = listIds.getSize(1);                                 \
                                                                          \
        DeviceTensor<float, 3, true> distanceTemp(                        \
                res,                                                      \
                makeTempAlloc(AllocType::Other, stream),                  \
                {queries.getSize(0), listIds.getSize(1), k});             \
        DeviceTensor<int, 3, true> indicesTemp(                           \
                res,                                                      \
                makeTempAlloc(AllocType::Other, stream),                  \
                {queries.getSize(0), listIds.getSize(1), k});             \
                                                                          \
        if (metric == MetricType::METRIC_L2) {                            \
            L2Distance metric;                                            \
            IVFINT_CODECS(L2Distance, THREADS, NUM_WARP_Q, NUM_THREAD_Q); \
        } else if (metric == MetricType::METRIC_INNER_PRODUCT) {          \
            IPDistance metric;                                            \
            IVFINT_CODECS(IPDistance, THREADS, NUM_WARP_Q, NUM_THREAD_Q); \
        } else {                                                          \
            FAISS_ASSERT(false);                                          \
        }                                                                 \
    } while (0)

// Top-level IVF scan function for the interleaved by 32 layout
// with all implementations
void runIVFInterleavedScan(
        Tensor<float, 2, true>& queries,
        Tensor<int, 2, true>& listIds,
        thrust::device_vector<void*>& listData,
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        thrust::device_vector<int>& listLengths,
        int k,
        faiss::MetricType metric,
        bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<Index::idx_t, 2, true>& outIndices,
        GpuResources* res);

// Second pass of IVF list scanning to perform final k-selection and look up the
// user indices
void runIVFInterleavedScan2(
        Tensor<float, 3, true>& distanceIn,
        Tensor<int, 3, true>& indicesIn,
        Tensor<int, 2, true>& listIds,
        int k,
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        bool dir,
        Tensor<float, 2, true>& distanceOut,
        Tensor<Index::idx_t, 2, true>& indicesOut,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
