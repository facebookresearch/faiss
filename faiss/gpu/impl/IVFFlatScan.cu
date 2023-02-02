/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/utils/Comparators.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>

#include <algorithm>

namespace faiss {
namespace gpu {

namespace {

/// Sort direction per each metric
inline bool metricToSortDirection(MetricType mt) {
    switch (mt) {
        case MetricType::METRIC_INNER_PRODUCT:
            // highest
            return true;
        case MetricType::METRIC_L2:
            // lowest
            return false;
        default:
            // unhandled metric
            FAISS_ASSERT(false);
            return false;
    }
}

} // namespace

// Number of warps we create per block of IVFFlatScan
constexpr int kIVFFlatScanWarps = 4;

// Works for any dimension size
template <typename Codec, typename Metric>
struct IVFFlatScan {
    static __device__ void scan(
            float* query,
            bool useResidual,
            float* residualBaseSlice,
            void* vecData,
            const Codec& codec,
            const Metric& metric,
            idx_t numVecs,
            int dim,
            float* distanceOut) {
        // How many separate loading points are there for the decoder?
        int limit = utils::divDown(dim, Codec::kDimPerIter);

        // Each warp handles a separate chunk of vectors
        int warpId = threadIdx.x / kWarpSize;
        // FIXME: why does getLaneId() not work when we write out below!?!?!
        int laneId = threadIdx.x % kWarpSize; // getLaneId();

        // Divide the set of vectors among the warps
        idx_t vecsPerWarp = utils::divUp(numVecs, kIVFFlatScanWarps);

        idx_t vecStart = vecsPerWarp * warpId;
        idx_t vecEnd = min(vecsPerWarp * (warpId + 1), numVecs);

        // Walk the list of vectors for this warp
        for (idx_t vec = vecStart; vec < vecEnd; ++vec) {
            Metric dist = metric.zero();

            // Scan the dimensions available that have whole units for the
            // decoder, as the decoder may handle more than one dimension at
            // once (leaving the remainder to be handled separately)
            for (int d = laneId; d < limit; d += kWarpSize) {
                int realDim = d * Codec::kDimPerIter;
                float vecVal[Codec::kDimPerIter];

                // Decode the kDimPerIter dimensions
                codec.decode(vecData, vec, d, vecVal);

#pragma unroll
                for (int j = 0; j < Codec::kDimPerIter; ++j) {
                    vecVal[j] +=
                            useResidual ? residualBaseSlice[realDim + j] : 0.0f;
                }

#pragma unroll
                for (int j = 0; j < Codec::kDimPerIter; ++j) {
                    dist.handle(query[realDim + j], vecVal[j]);
                }
            }

            // Handle remainder by a single thread, if any
            // Not needed if we decode 1 dim per time
            if (Codec::kDimPerIter > 1) {
                int realDim = limit * Codec::kDimPerIter;

                // Was there any remainder?
                if (realDim < dim) {
                    // Let the first threads in the block sequentially perform
                    // it
                    int remainderDim = realDim + laneId;

                    if (remainderDim < dim) {
                        float vecVal = codec.decodePartial(
                                vecData, vec, limit, laneId);
                        vecVal += useResidual ? residualBaseSlice[remainderDim]
                                              : 0.0f;
                        dist.handle(query[remainderDim], vecVal);
                    }
                }
            }

            // Reduce distance within warp
            auto warpDist = warpReduceAllSum(dist.reduce());

            if (laneId == 0) {
                distanceOut[vec] = warpDist;
            }
        }
    }
};

template <typename Codec, typename Metric>
__global__ void ivfFlatScan(
        Tensor<float, 2, true> queries,
        bool useResidual,
        Tensor<float, 3, true> residualBase,
        Tensor<idx_t, 2, true> listIds,
        void** allListData,
        idx_t* listLengths,
        Codec codec,
        Metric metric,
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance) {
    extern __shared__ float smem[];

    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;

    // This is where we start writing out data
    // We ensure that before the array (at offset -1), there is a 0 value
    auto outBase = *(prefixSumOffsets[queryId][probeId].data() - 1);

    idx_t listId = listIds[queryId][probeId];
    // Safety guard in case NaNs in input cause no list ID to be generated
    if (listId == -1) {
        return;
    }

    auto query = queries[queryId].data();
    auto vecs = allListData[listId];
    auto numVecs = listLengths[listId];
    auto dim = queries.getSize(1);
    auto distanceOut = distance[outBase].data();

    auto residualBaseSlice = residualBase[queryId][probeId].data();

    codec.initKernel(smem, dim);
    __syncthreads();

    IVFFlatScan<Codec, Metric>::scan(
            query,
            useResidual,
            residualBaseSlice,
            vecs,
            codec,
            metric,
            numVecs,
            dim,
            distanceOut);
}

void runIVFFlatScanTile(
        GpuResources* res,
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        Tensor<char, 1, true>& thrustMem,
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<float, 1, true>& allDistances,
        Tensor<float, 3, true>& heapDistances,
        Tensor<idx_t, 3, true>& heapIndices,
        int k,
        bool use64BitSelection,
        faiss::MetricType metricType,
        bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        cudaStream_t stream) {
    auto dim = queries.getSize(1);

    // Calculate offset lengths, so we know where to write out
    // intermediate results
    runCalcListOffsets(
            res, listIds, listLengths, prefixSumOffsets, thrustMem, stream);

    auto grid = dim3(listIds.getSize(1), listIds.getSize(0));
    auto block = dim3(kWarpSize * kIVFFlatScanWarps);

#define RUN_IVF_FLAT                                                  \
    do {                                                              \
        ivfFlatScan<<<grid, block, codec.getSmemSize(dim), stream>>>( \
                queries,                                              \
                useResidual,                                          \
                residualBase,                                         \
                listIds,                                              \
                listData.data(),                                      \
                listLengths.data(),                                   \
                codec,                                                \
                metric,                                               \
                prefixSumOffsets,                                     \
                allDistances);                                        \
    } while (0)

#define HANDLE_METRICS                             \
    do {                                           \
        if (metricType == MetricType::METRIC_L2) { \
            L2Distance metric;                     \
            RUN_IVF_FLAT;                          \
        } else {                                   \
            IPDistance metric;                     \
            RUN_IVF_FLAT;                          \
        }                                          \
    } while (0)

    if (!scalarQ) {
        CodecFloat codec(dim * sizeof(float));
        HANDLE_METRICS;
    } else {
        switch (scalarQ->qtype) {
            case ScalarQuantizer::QuantizerType::QT_8bit: {
                Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1> codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);
                HANDLE_METRICS;
            } break;
            case ScalarQuantizer::QuantizerType::QT_8bit_uniform: {
                Codec<ScalarQuantizer::QuantizerType::QT_8bit_uniform, 1> codec(
                        scalarQ->code_size,
                        scalarQ->trained[0],
                        scalarQ->trained[1]);
                HANDLE_METRICS;
            } break;
            case ScalarQuantizer::QuantizerType::QT_fp16: {
                Codec<ScalarQuantizer::QuantizerType::QT_fp16, 1> codec(
                        scalarQ->code_size);
                HANDLE_METRICS;
            } break;
            case ScalarQuantizer::QuantizerType::QT_8bit_direct: {
                Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct, 1> codec(
                        scalarQ->code_size);
                HANDLE_METRICS;
            } break;
            case ScalarQuantizer::QuantizerType::QT_4bit: {
                Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1> codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);
                HANDLE_METRICS;
            } break;
            case ScalarQuantizer::QuantizerType::QT_4bit_uniform: {
                Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform, 1> codec(
                        scalarQ->code_size,
                        scalarQ->trained[0],
                        scalarQ->trained[1]);
                HANDLE_METRICS;
            } break;
            default:
                // unimplemented, should be handled at a higher level
                FAISS_ASSERT(false);
        }
    }

    CUDA_TEST_ERROR();

#undef HANDLE_METRICS
#undef RUN_IVF_FLAT

    // k-select the output in chunks, to increase parallelism
    runPass1SelectLists(
            prefixSumOffsets,
            allDistances,
            listIds.getSize(1),
            k,
            use64BitSelection,
            metricToSortDirection(metricType),
            heapDistances,
            heapIndices,
            stream);

    // k-select final output
    auto flatHeapDistances = heapDistances.downcastInner<2>();
    auto flatHeapIndices = heapIndices.downcastInner<2>();

    runPass2SelectLists(
            flatHeapDistances,
            flatHeapIndices,
            listIndices,
            indicesOptions,
            prefixSumOffsets,
            listIds,
            k,
            use64BitSelection,
            metricToSortDirection(metricType),
            outDistances,
            outIndices,
            stream);
}

void runIVFFlatScan(
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
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
    auto stream = res->getDefaultStreamCurrentDevice();

    constexpr idx_t kMinQueryTileSize = 8;
    constexpr idx_t kMaxQueryTileSize = 65536; // used as blockIdx.y dimension
    constexpr idx_t kThrustMemSize = 16384;

    auto nprobe = listIds.getSize(1);

    // If the maximum list length (in terms of number of vectors) times nprobe
    // (number of lists) is > 2^31 - 1, then we will use 64-bit indexing in the
    // selection kernels
    bool use64BitSelection =
            maxListLength * nprobe > idx_t(std::numeric_limits<int32_t>::max());

    // Make a reservation for Thrust to do its dirty work (global memory
    // cross-block reduction space); hopefully this is large enough.
    DeviceTensor<char, 1, true> thrustMem1(
            res, makeTempAlloc(AllocType::Other, stream), {kThrustMemSize});
    DeviceTensor<char, 1, true> thrustMem2(
            res, makeTempAlloc(AllocType::Other, stream), {kThrustMemSize});
    DeviceTensor<char, 1, true>* thrustMem[2] = {&thrustMem1, &thrustMem2};

    // How much temporary storage is available?
    // If possible, we'd like to fit within the space available.
    size_t sizeAvailable = res->getTempMemoryAvailableCurrentDevice();

    // We run two passes of heap selection
    // This is the size of the first-level heap passes
    constexpr idx_t kNProbeSplit = 8;
    idx_t pass2Chunks = std::min(nprobe, kNProbeSplit);

    idx_t sizeForFirstSelectPass =
            pass2Chunks * k * (sizeof(float) + sizeof(idx_t));

    // How much temporary storage we need per each query
    idx_t sizePerQuery = 2 *                            // # streams
            ((nprobe * sizeof(idx_t) + sizeof(idx_t)) + // prefixSumOffsets
             nprobe * maxListLength * sizeof(float) +   // allDistances
             sizeForFirstSelectPass);

    idx_t queryTileSize = sizeAvailable / sizePerQuery;

    if (queryTileSize < kMinQueryTileSize) {
        queryTileSize = kMinQueryTileSize;
    } else if (queryTileSize > kMaxQueryTileSize) {
        queryTileSize = kMaxQueryTileSize;
    }

    // Temporary memory buffers
    // Make sure there is space prior to the start which will be 0, and
    // will handle the boundary condition without branches
    DeviceTensor<idx_t, 1, true> prefixSumOffsetSpace1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe + 1});
    DeviceTensor<idx_t, 1, true> prefixSumOffsetSpace2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe + 1});

    DeviceTensor<idx_t, 2, true> prefixSumOffsets1(
            prefixSumOffsetSpace1[1].data(), {queryTileSize, nprobe});
    DeviceTensor<idx_t, 2, true> prefixSumOffsets2(
            prefixSumOffsetSpace2[1].data(), {queryTileSize, nprobe});
    DeviceTensor<idx_t, 2, true>* prefixSumOffsets[2] = {
            &prefixSumOffsets1, &prefixSumOffsets2};

    // Make sure the element before prefixSumOffsets is 0, since we
    // depend upon simple, boundary-less indexing to get proper results
    CUDA_VERIFY(cudaMemsetAsync(
            prefixSumOffsetSpace1.data(), 0, sizeof(idx_t), stream));
    CUDA_VERIFY(cudaMemsetAsync(
            prefixSumOffsetSpace2.data(), 0, sizeof(idx_t), stream));

    DeviceTensor<float, 1, true> allDistances1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe * maxListLength});
    DeviceTensor<float, 1, true> allDistances2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe * maxListLength});
    DeviceTensor<float, 1, true>* allDistances[2] = {
            &allDistances1, &allDistances2};

    DeviceTensor<float, 3, true> heapDistances1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<float, 3, true> heapDistances2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<float, 3, true>* heapDistances[2] = {
            &heapDistances1, &heapDistances2};

    DeviceTensor<idx_t, 3, true> heapIndices1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<idx_t, 3, true> heapIndices2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<idx_t, 3, true>* heapIndices[2] = {
            &heapIndices1, &heapIndices2};

    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    int curStream = 0;

    for (idx_t query = 0; query < queries.getSize(0); query += queryTileSize) {
        auto numQueriesInTile =
                std::min(queryTileSize, queries.getSize(0) - query);

        auto prefixSumOffsetsView =
                prefixSumOffsets[curStream]->narrowOutermost(
                        0, numQueriesInTile);

        auto listIdsView = listIds.narrowOutermost(query, numQueriesInTile);
        auto queryView = queries.narrowOutermost(query, numQueriesInTile);
        auto residualBaseView =
                residualBase.narrowOutermost(query, numQueriesInTile);

        auto heapDistancesView =
                heapDistances[curStream]->narrowOutermost(0, numQueriesInTile);
        auto heapIndicesView =
                heapIndices[curStream]->narrowOutermost(0, numQueriesInTile);

        auto outDistanceView =
                outDistances.narrowOutermost(query, numQueriesInTile);
        auto outIndicesView =
                outIndices.narrowOutermost(query, numQueriesInTile);

        runIVFFlatScanTile(
                res,
                queryView,
                listIdsView,
                listData,
                listIndices,
                indicesOptions,
                listLengths,
                *thrustMem[curStream],
                prefixSumOffsetsView,
                *allDistances[curStream],
                heapDistancesView,
                heapIndicesView,
                k,
                use64BitSelection,
                metric,
                useResidual,
                residualBaseView,
                scalarQ,
                outDistanceView,
                outIndicesView,
                streams[curStream]);

        curStream = (curStream + 1) % 2;
    }

    streamWait({stream}, streams);
}

} // namespace gpu
} // namespace faiss
