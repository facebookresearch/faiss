/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/BroadcastSum.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Transpose.cuh>

namespace faiss {
namespace gpu {

#if defined(USE_AMD_ROCM) && __AMDGCN_WAVEFRONT_SIZE == 64u
#define LAUNCH_BOUND 320
#else
#define LAUNCH_BOUND 288
#endif

// Kernel responsible for calculating distance from residual vector to
// each product quantizer code centroid
template <
        typename OutCodeT,
        typename CentroidT,
        int DimsPerSubQuantizer,
        bool L2Distance>
__global__ void __launch_bounds__(LAUNCH_BOUND, 3) pqCodeDistances(
        Tensor<float, 2, true> queries,
        int queriesPerBlock,
        Tensor<CentroidT, 2, true> coarseCentroids,
        Tensor<float, 3, true> pqCentroids,
        Tensor<idx_t, 2, true> coarseIndices,
        // (query id)(coarse)(subquantizer)(code) -> dist
        Tensor<OutCodeT, 4, true> outCodeDistances) {
    const auto numSubQuantizers = pqCentroids.getSize(0);
    const auto dimsPerSubQuantizer = pqCentroids.getSize(1);
    assert(DimsPerSubQuantizer == dimsPerSubQuantizer);
    const auto codesPerSubQuantizer = pqCentroids.getSize(2);

    bool isLoadingThread = threadIdx.x >= codesPerSubQuantizer;
    int loadingThreadId = threadIdx.x - codesPerSubQuantizer;

    extern __shared__ float smem[];

    // Each thread calculates a single code
    float subQuantizerData[DimsPerSubQuantizer];

    auto code = threadIdx.x;
    auto subQuantizer = blockIdx.y;

    // Each thread will load the pq centroid data for the code that it
    // is processing
    // The loading threads are out of bounds for the number of codes available
    if (!isLoadingThread) {
#pragma unroll
        for (int i = 0; i < DimsPerSubQuantizer; ++i) {
            subQuantizerData[i] = pqCentroids[subQuantizer][i][code].ldg();
        }
    }

    // Where we store our query vector
    float* smemQuery = smem;

    // Where we store our residual vector; this is double buffered so we
    // can be loading the next one while processing the current one
    float* smemResidual1 = &smemQuery[DimsPerSubQuantizer];
    float* smemResidual2 = &smemResidual1[DimsPerSubQuantizer];

    // Where we pre-load the coarse centroid IDs
    int* coarseIds = (int*)&smemResidual2[DimsPerSubQuantizer];

    // Each thread is calculating the distance for a single code,
    // performing the reductions locally

    // Handle multiple queries per block
    auto startQueryId = idx_t(blockIdx.x) * queriesPerBlock;
    auto numQueries = queries.getSize(0) - startQueryId;
    if (numQueries > queriesPerBlock) {
        numQueries = queriesPerBlock;
    }

    for (int query = 0; query < numQueries; ++query) {
        auto queryId = startQueryId + query;

        auto querySubQuantizer =
                queries[queryId][subQuantizer * DimsPerSubQuantizer].data();

        // Load current query vector
        for (int i = threadIdx.x; i < DimsPerSubQuantizer; i += blockDim.x) {
            smemQuery[i] = querySubQuantizer[i];
        }

        // Load list of coarse centroids found
        for (int i = threadIdx.x; i < coarseIndices.getSize(1);
             i += blockDim.x) {
            // FIXME: coarseIndices is now idx_t but the smem allocation
            // of coarseIds is still int. In practical limitation, everything
            // should still fit into int32
            coarseIds[i] = (int)coarseIndices[queryId][i];
        }

        // We need coarseIds below
        // FIXME: investigate loading separately, so we don't need this
        __syncthreads();

        // Preload first buffer of residual data
        if (isLoadingThread) {
            for (int i = loadingThreadId; i < DimsPerSubQuantizer;
                 i += blockDim.x - codesPerSubQuantizer) {
                auto coarseId = coarseIds[0];
                // In case NaNs were in the original query data
                coarseId = coarseId == -1 ? 0 : coarseId;
                auto coarseCentroidSubQuantizer =
                        coarseCentroids[coarseId]
                                       [subQuantizer * dimsPerSubQuantizer]
                                               .data();

                if (L2Distance) {
                    smemResidual1[i] = smemQuery[i] -
                            ConvertTo<float>::to(coarseCentroidSubQuantizer[i]);
                } else {
                    smemResidual1[i] =
                            ConvertTo<float>::to(coarseCentroidSubQuantizer[i]);
                }
            }
        }

        // The block walks the list for a single query
        for (int coarse = 0; coarse < coarseIndices.getSize(1); ++coarse) {
            // Wait for smemResidual1 to be loaded
            __syncthreads();

            if (isLoadingThread) {
                // Preload second buffer of residual data
                for (int i = loadingThreadId; i < DimsPerSubQuantizer;
                     i += blockDim.x - codesPerSubQuantizer) {
                    // FIXME: try always making this centroid id 0 so we can
                    // terminate
                    if (coarse != (coarseIndices.getSize(1) - 1)) {
                        auto coarseId = coarseIds[coarse + 1];
                        // In case NaNs were in the original query data
                        coarseId = coarseId == -1 ? 0 : coarseId;

                        auto coarseCentroidSubQuantizer =
                                coarseCentroids[coarseId]
                                               [subQuantizer *
                                                dimsPerSubQuantizer]
                                                       .data();

                        if (L2Distance) {
                            smemResidual2[i] =
                                    smemQuery[i] -
                                    ConvertTo<float>::to(
                                            coarseCentroidSubQuantizer[i]);
                        } else {
                            smemResidual2[i] = ConvertTo<float>::to(
                                    coarseCentroidSubQuantizer[i]);
                        }
                    }
                }
            } else {
                // These are the processing threads
                float dist = 0.0f;

                constexpr int kUnroll = 4;
                constexpr int kRemainder = DimsPerSubQuantizer % kUnroll;
                constexpr int kRemainderBase = DimsPerSubQuantizer - kRemainder;
                float vals[kUnroll];

                // Calculate residual - pqCentroid for each dim that we're
                // processing

                // Unrolled loop
                if (L2Distance) {
#pragma unroll
                    for (int i = 0; i < DimsPerSubQuantizer / kUnroll; ++i) {
#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            vals[j] = smemResidual1[i * kUnroll + j];
                        }

#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            vals[j] -= subQuantizerData[i * kUnroll + j];
                        }

#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            vals[j] *= vals[j];
                        }

#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            dist += vals[j];
                        }
                    }
                } else {
                    // Inner product: query slice against the reconstructed
                    // sub-quantizer for this coarse cell (query o (centroid +
                    // subQCentroid))
#pragma unroll
                    for (int i = 0; i < DimsPerSubQuantizer / kUnroll; ++i) {
#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            vals[j] = smemResidual1[i * kUnroll + j];
                        }

#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            vals[j] += subQuantizerData[i * kUnroll + j];
                        }

#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            vals[j] *= smemQuery[i * kUnroll + j];
                        }

#pragma unroll
                        for (int j = 0; j < kUnroll; ++j) {
                            dist += vals[j];
                        }
                    }
                }

                // Remainder loop
                if (L2Distance) {
#pragma unroll
                    for (int j = 0; j < kRemainder; ++j) {
                        vals[j] = smemResidual1[kRemainderBase + j];
                    }

#pragma unroll
                    for (int j = 0; j < kRemainder; ++j) {
                        vals[j] -= subQuantizerData[kRemainderBase + j];
                    }

#pragma unroll
                    for (int j = 0; j < kRemainder; ++j) {
                        vals[j] *= vals[j];
                    }
                } else {
                    // Inner product
                    // Inner product: query slice against the reconstructed
                    // sub-quantizer for this coarse cell (query o (centroid +
                    // subQCentroid))
#pragma unroll
                    for (int j = 0; j < kRemainder; ++j) {
                        vals[j] = smemResidual1[kRemainderBase + j];
                    }

#pragma unroll
                    for (int j = 0; j < kRemainder; ++j) {
                        vals[j] += subQuantizerData[kRemainderBase + j];
                    }

#pragma unroll
                    for (int j = 0; j < kRemainder; ++j) {
                        vals[j] *= smemQuery[kRemainderBase + j];
                    }
                }

#pragma unroll
                for (int j = 0; j < kRemainder; ++j) {
                    dist += vals[j];
                }

                // We have the distance for our code; write it out
                outCodeDistances[queryId][coarse][subQuantizer][code] =
                        ConvertTo<OutCodeT>::to(dist);
            } // !isLoadingThread

            // Swap residual buffers
            float* tmp = smemResidual1;
            smemResidual1 = smemResidual2;
            smemResidual2 = tmp;
        }
    }
}

template <typename CentroidT, bool L2Residual>
__global__ void pqResidualVector(
        Tensor<float, 2, true> queries,
        Tensor<CentroidT, 2, true> coarseCentroids,
        Tensor<idx_t, 2, true> coarseIndices,
        int numSubDim,
        // output is transposed:
        // (sub q)(query id)(centroid id)(sub dim)
        Tensor<float, 4, true> residual) {
    auto queryId = blockIdx.x;
    auto centroidId = blockIdx.y;

    idx_t realCentroidId = coarseIndices[queryId][centroidId];

    for (int dim = threadIdx.x; dim < queries.getSize(1); dim += blockDim.x) {
        float q = queries[queryId][dim];
        float c = ConvertTo<float>::to(coarseCentroids[realCentroidId][dim]);

        float r;

        if (L2Residual) {
            r = q - c;
        } else {
            // IP does not use a residual. Instead, the estimated distance is
            // (query . (centroid + sub quantizer centroid).
            //
            // This kernel is used to calculate (query . sub quantizer
            // centroid), providing the query value replicated across all of the
            // sub quantizers. The batch matrix multiplication in
            // runPQCodeDistancesMM will perform this inner product. The
            // adjustment (query . centroid) is added later.
            r = q;
        }

        residual[dim / numSubDim][queryId][centroidId][dim % numSubDim] = r;
    }
}

template <typename CentroidT>
void runPQResidualVector(
        Tensor<float, 3, true>& pqCentroids,
        Tensor<float, 2, true>& queries,
        Tensor<CentroidT, 2, true>& coarseCentroids,
        Tensor<idx_t, 2, true>& coarseIndices,
        Tensor<float, 4, true>& residual,
        bool l2Residual,
        cudaStream_t stream) {
    // blockDim.y is limited by nprobe
    auto grid = dim3(coarseIndices.getSize(0), coarseIndices.getSize(1));
    auto block = dim3(
            std::min(queries.getSize(1), (idx_t)getMaxThreadsCurrentDevice()));

    if (l2Residual) {
        pqResidualVector<CentroidT, true><<<grid, block, 0, stream>>>(
                queries,
                coarseCentroids,
                coarseIndices,
                pqCentroids.getSize(1),
                residual);
    } else {
        pqResidualVector<CentroidT, false><<<grid, block, 0, stream>>>(
                queries,
                coarseCentroids,
                coarseIndices,
                pqCentroids.getSize(1),
                residual);
    }

    CUDA_TEST_ERROR();
}

template <typename T>
__global__ void pqDistanceIPCorrection(
        Tensor<T, 3, true> codeDistances,
        Tensor<T, 2, true> coarseDistances,
        int numSubQ) {
    int centroid = blockIdx.x;
    int query = blockIdx.y;

    // We need to add the (query . centroid) correction factor (coarseDistances)
    // to all output code distances (q)(c)(sub q)(code).
    // However, there are numSubQ code distance sums per each approximated
    // distance, so we need to divide this correction by numSubQ since we will
    // be adding it numSubQ times.
    auto d = coarseDistances[query][centroid] / (float)numSubQ;

    auto base = codeDistances[query][centroid].data();

    for (int i = threadIdx.x; i < codeDistances.getSize(2); i += blockDim.x) {
        base[i] += d;
    }
}

// We have previously calculated (query . sub quantizer centroid), but we
// need to calculate (query . (centroid + sub quantizer centroid). This will add
// in the correction factor to each calculated code distance.
template <typename T>
void runPQDistanceIPCorrection(
        Tensor<T, 4, true>& codeDistances,
        Tensor<T, 2, true>& coarseDistances,
        cudaStream_t stream) {
    // blockDim.y is limited by nprobe
    auto grid = dim3(coarseDistances.getSize(1), coarseDistances.getSize(0));
    auto block = 512;

    auto codeView = codeDistances.template downcastInner<3>();

    pqDistanceIPCorrection<<<grid, block, 0, stream>>>(
            codeView, coarseDistances, codeDistances.getSize(2));
}

// This is a general purpose implementation that leverages GEMM to calculate
// code distances for PQ codes for any number of dimensions per sub-quantizer /
// number of sub-quantizers
template <typename CentroidT>
void runPQCodeDistancesMM(
        GpuResources* res,
        Tensor<float, 3, true>& pqCentroids,
        Tensor<float, 2, true>& queries,
        Tensor<CentroidT, 2, true>& coarseCentroids,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        // Output is (query)(centroid)(sub q)(code)
        NoTypeTensor<4, true>& outCodeDistances,
        bool l2Distance,
        bool useFloat16Lookup,
        cudaStream_t stream) {
    // We construct our float32 output in outCodeDistancesF
    Tensor<float, 4, true> outCodeDistancesF;
    DeviceTensor<float, 4, true> outCodeDistancesFloatMem;

    if (useFloat16Lookup) {
        // outCodeDistances has half memory, we need to allocate a buffer for
        // float
        outCodeDistancesFloatMem = DeviceTensor<float, 4, true>(
                res,
                makeTempAlloc(AllocType::Other, stream),
                {outCodeDistances.getSize(0),
                 outCodeDistances.getSize(1),
                 outCodeDistances.getSize(2),
                 outCodeDistances.getSize(3)});

        outCodeDistancesF = outCodeDistancesFloatMem;
    } else {
        // We can use the memory that we were given
        outCodeDistancesF = outCodeDistances.toTensor<float>();
    }

    // Calculate (q - c) residual vector if L2. Otherwise, for IP, this kernel
    // will just replicate q
    //
    // (sub q)(query id)(centroid id)(sub dim)
    DeviceTensor<float, 4, true> residual(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {pqCentroids.getSize(0),
             coarseIndices.getSize(0),
             coarseIndices.getSize(1),
             pqCentroids.getSize(1)});

    runPQResidualVector(
            pqCentroids,
            queries,
            coarseCentroids,
            coarseIndices,
            residual,
            l2Distance,
            stream);

    // Perform a batch MM:
    // (sub q) x {(q * c)(sub dim) x (sub dim)(code)} =>
    // (sub q) x {(q * c)(code)}
    auto residualView3 = residual.template view<3>(
            {pqCentroids.getSize(0),
             coarseIndices.getSize(0) * coarseIndices.getSize(1),
             pqCentroids.getSize(1)});

    DeviceTensor<float, 3, true> residualDistance(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {pqCentroids.getSize(0),
             coarseIndices.getSize(0) * coarseIndices.getSize(1),
             pqCentroids.getSize(2)});

    runBatchMatrixMult(
            residualDistance,
            false,
            residualView3,
            false,
            pqCentroids,
            false,
            l2Distance ? -2.0f : 1.0f,
            0.0f,
            res->getBlasHandleCurrentDevice(),
            stream);

    if (l2Distance) {
        // Calculate ||q - c||^2
        DeviceTensor<float, 1, true> residualNorms(
                res,
                makeTempAlloc(AllocType::Other, stream),
                {pqCentroids.getSize(0) * coarseIndices.getSize(0) *
                 coarseIndices.getSize(1)});

        auto residualView2 = residual.template view<2>(
                {pqCentroids.getSize(0) * coarseIndices.getSize(0) *
                         coarseIndices.getSize(1),
                 pqCentroids.getSize(1)});

        runL2Norm(residualView2, true, residualNorms, true, stream);

        // Sum ||q - c||^2 along rows
        auto residualDistanceView2 = residualDistance.template view<2>(
                {pqCentroids.getSize(0) * coarseIndices.getSize(0) *
                         coarseIndices.getSize(1),
                 pqCentroids.getSize(2)});

        runSumAlongRows(residualNorms, residualDistanceView2, false, stream);
    }

    // Transpose (sub q)(q * c)(code) to (q * c)(sub q)(code) (which
    // is where we build our output distances). L2 version of this has an added
    // -2 multiplicative factor
    auto outCodeDistancesView = outCodeDistancesF.template view<3>(
            {coarseIndices.getSize(0) * coarseIndices.getSize(1),
             outCodeDistances.getSize(2),
             outCodeDistances.getSize(3)});

    runTransposeAny(residualDistance, 0, 1, outCodeDistancesView, stream);

    if (l2Distance) {
        // Calculate code norms per each sub-dim
        // (sub q)(sub dim)(code) is pqCentroids
        // transpose to (sub q)(code)(sub dim)
        DeviceTensor<float, 3, true> pqCentroidsTranspose(
                res,
                makeTempAlloc(AllocType::Other, stream),
                {pqCentroids.getSize(0),
                 pqCentroids.getSize(2),
                 pqCentroids.getSize(1)});

        runTransposeAny(pqCentroids, 1, 2, pqCentroidsTranspose, stream);

        auto pqCentroidsTransposeView = pqCentroidsTranspose.template view<2>(
                {pqCentroids.getSize(0) * pqCentroids.getSize(2),
                 pqCentroids.getSize(1)});

        // The norm of each (sub q)(code)
        DeviceTensor<float, 1, true> pqCentroidsNorm(
                res,
                makeTempAlloc(AllocType::Other, stream),
                {pqCentroids.getSize(0) * pqCentroids.getSize(2)});

        runL2Norm(
                pqCentroidsTransposeView, true, pqCentroidsNorm, true, stream);

        // View output as (q * c)(sub q * code), and add centroid norm to
        // each row
        auto outDistancesCodeViewCols = outCodeDistancesView.template view<2>(
                {coarseIndices.getSize(0) * coarseIndices.getSize(1),
                 outCodeDistances.getSize(2) * outCodeDistances.getSize(3)});

        runSumAlongColumns(pqCentroidsNorm, outDistancesCodeViewCols, stream);
    } else {
        // We have previously calculated (query . sub quantizer centroid), but
        // we need to calculate (query . (centroid + sub quantizer centroid).
        //
        // We need to add the (query . centroid) correction factor
        // (coarseDistances) to all output code distances (q)(c)(sub q)(code).
        runPQDistanceIPCorrection(outCodeDistancesF, coarseDistances, stream);
    }

    if (useFloat16Lookup) {
        // Need to convert back to half in the output memory
        auto outCodeDistancesH = outCodeDistances.toTensor<half>();
        convertTensor<float, half, 4>(
                stream, outCodeDistancesF, outCodeDistancesH);
    }
}

// Must be kept in sync with runPQDistances
inline bool isSpecializedPQCodeDistanceDims(int dims) {
    switch (dims) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 8:
        case 10:
        case 12:
        case 16:
        case 20:
        case 24:
        case 28:
        case 32:
            return true;
        default:
            return false;
    }
}

template <typename CentroidT>
void runPQCodeDistances(
        GpuResources* res,
        Tensor<float, 3, true>& pqCentroids,
        Tensor<float, 2, true>& queries,
        Tensor<CentroidT, 2, true>& coarseCentroids,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        NoTypeTensor<4, true>& outCodeDistances,
        bool useMMImplementation,
        bool l2Distance,
        bool useFloat16Lookup,
        cudaStream_t stream) {
    const auto numSubQuantizers = pqCentroids.getSize(0);
    const auto dimsPerSubQuantizer = pqCentroids.getSize(1);
    const auto codesPerSubQuantizer = pqCentroids.getSize(2);

    // Only a certain number of dimensions per sub quantizer are supported by
    // the specialized implementation. Every other value falls back to the
    // generalized MM implementation.
    if (!isSpecializedPQCodeDistanceDims(dimsPerSubQuantizer) ||
        useMMImplementation) {
        // Use the general purpose matrix multiplication implementation which
        // handles any number of sub-quantizers and dimensions per sub-quantizer
        runPQCodeDistancesMM<CentroidT>(
                res,
                pqCentroids,
                queries,
                coarseCentroids,
                coarseDistances,
                coarseIndices,
                outCodeDistances,
                l2Distance,
                useFloat16Lookup,
                stream);
        return;
    }

    // FIXME: tune
    // Reuse of pq centroid data is based on both # of queries * nprobe,
    // and we should really be tiling in both dimensions
    constexpr int kQueriesPerBlock = 8;

    auto grid =
            dim3(utils::divUp(queries.getSize(0), kQueriesPerBlock),
                 numSubQuantizers);

    // Reserve one block of threads for double buffering
    // FIXME: probably impractical for large # of dims?
    int warpSize = getWarpSizeCurrentDevice();
    auto loadingThreads = utils::roundUp(dimsPerSubQuantizer, warpSize);
    auto block = dim3(codesPerSubQuantizer + loadingThreads);

    auto smem = (3 * dimsPerSubQuantizer) * sizeof(float) +
            coarseIndices.getSize(1) * sizeof(int);

#define RUN_CODE(DIMS, L2)                                               \
    do {                                                                 \
        if (useFloat16Lookup) {                                          \
            auto outCodeDistancesT = outCodeDistances.toTensor<half>();  \
                                                                         \
            pqCodeDistances<half, CentroidT, DIMS, L2>                   \
                    <<<grid, block, smem, stream>>>(                     \
                            queries,                                     \
                            kQueriesPerBlock,                            \
                            coarseCentroids,                             \
                            pqCentroids,                                 \
                            coarseIndices,                               \
                            outCodeDistancesT);                          \
        } else {                                                         \
            auto outCodeDistancesT = outCodeDistances.toTensor<float>(); \
                                                                         \
            pqCodeDistances<float, CentroidT, DIMS, L2>                  \
                    <<<grid, block, smem, stream>>>(                     \
                            queries,                                     \
                            kQueriesPerBlock,                            \
                            coarseCentroids,                             \
                            pqCentroids,                                 \
                            coarseIndices,                               \
                            outCodeDistancesT);                          \
        }                                                                \
    } while (0)

#define CODE_L2(DIMS)              \
    do {                           \
        if (l2Distance) {          \
            RUN_CODE(DIMS, true);  \
        } else {                   \
            RUN_CODE(DIMS, false); \
        }                          \
    } while (0)

    switch (dimsPerSubQuantizer) {
        case 1:
            CODE_L2(1);
            break;
        case 2:
            CODE_L2(2);
            break;
        case 3:
            CODE_L2(3);
            break;
        case 4:
            CODE_L2(4);
            break;
        case 5:
            CODE_L2(5);
            break;
        case 6:
            CODE_L2(6);
            break;
        case 8:
            CODE_L2(8);
            break;
        case 10:
            CODE_L2(10);
            break;
        case 12:
            CODE_L2(12);
            break;
        case 16:
            CODE_L2(16);
            break;
        case 20:
            CODE_L2(20);
            break;
        case 24:
            CODE_L2(24);
            break;
        case 28:
            CODE_L2(28);
            break;
        case 32:
            CODE_L2(32);
            break;
        default:
            // This should not be reached, we should fall back to the MM
            // implementation
            FAISS_ASSERT(false);
            break;
    }

#undef RUN_CODE
#undef CODE_L2

    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
