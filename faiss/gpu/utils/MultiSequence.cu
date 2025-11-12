/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Heap-inl.cuh>
#include <faiss/gpu/utils/MultiSequence.cuh>
#include <iostream>
#include <limits>

namespace faiss {
namespace gpu {

template <typename T, typename TVec2>
__device__ inline void toMultiIndex(T& index1, T& index2, TVec2& multiIndex) {
    multiIndex.x = index1;
    multiIndex.y = index2;
}

template <typename IndexT, typename MultiIndexT>
__device__ inline void toMultiIndex(
        int& codebookSize,
        IndexT& index1,
        IndexT& index2,
        MultiIndexT& multiIndex) {
    multiIndex = index1 + index2 * (MultiIndexT)codebookSize;
}

template <typename T, typename TVec2>
__device__ inline void multiSequence_thread(
        float* heapValues,
        ushort2* heapIds,
        unsigned short* traversed,
        unsigned w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
    unsigned short heapSize = 0;

    for (unsigned i = 0; i < w; i++) {
        traversed[i] = 0;
    }

    dr[0] = d1[0] + d2[0];
    toMultiIndex<T, TVec2>(i1[0], i2[0], ir[0]);
    unsigned short i = 0, j = 0;
    for (unsigned currentW = 1; currentW < w; currentW++) {
        traversed[i] = j + 1;

        if (traversed[i + 1] == j) {
            heap_push<unsigned short, ushort2>(
                    heapSize, heapValues, heapIds, d1[i + 1] + d2[j], i + 1, j);
        }
        if (i == 0 || traversed[i - 1] > j + 1) {
            heap_push<unsigned short, ushort2>(
                    heapSize, heapValues, heapIds, d1[i] + d2[j + 1], i, j + 1);
        }

        dr[currentW] = heapValues[0];
        i = heapIds[0].x;
        j = heapIds[0].y;
        toMultiIndex<T, TVec2>(i1[i], i2[j], ir[currentW]);
        heap_pop<unsigned short, ushort2>(heapSize, heapValues, heapIds);
    }
}

template <typename T, typename TVec2>
__device__ inline void multiSequence_thread(
        float* heapValues,
        ushort2* heapIds,
        unsigned short* traversed,
        unsigned w,
        unsigned short inputLength,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
    unsigned short heapSize = 0;

    for (unsigned i = 0; i < inputLength; i++) {
        traversed[i] = 0;
    }

    dr[0] = d1[0] + d2[0];
    toMultiIndex<T, TVec2>(i1[0], i2[0], ir[0]);
    unsigned short i = 0, j = 0;
    for (unsigned short currentW = 1; currentW < w; currentW++) {
        traversed[i] = j + 1;

        if (i < inputLength - 1 && (traversed[i + 1] == j)) {
            heap_push<unsigned short, ushort2>(
                    heapSize, heapValues, heapIds, d1[i + 1] + d2[j], i + 1, j);
        }
        if (j < inputLength - 1 && (i == 0 || traversed[i - 1] > j + 1)) {
            heap_push<unsigned short, ushort2>(
                    heapSize, heapValues, heapIds, d1[i] + d2[j + 1], i, j + 1);
        }

        dr[currentW] = heapValues[0];
        i = heapIds[0].x;
        j = heapIds[0].y;
        toMultiIndex<T, TVec2>(i1[i], i2[j], ir[currentW]);
        heap_pop<unsigned short, ushort2>(heapSize, heapValues, heapIds);
    }
}

template <typename T, typename MultiIndexT>
__device__ inline void multiSequence_thread(
        float* heapValues,
        ushort2* heapIds,
        unsigned short* traversed,
        unsigned w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        int codebookSize,
        MultiIndexT* ir) {
    unsigned short heapSize = 0;

    for (unsigned i = 0; i < w; i++) {
        traversed[i] = 0;
    }

    dr[0] = d1[0] + d2[0];
    toMultiIndex<T, MultiIndexT>(codebookSize, i1[0], i2[0], ir[0]);
    unsigned short i = 0, j = 0;
    for (unsigned currentW = 1; currentW < w; currentW++) {
        traversed[i] = j + 1;

        if (traversed[i + 1] == j) {
            heap_push<unsigned short, ushort2>(
                    heapSize, heapValues, heapIds, d1[i + 1] + d2[j], i + 1, j);
        }
        if (i == 0 || traversed[i - 1] > j + 1) {
            heap_push<unsigned short, ushort2>(
                    heapSize, heapValues, heapIds, d1[i] + d2[j + 1], i, j + 1);
        }

        dr[currentW] = heapValues[0];
        i = heapIds[0].x;
        j = heapIds[0].y;
        toMultiIndex<T, MultiIndexT>(codebookSize, i1[i], i2[j], ir[currentW]);
        heap_pop<unsigned short, ushort2>(heapSize, heapValues, heapIds);
    }
}

template <typename T, typename MultiIndexT>
__device__ inline void multiSequence_thread(
        float* heapValues,
        ushort2* heapIds,
        unsigned short* traversed,
        unsigned w,
        unsigned short inputLength,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        int codebookSize,
        MultiIndexT* ir) {
    unsigned short heapSize = 0;

    for (unsigned i = 0; i < inputLength; i++) {
        traversed[i] = 0;
    }

    dr[0] = d1[0] + d2[0];
    toMultiIndex<T, MultiIndexT>(codebookSize, i1[0], i2[0], ir[0]);
    unsigned short i = 0, j = 0;
    for (unsigned short currentW = 1; currentW < w; currentW++) {
        traversed[i] = j + 1;

        if (i < inputLength - 1 && (traversed[i + 1] == j)) {
            heap_push<unsigned short, ushort2>(
                    heapSize, heapValues, heapIds, d1[i + 1] + d2[j], i + 1, j);
        }
        if (j < inputLength - 1 && (i == 0 || traversed[i - 1] > j + 1)) {
            heap_push<unsigned short, ushort2>(
                    heapSize, heapValues, heapIds, d1[i] + d2[j + 1], i, j + 1);
        }

        dr[currentW] = heapValues[0];
        i = heapIds[0].x;
        j = heapIds[0].y;
        toMultiIndex<T, MultiIndexT>(codebookSize, i1[i], i2[j], ir[currentW]);
        heap_pop<unsigned short, ushort2>(heapSize, heapValues, heapIds);
    }
}

template <unsigned MaxInLength, typename T, typename TVec2>
__global__ void multiSequence_local(
        unsigned numThreadsGrid,
        unsigned w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
    unsigned threadIdxGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdxGrid < numThreadsGrid) {
        float heapValues[MaxInLength];
        ushort2 heapIds[MaxInLength];
        unsigned short traversed[MaxInLength];

        unsigned stride = w * threadIdxGrid;
        float* threadD1 = d1 + stride;
        float* threadD2 = d2 + stride;
        T* threadI1 = i1 + stride;
        T* threadI2 = i2 + stride;
        float* threadDr = dr + stride;
        TVec2* threadIr = ir + stride;

        multiSequence_thread<T, TVec2>(
                heapValues,
                heapIds,
                traversed,
                w,
                threadD1,
                threadI1,
                threadD2,
                threadI2,
                threadDr,
                threadIr);
    }
}

template <unsigned MaxInLength, typename T, typename TVec2>
__global__ void multiSequence_local(
        unsigned numThreadsGrid,
        unsigned w,
        unsigned short inputLength,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
    unsigned threadIdxGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdxGrid < numThreadsGrid) {
        float heapValues[MaxInLength];
        ushort2 heapIds[MaxInLength];
        unsigned short traversed[MaxInLength];

        unsigned inStride = inputLength * threadIdxGrid;
        unsigned outStride = w * threadIdxGrid;
        float* threadD1 = d1 + inStride;
        float* threadD2 = d2 + inStride;
        T* threadI1 = i1 + inStride;
        T* threadI2 = i2 + inStride;
        float* threadDr = dr + outStride;
        TVec2* threadIr = ir + outStride;

        multiSequence_thread<T, TVec2>(
                heapValues,
                heapIds,
                traversed,
                w,
                inputLength,
                threadD1,
                threadI1,
                threadD2,
                threadI2,
                threadDr,
                threadIr);
    }
}

template <unsigned MaxInLength, typename T, typename MultiIndexT>
__global__ void multiSequence_local(
        unsigned numThreadsGrid,
        unsigned w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        int codebookSize,
        MultiIndexT* ir) {
    unsigned threadIdxGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdxGrid < numThreadsGrid) {
        float heapValues[MaxInLength];
        ushort2 heapIds[MaxInLength];
        unsigned short traversed[MaxInLength];

        unsigned stride = w * threadIdxGrid;
        float* threadD1 = d1 + stride;
        float* threadD2 = d2 + stride;
        T* threadI1 = i1 + stride;
        T* threadI2 = i2 + stride;
        float* threadDr = dr + stride;
        MultiIndexT* threadIr = ir + stride;

        multiSequence_thread<T, MultiIndexT>(
                heapValues,
                heapIds,
                traversed,
                w,
                threadD1,
                threadI1,
                threadD2,
                threadI2,
                threadDr,
                codebookSize,
                threadIr);
    }
}

template <unsigned MaxInLength, typename T, typename MultiIndexT>
__global__ void multiSequence_local(
        unsigned numThreadsGrid,
        unsigned w,
        unsigned short inputLength,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        int codebookSize,
        MultiIndexT* ir) {
    unsigned threadIdxGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdxGrid < numThreadsGrid) {
        float heapValues[MaxInLength];
        ushort2 heapIds[MaxInLength];
        unsigned short traversed[MaxInLength];

        unsigned inStride = inputLength * threadIdxGrid;
        unsigned outStride = w * threadIdxGrid;
        float* threadD1 = d1 + inStride;
        float* threadD2 = d2 + inStride;
        T* threadI1 = i1 + inStride;
        T* threadI2 = i2 + inStride;
        float* threadDr = dr + outStride;
        MultiIndexT* threadIr = ir + outStride;

        multiSequence_thread<T, MultiIndexT>(
                heapValues,
                heapIds,
                traversed,
                w,
                inputLength,
                threadD1,
                threadI1,
                threadD2,
                threadI2,
                threadDr,
                codebookSize,
                threadIr);
    }
}

// performs much faster than local, but has memory limit
template <typename T, typename TVec2>
__global__ void multiSequence_shared(
        unsigned numThreadsGrid,
        unsigned w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
    extern __shared__ char buffer[];

    unsigned threadIdxGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdxGrid < numThreadsGrid) {
        unsigned bufferStride =
                ((sizeof(float) + sizeof(ushort2) + sizeof(unsigned short)) *
                 w) *
                threadIdx.x;
        float* heapValues = (float*)&buffer[bufferStride];
        ushort2* heapIds = (ushort2*)&buffer[bufferStride + sizeof(float) * w];
        unsigned short* traversed = (unsigned short*)&buffer
                [bufferStride + (sizeof(float) + sizeof(ushort2)) * w];

        unsigned stride = w * threadIdxGrid;
        float* threadD1 = d1 + stride;
        float* threadD2 = d2 + stride;
        T* threadI1 = i1 + stride;
        T* threadI2 = i2 + stride;
        float* threadDr = dr + stride;
        TVec2* threadIr = ir + stride;

        multiSequence_thread<T, TVec2>(
                heapValues,
                heapIds,
                traversed,
                w,
                threadD1,
                threadI1,
                threadD2,
                threadI2,
                threadDr,
                threadIr);
    }
}

template <typename T, typename TVec2>
__global__ void multiSequence_shared(
        unsigned numThreadsGrid,
        unsigned w,
        unsigned short inputLength,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
    extern __shared__ char buffer[];

    unsigned threadIdxGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdxGrid < numThreadsGrid) {
        unsigned bufferStride =
                ((sizeof(float) + sizeof(ushort2) + sizeof(unsigned short)) *
                 inputLength) *
                threadIdx.x;
        float* heapValues = (float*)&buffer[bufferStride];
        ushort2* heapIds =
                (ushort2*)&buffer[bufferStride + sizeof(float) * inputLength];
        unsigned short* traversed = (unsigned short*)&buffer
                [bufferStride +
                 (sizeof(float) + sizeof(ushort2)) * inputLength];

        unsigned inStride = inputLength * threadIdxGrid;
        unsigned outStride = w * threadIdxGrid;
        float* threadD1 = d1 + inStride;
        float* threadD2 = d2 + inStride;
        T* threadI1 = i1 + inStride;
        T* threadI2 = i2 + inStride;
        float* threadDr = dr + outStride;
        TVec2* threadIr = ir + outStride;

        multiSequence_thread<T, TVec2>(
                heapValues,
                heapIds,
                traversed,
                w,
                inputLength,
                threadD1,
                threadI1,
                threadD2,
                threadI2,
                threadDr,
                threadIr);
    }
}

template <typename T, typename MultiIndexT>
__global__ void multiSequence_shared(
        unsigned numThreadsGrid,
        unsigned w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        int codebookSize,
        MultiIndexT* ir) {
    extern __shared__ char buffer[];
    unsigned bufferStride =
            ((sizeof(float) + sizeof(ushort2) + sizeof(unsigned short)) * w) *
            threadIdx.x;
    float* heapValues = (float*)&buffer[bufferStride];
    ushort2* heapIds = (ushort2*)&buffer[bufferStride + sizeof(float) * w];
    unsigned short* traversed = (unsigned short*)&buffer
            [bufferStride + (sizeof(float) + sizeof(ushort2)) * w];

    unsigned threadIdxGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdxGrid < numThreadsGrid) {
        unsigned stride = w * threadIdxGrid;
        float* threadD1 = d1 + stride;
        float* threadD2 = d2 + stride;
        T* threadI1 = i1 + stride;
        T* threadI2 = i2 + stride;
        float* threadDr = dr + stride;
        MultiIndexT* threadIr = ir + stride;

        multiSequence_thread<T, MultiIndexT>(
                heapValues,
                heapIds,
                traversed,
                w,
                threadD1,
                threadI1,
                threadD2,
                threadI2,
                threadDr,
                codebookSize,
                threadIr);
    }
}

template <typename T, typename MultiIndexT>
__global__ void multiSequence_shared(
        unsigned numThreadsGrid,
        unsigned w,
        unsigned short inputLength,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        int codebookSize,
        MultiIndexT* ir) {
    extern __shared__ char buffer[];

    unsigned threadIdxGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdxGrid < numThreadsGrid) {
        unsigned bufferStride =
                ((sizeof(float) + sizeof(ushort2) + sizeof(unsigned short)) *
                 inputLength) *
                threadIdx.x;
        float* heapValues = (float*)&buffer[bufferStride];
        ushort2* heapIds =
                (ushort2*)&buffer[bufferStride + sizeof(float) * inputLength];
        unsigned short* traversed = (unsigned short*)&buffer
                [bufferStride +
                 (sizeof(float) + sizeof(ushort2)) * inputLength];

        unsigned inStride = inputLength * threadIdxGrid;
        unsigned outStride = w * threadIdxGrid;
        float* threadD1 = d1 + inStride;
        float* threadD2 = d2 + inStride;
        T* threadI1 = i1 + inStride;
        T* threadI2 = i2 + inStride;
        float* threadDr = dr + outStride;
        MultiIndexT* threadIr = ir + outStride;

        multiSequence_thread<T, MultiIndexT>(
                heapValues,
                heapIds,
                traversed,
                w,
                inputLength,
                threadD1,
                threadI1,
                threadD2,
                threadI2,
                threadDr,
                codebookSize,
                threadIr);
    }
}

template <typename T, typename TVec2>
void multiSequenceLocalKernel(
        int numOfBlocks,
        int blockSize,
        cudaStream_t stream,
        int numThreadsGrid,
        int w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
#define RUN_MS(MaxInLength)                                         \
    do {                                                            \
        multiSequence_local<MaxInLength, T, TVec2>                  \
                <<<numOfBlocks, blockSize, 0, stream>>>(            \
                        numThreadsGrid, w, d1, i1, d2, i2, dr, ir); \
    } while (0)

    if (w == 1) {
        RUN_MS(1);
    } else if (w == 2) {
        RUN_MS(2);
    } else if (w <= 4) {
        RUN_MS(4);
    } else if (w <= 8) {
        RUN_MS(8);
    } else if (w <= 16) {
        RUN_MS(16);
    } else if (w <= 32) {
        RUN_MS(32);
    } else if (w <= 64) {
        RUN_MS(64);
    } else if (w <= 128) {
        RUN_MS(128);
    } else if (w <= 256) {
        RUN_MS(256);
    } else if (w <= 512) {
        RUN_MS(512);
    } else if (w <= 1024) {
        RUN_MS(1024);
    } else if (w <= GPU_MAX_SELECTION_K) {
        RUN_MS(GPU_MAX_SELECTION_K);
    } else {
        FAISS_ASSERT(false);
    }

#undef RUN_MS
}

template <typename T, typename TVec2>
void multiSequenceLocalKernel(
        int numOfBlocks,
        int blockSize,
        cudaStream_t stream,
        int numThreadsGrid,
        int w,
        unsigned short inputLength,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
#define RUN_MS(MaxInLength)                              \
    do {                                                 \
        multiSequence_local<MaxInLength, T, TVec2>       \
                <<<numOfBlocks, blockSize, 0, stream>>>( \
                        numThreadsGrid,                  \
                        w,                               \
                        inputLength,                     \
                        d1,                              \
                        i1,                              \
                        d2,                              \
                        i2,                              \
                        dr,                              \
                        ir);                             \
    } while (0)

    if (inputLength == 1) {
        RUN_MS(1);
    } else if (inputLength == 2) {
        RUN_MS(2);
    } else if (inputLength <= 4) {
        RUN_MS(4);
    } else if (inputLength <= 8) {
        RUN_MS(8);
    } else if (inputLength <= 16) {
        RUN_MS(16);
    } else if (inputLength <= 32) {
        RUN_MS(32);
    } else if (inputLength <= 64) {
        RUN_MS(64);
    } else if (inputLength <= 128) {
        RUN_MS(128);
    } else if (inputLength <= 256) {
        RUN_MS(256);
    } else if (inputLength <= 512) {
        RUN_MS(512);
    } else if (inputLength <= 1024) {
        RUN_MS(1024);
    } else if (inputLength <= GPU_MAX_SELECTION_K) {
        RUN_MS(GPU_MAX_SELECTION_K);
    } else {
        FAISS_ASSERT(false);
    }

#undef RUN_MS
}

template <typename T, typename MultiIndexT>
void multiSequenceLocalKernel(
        int numOfBlocks,
        int blockSize,
        cudaStream_t stream,
        int numThreadsGrid,
        int w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        int codebookSize,
        MultiIndexT* ir) {
#define RUN_MS(MaxInLength)                              \
    do {                                                 \
        multiSequence_local<MaxInLength, T, MultiIndexT> \
                <<<numOfBlocks, blockSize, 0, stream>>>( \
                        numThreadsGrid,                  \
                        w,                               \
                        d1,                              \
                        i1,                              \
                        d2,                              \
                        i2,                              \
                        dr,                              \
                        codebookSize,                    \
                        ir);                             \
    } while (0)

    if (w == 1) {
        RUN_MS(1);
    } else if (w == 2) {
        RUN_MS(2);
    } else if (w <= 4) {
        RUN_MS(4);
    } else if (w <= 8) {
        RUN_MS(8);
    } else if (w <= 16) {
        RUN_MS(16);
    } else if (w <= 32) {
        RUN_MS(32);
    } else if (w <= 64) {
        RUN_MS(64);
    } else if (w <= 128) {
        RUN_MS(128);
    } else if (w <= 256) {
        RUN_MS(256);
    } else if (w <= 512) {
        RUN_MS(512);
    } else if (w <= 1024) {
        RUN_MS(1024);
    } else if (w <= GPU_MAX_SELECTION_K) {
        RUN_MS(GPU_MAX_SELECTION_K);
    } else {
        FAISS_ASSERT(false);
    }

#undef RUN_MS
}

template <typename T, typename MultiIndexT>
void multiSequenceLocalKernel(
        int numOfBlocks,
        int blockSize,
        cudaStream_t stream,
        int numThreadsGrid,
        int w,
        unsigned short inputLength,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        int codebookSize,
        MultiIndexT* ir) {
#define RUN_MS(MaxInLength)                              \
    do {                                                 \
        multiSequence_local<MaxInLength, T, MultiIndexT> \
                <<<numOfBlocks, blockSize, 0, stream>>>( \
                        numThreadsGrid,                  \
                        w,                               \
                        inputLength,                     \
                        d1,                              \
                        i1,                              \
                        d2,                              \
                        i2,                              \
                        dr,                              \
                        codebookSize,                    \
                        ir);                             \
    } while (0)

    if (inputLength == 1) {
        RUN_MS(1);
    } else if (inputLength == 2) {
        RUN_MS(2);
    } else if (inputLength <= 4) {
        RUN_MS(4);
    } else if (inputLength <= 8) {
        RUN_MS(8);
    } else if (inputLength <= 16) {
        RUN_MS(16);
    } else if (inputLength <= 32) {
        RUN_MS(32);
    } else if (inputLength <= 64) {
        RUN_MS(64);
    } else if (inputLength <= 128) {
        RUN_MS(128);
    } else if (inputLength <= 256) {
        RUN_MS(256);
    } else if (inputLength <= 512) {
        RUN_MS(512);
    } else if (inputLength <= 1024) {
        RUN_MS(1024);
    } else if (inputLength <= GPU_MAX_SELECTION_K) {
        RUN_MS(GPU_MAX_SELECTION_K);
    } else {
        FAISS_ASSERT(false);
    }

#undef RUN_MS
}

template <typename T, typename TVec2>
void chooseMultiSequence2Params(
        const int inLength,
        bool& useSharedMemory,
        int& blockSize) {
    int maxSharedMemoryPerBlock = getMaxSharedMemPerBlockCurrentDevice();
    int sharedMemoryPerThread =
            (sizeof(float) + sizeof(TVec2) + sizeof(T)) * inLength;
    int maxBlockSize = maxSharedMemoryPerBlock / sharedMemoryPerThread;
    maxBlockSize = utils::nextHighestPowerOf2(maxBlockSize) / 2;
    useSharedMemory = maxBlockSize >= 64;
    if (useSharedMemory) {
        blockSize = std::min(maxBlockSize, 256);
    } else {
        blockSize = 256;
    }
}

template <typename T, typename TVec2>
void runMultiSequence2T(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<T, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<TVec2, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream) {
    constexpr int NUM_CODEBOOKS = 2;

    FAISS_ASSERT(inDistances.getSize(0) == NUM_CODEBOOKS);
    FAISS_ASSERT(inIndices.getSize(0) == NUM_CODEBOOKS);

    FAISS_ASSERT(numQueries <= inDistances.getSize(1));
    FAISS_ASSERT(numQueries <= inIndices.getSize(1));
    FAISS_ASSERT(numQueries <= outDistances.getSize(0));
    FAISS_ASSERT(numQueries <= outIndices.getSize(0));

    FAISS_ASSERT(inLength <= inDistances.getSize(2));
    FAISS_ASSERT(inLength <= inIndices.getSize(2));

    FAISS_ASSERT(w <= outDistances.getSize(1));
    FAISS_ASSERT(w <= outIndices.getSize(1));

    FAISS_ASSERT(inLength <= std::numeric_limits<unsigned short>::max());
    FAISS_ASSERT(w <= inLength * inLength);
    if (w <= inLength)
        FAISS_ASSERT(w <= GPU_MAX_SELECTION_K);
    else
        FAISS_ASSERT(w <= GPU_MAX_SELECTION_K * GPU_MAX_SELECTION_K);

    bool useSharedMemory;
    int blockSize;

    if (w <= inLength) {
        chooseMultiSequence2Params<T, TVec2>(w, useSharedMemory, blockSize);
    } else {
        chooseMultiSequence2Params<T, TVec2>(
                inLength, useSharedMemory, blockSize);
    }

    blockSize = std::min(blockSize, numQueries);
    const int numOfBlocks = (numQueries + blockSize - 1) / blockSize;
    const int numThreadsGrid = blockSize * (int)(numQueries / blockSize) +
            (numQueries % blockSize);

    auto inDistances1View = inDistances[0].view();
    auto inDistances2View = inDistances[1].view();
    auto inIndices1View = inIndices[0].view();
    auto inIndices2View = inIndices[1].view();

    if (useSharedMemory) {
        if (w <= inLength) {
            const int smem = blockSize * w *
                    (sizeof(float) + sizeof(ushort2) + sizeof(unsigned short));
            FAISS_ASSERT(smem <= getMaxSharedMemPerBlockCurrentDevice());
            multiSequence_shared<T, TVec2>
                    <<<numOfBlocks, blockSize, smem, stream>>>(
                            numThreadsGrid,
                            w,
                            inDistances1View.data(),
                            inIndices1View.data(),
                            inDistances2View.data(),
                            inIndices2View.data(),
                            outDistances.data(),
                            outIndices.data());
        } else {
            const int smem = blockSize * inLength *
                    (sizeof(float) + sizeof(ushort2) + sizeof(unsigned short));
            FAISS_ASSERT(smem <= getMaxSharedMemPerBlockCurrentDevice());
            multiSequence_shared<T, TVec2>
                    <<<numOfBlocks, blockSize, smem, stream>>>(
                            numThreadsGrid,
                            w,
                            inLength,
                            inDistances1View.data(),
                            inIndices1View.data(),
                            inDistances2View.data(),
                            inIndices2View.data(),
                            outDistances.data(),
                            outIndices.data());
        }
    } else {
        if (w <= inLength) {
            multiSequenceLocalKernel<T, TVec2>(
                    numOfBlocks,
                    blockSize,
                    stream,
                    numThreadsGrid,
                    w,
                    inDistances1View.data(),
                    inIndices1View.data(),
                    inDistances2View.data(),
                    inIndices2View.data(),
                    outDistances.data(),
                    outIndices.data());
        } else {
            multiSequenceLocalKernel<T, TVec2>(
                    numOfBlocks,
                    blockSize,
                    stream,
                    numThreadsGrid,
                    w,
                    inLength,
                    inDistances1View.data(),
                    inIndices1View.data(),
                    inDistances2View.data(),
                    inIndices2View.data(),
                    outDistances.data(),
                    outIndices.data());
        }
    }
}

template <typename T, typename MultiIndexT>
void runMultiSequence2T(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<T, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        const int codebookSize,
        Tensor<MultiIndexT, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream) {
    constexpr int NUM_CODEBOOKS = 2;

    FAISS_ASSERT(inDistances.getSize(0) == NUM_CODEBOOKS);
    FAISS_ASSERT(inIndices.getSize(0) == NUM_CODEBOOKS);

    FAISS_ASSERT(numQueries <= inDistances.getSize(1));
    FAISS_ASSERT(numQueries <= inIndices.getSize(1));
    FAISS_ASSERT(numQueries <= outDistances.getSize(0));
    FAISS_ASSERT(numQueries <= outIndices.getSize(0));

    FAISS_ASSERT(inLength <= inDistances.getSize(2));
    FAISS_ASSERT(inLength <= inIndices.getSize(2));

    FAISS_ASSERT(w <= outDistances.getSize(1));
    FAISS_ASSERT(w <= outIndices.getSize(1));

    FAISS_ASSERT(inLength <= std::numeric_limits<unsigned short>::max());
    FAISS_ASSERT(w <= inLength * inLength);
    if (w <= inLength)
        FAISS_ASSERT(w <= GPU_MAX_SELECTION_K);
    else
        FAISS_ASSERT(w <= GPU_MAX_SELECTION_K * GPU_MAX_SELECTION_K);

    bool useSharedMemory;
    int blockSize;

    if (w <= inLength) {
        chooseMultiSequence2Params<T, MultiIndexT>(
                w, useSharedMemory, blockSize);
    } else {
        chooseMultiSequence2Params<T, MultiIndexT>(
                inLength, useSharedMemory, blockSize);
    }

    blockSize = std::min(blockSize, numQueries);
    const int numOfBlocks = (numQueries + blockSize - 1) / blockSize;
    const int numThreadsGrid = blockSize * (int)(numQueries / blockSize) +
            (numQueries % blockSize);

    auto inDistances1View = inDistances[0].view();
    auto inDistances2View = inDistances[1].view();
    auto inIndices1View = inIndices[0].view();
    auto inIndices2View = inIndices[1].view();

    if (useSharedMemory) {
        if (w <= inLength) {
            const int smem = blockSize * w *
                    (sizeof(float) + sizeof(ushort2) + sizeof(unsigned short));
            FAISS_ASSERT(smem <= getMaxSharedMemPerBlockCurrentDevice());
            multiSequence_shared<T, MultiIndexT>
                    <<<numOfBlocks, blockSize, smem, stream>>>(
                            numThreadsGrid,
                            w,
                            inDistances1View.data(),
                            inIndices1View.data(),
                            inDistances2View.data(),
                            inIndices2View.data(),
                            outDistances.data(),
                            codebookSize,
                            outIndices.data());
        } else {
            const int smem = blockSize * inLength *
                    (sizeof(float) + sizeof(ushort2) + sizeof(unsigned short));
            FAISS_ASSERT(smem <= getMaxSharedMemPerBlockCurrentDevice());
            multiSequence_shared<T, MultiIndexT>
                    <<<numOfBlocks, blockSize, smem, stream>>>(
                            numThreadsGrid,
                            w,
                            inLength,
                            inDistances1View.data(),
                            inIndices1View.data(),
                            inDistances2View.data(),
                            inIndices2View.data(),
                            outDistances.data(),
                            codebookSize,
                            outIndices.data());
        }
    } else {
        if (w <= inLength) {
            multiSequenceLocalKernel<T, MultiIndexT>(
                    numOfBlocks,
                    blockSize,
                    stream,
                    numThreadsGrid,
                    w,
                    inDistances1View.data(),
                    inIndices1View.data(),
                    inDistances2View.data(),
                    inIndices2View.data(),
                    outDistances.data(),
                    codebookSize,
                    outIndices.data());
        } else {
            multiSequenceLocalKernel<T, MultiIndexT>(
                    numOfBlocks,
                    blockSize,
                    stream,
                    numThreadsGrid,
                    w,
                    inLength,
                    inDistances1View.data(),
                    inIndices1View.data(),
                    inDistances2View.data(),
                    inIndices2View.data(),
                    outDistances.data(),
                    codebookSize,
                    outIndices.data());
        }
    }
}

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<ushort, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<ushort2, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream) {
    runMultiSequence2T<ushort, ushort2>(
            numQueries,
            inLength,
            w,
            inDistances,
            inIndices,
            outDistances,
            outIndices,
            res,
            stream);
}

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<ushort, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<ushort2, 2, true>& outIndices,
        GpuResources* res) {
    runMultiSequence2T<ushort, ushort2>(
            numQueries,
            inLength,
            w,
            inDistances,
            inIndices,
            outDistances,
            outIndices,
            res,
            res->getDefaultStreamCurrentDevice());
}

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<idx_t, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<int2, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream) {
    runMultiSequence2T<idx_t, int2>(
            numQueries,
            inLength,
            w,
            inDistances,
            inIndices,
            outDistances,
            outIndices,
            res,
            stream);
}

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<idx_t, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        Tensor<int2, 2, true>& outIndices,
        GpuResources* res) {
    runMultiSequence2T<idx_t, int2>(
            numQueries,
            inLength,
            w,
            inDistances,
            inIndices,
            outDistances,
            outIndices,
            res,
            res->getDefaultStreamCurrentDevice());
}

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<ushort, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        const int codebookSize,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream) {
    runMultiSequence2T<ushort, idx_t>(
            numQueries,
            inLength,
            w,
            inDistances,
            inIndices,
            outDistances,
            codebookSize,
            outIndices,
            res,
            stream);
}

void runMultiSequence2(
        const int numQueries,
        const int inLength,
        const int w,
        Tensor<float, 3, true>& inDistances,
        Tensor<ushort, 3, true>& inIndices,
        Tensor<float, 2, true>& outDistances,
        const int codebookSize,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res) {
    runMultiSequence2T<ushort, idx_t>(
            numQueries,
            inLength,
            w,
            inDistances,
            inIndices,
            outDistances,
            codebookSize,
            outIndices,
            res,
            res->getDefaultStreamCurrentDevice());
}

} // namespace gpu
} // namespace faiss
