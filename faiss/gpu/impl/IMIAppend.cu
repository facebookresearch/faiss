/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/IMIAppend.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/MultiSequence-inl.cuh>
#include <faiss/gpu/utils/WarpPackedBits.cuh>
#include <faiss/gpu/utils/WarpShuffles.cuh>

namespace faiss {
namespace gpu {

__global__ void imiUpdateStartOffsets(
        Tensor<unsigned int, 1, true> listStartOffsets,
        Tensor<unsigned int, 1, true> newlistStartOffsets) {
    int vec = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec >= newlistStartOffsets.getSize(0)) {
        return;
    }

    unsigned int offset = newlistStartOffsets[vec];
    listStartOffsets[vec] = offset;
}

void runIMIUpdateStartOffsets(
        Tensor<unsigned int, 1, true>& listStartOffsets,
        Tensor<unsigned int, 1, true>& newlistStartOffsets,
        cudaStream_t stream) {
    FAISS_ASSERT(newlistStartOffsets.getSize(0) == listStartOffsets.getSize(0));

    int num = newlistStartOffsets.getSize(0);
    int threads = std::min(num, getMaxThreadsCurrentDevice());
    int blocks = utils::divUp(num, threads);

    imiUpdateStartOffsets<<<blocks, threads, 0, stream>>>(
            listStartOffsets, newlistStartOffsets);

    CUDA_TEST_ERROR();
}

// Appends new indices for vectors being added to the IMI indices lists
__global__ void imiIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true> listIds,
        Tensor<int, 1, true> listOffset,
        Tensor<idx_t, 1, true> indices,
        IndicesOptions opt,
        void** listIndices) {
    int vec = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec >= listIds.getSize(0)) {
        return;
    }

    ushort2 listId2 = listIds[vec];
    int listId = toMultiIndex<ushort, int>(codebookSize, listId2.x, listId2.y);
    int offset = listOffset[vec];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || offset == -1) {
        return;
    }

    auto index = indices[vec];

    if (opt == INDICES_32_BIT) {
        // FIXME: there could be overflow here, but where should we check this?
        ((int*)listIndices[listId])[offset] = (int)index;
    } else if (opt == INDICES_64_BIT) {
        ((idx_t*)listIndices[listId])[offset] = index;
    }
}

// Appends new indices for vectors being added to the IMI indices lists
template <typename T>
__global__ void imiIndicesAppendPointer(
        int codebookSize,
        Tensor<ushort2, 1, true> listIds,
        Tensor<int, 1, true> listOffset,
        Tensor<idx_t, 1, true> indices,
        Tensor<T*, 1, true> listIndices) {
    int vec = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec >= listIds.getSize(0)) {
        return;
    }

    ushort2 listId2 = listIds[vec];
    int listId = toMultiIndex<ushort, int>(codebookSize, listId2.x, listId2.y);
    int offset = listOffset[vec];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || offset == -1) {
        return;
    }

    auto index = indices[vec];

    listIndices[listId][offset] = (T)index;
}

// Appends new indices for vectors being added to the IMI indices lists
template <typename T>
__global__ void imiIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true> listIds,
        Tensor<int, 1, true> listOffset,
        Tensor<idx_t, 1, true> indices,
        Tensor<T, 1, true> listIndices,
        Tensor<unsigned int, 1, true> listStartOffsets) {
    int vec = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec >= listIds.getSize(0)) {
        return;
    }

    ushort2 listId2 = listIds[vec];
    int listId = toMultiIndex<ushort, int>(codebookSize, listId2.x, listId2.y);
    int offset = listOffset[vec];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || offset == -1) {
        return;
    }

    auto index = indices[vec];
    auto startOffset = listStartOffsets[listId];

    listIndices[startOffset + offset] = (T)index;
}

void runIMIIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        thrust::device_vector<void*>& listIndices,
        cudaStream_t stream) {
    FAISS_ASSERT(
            opt == INDICES_CPU || opt == INDICES_IVF || opt == INDICES_32_BIT ||
            opt == INDICES_64_BIT);

    if (opt != INDICES_CPU && opt != INDICES_IVF) {
        int num = listIds.getSize(0);
        int threads = std::min(num, getMaxThreadsCurrentDevice());
        int blocks = utils::divUp(num, threads);

        imiIndicesAppend<<<blocks, threads, 0, stream>>>(
                codebookSize,
                listIds,
                listOffset,
                indices,
                opt,
                listIndices.data().get());

        CUDA_TEST_ERROR();
    }
}

template <typename T>
void runIMIIndicesAppendTPointer(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        Tensor<T*, 1, true>& listIndices,
        cudaStream_t stream) {
    FAISS_ASSERT(
            opt == INDICES_CPU || opt == INDICES_IVF || opt == INDICES_32_BIT ||
            opt == INDICES_64_BIT);

    if (opt != INDICES_CPU && opt != INDICES_IVF) {
        int num = listIds.getSize(0);
        int threads = std::min(num, getMaxThreadsCurrentDevice());
        int blocks = utils::divUp(num, threads);

        imiIndicesAppendPointer<<<blocks, threads, 0, stream>>>(
                codebookSize, listIds, listOffset, indices, listIndices);

        CUDA_TEST_ERROR();
    }
}

void runIMIIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        Tensor<int*, 1, true>& listIndices,
        cudaStream_t stream) {
    runIMIIndicesAppendTPointer<int>(
            codebookSize,
            listIds,
            listOffset,
            indices,
            opt,
            listIndices,
            stream);
}

void runIMIIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        Tensor<idx_t*, 1, true>& listIndices,
        cudaStream_t stream) {
    runIMIIndicesAppendTPointer<idx_t>(
            codebookSize,
            listIds,
            listOffset,
            indices,
            opt,
            listIndices,
            stream);
}

template <typename T>
void runIMIIndicesAppendT(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        Tensor<T, 1, true>& listIndices,
        Tensor<unsigned int, 1, true>& listStartOffsets,
        cudaStream_t stream) {
    FAISS_ASSERT(
            opt == INDICES_CPU || opt == INDICES_IVF || opt == INDICES_32_BIT ||
            opt == INDICES_64_BIT);

    if (opt != INDICES_CPU && opt != INDICES_IVF) {
        int num = listIds.getSize(0);
        int threads = std::min(num, getMaxThreadsCurrentDevice());
        int blocks = utils::divUp(num, threads);

        imiIndicesAppend<<<blocks, threads, 0, stream>>>(
                codebookSize,
                listIds,
                listOffset,
                indices,
                listIndices,
                listStartOffsets);

        CUDA_TEST_ERROR();
    }
}

void runIMIIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        Tensor<int, 1, true>& listIndices,
        Tensor<unsigned int, 1, true>& listStartOffsets,
        cudaStream_t stream) {
    runIMIIndicesAppendT<int>(
            codebookSize,
            listIds,
            listOffset,
            indices,
            opt,
            listIndices,
            listStartOffsets,
            stream);
}

void runIMIIndicesAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        Tensor<idx_t, 1, true>& listIndices,
        Tensor<unsigned int, 1, true>& listStartOffsets,
        cudaStream_t stream) {
    runIMIIndicesAppendT<idx_t>(
            codebookSize,
            listIds,
            listOffset,
            indices,
            opt,
            listIndices,
            listStartOffsets,
            stream);
}

//
// IMI non-interleaved append
//

__global__ void imipqAppend(
        int codebookSize,
        Tensor<ushort2, 1, true> listIds,
        Tensor<int, 1, true> listOffset,
        Tensor<uint8_t, 2, true> encodings,
        void** listCodes) {
    int encodingToAdd = blockIdx.x * blockDim.x + threadIdx.x;

    if (encodingToAdd >= listIds.getSize(0)) {
        return;
    }

    ushort2 listId2 = listIds[encodingToAdd];
    int listId = toMultiIndex<ushort, int>(codebookSize, listId2.x, listId2.y);
    int vectorNumInList = listOffset[encodingToAdd];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || vectorNumInList == -1) {
        return;
    }

    auto encoding = encodings[encodingToAdd];

    // Layout with dimensions innermost
    uint8_t* codeStart = ((uint8_t*)listCodes[listId]) +
            vectorNumInList * encodings.getSize(1);

    // FIXME: stride with threads instead of single thread
    for (int i = 0; i < encodings.getSize(1); ++i) {
        codeStart[i] = encoding[i];
    }
}

__global__ void imipqAppend(
        int codebookSize,
        Tensor<ushort2, 1, true> listIds,
        Tensor<int, 1, true> listOffset,
        Tensor<uint8_t, 2, true> encodings,
        Tensor<uint8_t*, 1, true> listCodes) {
    int encodingToAdd = blockIdx.x * blockDim.x + threadIdx.x;

    if (encodingToAdd >= listIds.getSize(0)) {
        return;
    }

    ushort2 listId2 = listIds[encodingToAdd];
    int listId = toMultiIndex<ushort, int>(codebookSize, listId2.x, listId2.y);
    int vectorNumInList = listOffset[encodingToAdd];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || vectorNumInList == -1) {
        return;
    }

    auto encoding = encodings[encodingToAdd];

    // Layout with dimensions innermost
    uint8_t* codeStart =
            listCodes[listId] + vectorNumInList * encodings.getSize(1);

    // FIXME: stride with threads instead of single thread
    for (int i = 0; i < encodings.getSize(1); ++i) {
        codeStart[i] = encoding[i];
    }
}

__global__ void imipqAppend(
        int codebookSize,
        Tensor<ushort2, 1, true> listIds,
        Tensor<int, 1, true> listOffset,
        Tensor<uint8_t, 2, true> encodings,
        Tensor<uint8_t, 1, true, long> listCodes,
        Tensor<unsigned int, 1, true> listStartOffsets,
        int encodingNumBytes) {
    int encodingToAdd = blockIdx.x * blockDim.x + threadIdx.x;

    if (encodingToAdd >= listIds.getSize(0)) {
        return;
    }

    ushort2 listId2 = listIds[encodingToAdd];
    int listId = toMultiIndex<ushort, int>(codebookSize, listId2.x, listId2.y);
    int vectorNumInList = listOffset[encodingToAdd];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || vectorNumInList == -1) {
        return;
    }

    auto encoding = encodings[encodingToAdd];
    size_t startOffset = (size_t)listStartOffsets[listId] * encodingNumBytes;

    // Layout with dimensions innermost
    size_t codeStart = startOffset + vectorNumInList * encodings.getSize(1);

    // FIXME: stride with threads instead of single thread
    for (int i = 0; i < encodings.getSize(1); ++i) {
        listCodes[codeStart + i] = encoding[i];
    }
}

void runIMIPQAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<uint8_t, 2, true>& encodings,
        thrust::device_vector<void*>& listCodes,
        cudaStream_t stream) {
    int threads =
            std::min(listIds.getSize(0), (idx_t)getMaxThreadsCurrentDevice());
    int blocks = utils::divUp(listIds.getSize(0), threads);

    imipqAppend<<<threads, blocks, 0, stream>>>(
            codebookSize,
            listIds,
            listOffset,
            encodings,
            listCodes.data().get());

    CUDA_TEST_ERROR();
}

void runIMIPQAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<uint8_t, 2, true>& encodings,
        Tensor<uint8_t*, 1, true>& listCodes,
        cudaStream_t stream) {
    int threads =
            std::min(listIds.getSize(0), (idx_t)getMaxThreadsCurrentDevice());
    int blocks = utils::divUp(listIds.getSize(0), threads);

    imipqAppend<<<threads, blocks, 0, stream>>>(
            codebookSize, listIds, listOffset, encodings, listCodes);

    CUDA_TEST_ERROR();
}

void runIMIPQAppend(
        int codebookSize,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        Tensor<uint8_t, 2, true>& encodings,
        Tensor<uint8_t, 1, true, long>& listCodes,
        Tensor<unsigned int, 1, true>& listStartOffsets,
        int encodingNumBytes,
        cudaStream_t stream) {
    int threads =
            std::min(listIds.getSize(0), (idx_t)getMaxThreadsCurrentDevice());
    int blocks = utils::divUp(listIds.getSize(0), threads);

    imipqAppend<<<threads, blocks, 0, stream>>>(
            codebookSize,
            listIds,
            listOffset,
            encodings,
            listCodes,
            listStartOffsets,
            encodingNumBytes);

    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
