/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/gpu/utils/WarpPackedBits.cuh>
#include <faiss/gpu/utils/WarpShuffles.cuh>

#include <algorithm>

namespace faiss {
namespace gpu {

//
// IVF list metadata aupdate
//

// Updates the device-size array of list start pointers for codes and indices
__global__ void runUpdateListPointers(
        Tensor<idx_t, 1, true> listIds,
        Tensor<idx_t, 1, true> newListLength,
        Tensor<void*, 1, true> newCodePointers,
        Tensor<void*, 1, true> newIndexPointers,
        idx_t* listLengths,
        void** listCodes,
        void** listIndices) {
    idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < listIds.getSize(0)) {
        idx_t listId = listIds[i];
        listLengths[listId] = newListLength[i];
        listCodes[listId] = newCodePointers[i];
        listIndices[listId] = newIndexPointers[i];
    }
}

void runUpdateListPointers(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& newListLength,
        Tensor<void*, 1, true>& newCodePointers,
        Tensor<void*, 1, true>& newIndexPointers,
        DeviceVector<idx_t>& listLengths,
        DeviceVector<void*>& listCodes,
        DeviceVector<void*>& listIndices,
        cudaStream_t stream) {
    auto numThreads =
            std::min(listIds.getSize(0), (idx_t)getMaxThreadsCurrentDevice());
    auto numBlocks = utils::divUp(listIds.getSize(0), numThreads);

    dim3 grid(numBlocks);
    dim3 block(numThreads);

    runUpdateListPointers<<<grid, block, 0, stream>>>(
            listIds,
            newListLength,
            newCodePointers,
            newIndexPointers,
            listLengths.data(),
            listCodes.data(),
            listIndices.data());

    CUDA_TEST_ERROR();
}

// Appends new indices for vectors being added to the IVF indices lists
__global__ void ivfIndicesAppend(
        Tensor<idx_t, 1, true> listIds,
        Tensor<idx_t, 1, true> listOffset,
        Tensor<idx_t, 1, true> indices,
        IndicesOptions opt,
        void** listIndices) {
    idx_t vec = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (vec >= listIds.getSize(0)) {
        return;
    }

    idx_t listId = listIds[vec];
    idx_t offset = listOffset[vec];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || offset == -1) {
        return;
    }

    idx_t index = indices[vec];

    if (opt == INDICES_32_BIT) {
        // FIXME: there could be overflow here, but where should we check this?
        ((int*)listIndices[listId])[offset] = (int)index;
    } else if (opt == INDICES_64_BIT) {
        ((idx_t*)listIndices[listId])[offset] = index;
    }
}

void runIVFIndicesAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& indices,
        IndicesOptions opt,
        DeviceVector<void*>& listIndices,
        cudaStream_t stream) {
    FAISS_ASSERT(
            opt == INDICES_CPU || opt == INDICES_IVF || opt == INDICES_32_BIT ||
            opt == INDICES_64_BIT);

    if (opt != INDICES_CPU && opt != INDICES_IVF) {
        auto num = listIds.getSize(0);
        auto threads = std::min(num, (idx_t)getMaxThreadsCurrentDevice());
        auto blocks = utils::divUp(num, threads);

        ivfIndicesAppend<<<blocks, threads, 0, stream>>>(
                listIds, listOffset, indices, opt, listIndices.data());

        CUDA_TEST_ERROR();
    }
}

//
// IVF non-interleaved append
//

template <typename Codec>
__global__ void ivfFlatAppend(
        Tensor<idx_t, 1, true> listIds,
        Tensor<idx_t, 1, true> listOffset,
        Tensor<float, 2, true> vecs,
        void** listData,
        Codec codec) {
    idx_t vec = blockIdx.x;

    idx_t listId = listIds[vec];
    idx_t offset = listOffset[vec];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || offset == -1) {
        return;
    }

    // Handle whole encoding (only thread 0 will handle the remainder)
    // FIXME: dimension < max int?
    idx_t limit = utils::divDown(vecs.getSize(1), Codec::kDimPerIter);

    idx_t i;
    for (i = threadIdx.x; i < limit; i += blockDim.x) {
        int realDim = i * Codec::kDimPerIter;
        float toEncode[Codec::kDimPerIter];

#pragma unroll
        for (int j = 0; j < Codec::kDimPerIter; ++j) {
            toEncode[j] = vecs[vec][realDim + j];
        }

        codec.encode(listData[listId], offset, i, toEncode);
    }

    // Handle remainder with a single thread, if any
    if (Codec::kDimPerIter > 1) {
        int realDim = limit * Codec::kDimPerIter;

        // Was there any remainder?
        if (realDim < vecs.getSize(1)) {
            if (threadIdx.x == 0) {
                float toEncode[Codec::kDimPerIter];

                // How many remaining that we need to encode
                int remaining = vecs.getSize(1) - realDim;

#pragma unroll
                for (int j = 0; j < Codec::kDimPerIter; ++j) {
                    int idx = realDim + j;
                    toEncode[j] = idx < vecs.getSize(1) ? vecs[vec][idx] : 0.0f;
                }

                codec.encodePartial(
                        listData[listId], offset, i, remaining, toEncode);
            }
        }
    }
}

void runIVFFlatAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<float, 2, true>& vecs,
        GpuScalarQuantizer* scalarQ,
        DeviceVector<void*>& listData,
        cudaStream_t stream) {
    auto dim = vecs.getSize(1);
    idx_t maxThreads = getMaxThreadsCurrentDevice();

    // Each block will handle appending a single vector
#define RUN_APPEND                                                  \
    do {                                                            \
        dim3 grid(vecs.getSize(0));                                 \
        dim3 block(std::min(dim / codec.kDimPerIter, maxThreads));  \
        ivfFlatAppend<<<grid, block, 0, stream>>>(                  \
                listIds, listOffset, vecs, listData.data(), codec); \
    } while (0)

    if (!scalarQ) {
        CodecFloat codec(dim * sizeof(float));
        RUN_APPEND;
    } else {
        switch (scalarQ->qtype) {
            case ScalarQuantizer::QuantizerType::QT_8bit: {
                Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1> codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);
                RUN_APPEND;
            } break;
            case ScalarQuantizer::QuantizerType::QT_8bit_uniform: {
                Codec<ScalarQuantizer::QuantizerType::QT_8bit_uniform, 1> codec(
                        scalarQ->code_size,
                        scalarQ->trained[0],
                        scalarQ->trained[1]);
                RUN_APPEND;
            } break;
            case ScalarQuantizer::QuantizerType::QT_fp16: {
                Codec<ScalarQuantizer::QuantizerType::QT_fp16, 1> codec(
                        scalarQ->code_size);
                RUN_APPEND;
            } break;
            case ScalarQuantizer::QuantizerType::QT_8bit_direct: {
                Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct, 1> codec(
                        scalarQ->code_size);
                RUN_APPEND;
            } break;
            case ScalarQuantizer::QuantizerType::QT_4bit: {
                Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1> codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);
                RUN_APPEND;
            } break;
            case ScalarQuantizer::QuantizerType::QT_4bit_uniform: {
                Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform, 1> codec(
                        scalarQ->code_size,
                        scalarQ->trained[0],
                        scalarQ->trained[1]);
                RUN_APPEND;
            } break;
            default:
                // unimplemented, should be handled at a higher level
                FAISS_ASSERT(false);
        }
    }

    CUDA_TEST_ERROR();

#undef RUN_APPEND
}

__global__ void ivfpqAppend(
        Tensor<idx_t, 1, true> listIds,
        Tensor<idx_t, 1, true> listOffset,
        Tensor<uint8_t, 2, true> encodings,
        void** listCodes) {
    idx_t encodingToAdd = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (encodingToAdd >= listIds.getSize(0)) {
        return;
    }

    idx_t listId = listIds[encodingToAdd];
    idx_t vectorNumInList = listOffset[encodingToAdd];

    // Add vector could be invalid (contains NaNs etc)
    if (listId == -1 || vectorNumInList == -1) {
        return;
    }

    auto encoding = encodings[encodingToAdd];

    // Layout with dimensions innermost
    uint8_t* codeStart = ((uint8_t*)listCodes[listId]) +
            vectorNumInList * encodings.getSize(1);

    // FIXME: stride with threads instead of single thread
    for (idx_t i = 0; i < encodings.getSize(1); ++i) {
        codeStart[i] = encoding[i];
    }
}

void runIVFPQAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<uint8_t, 2, true>& encodings,
        DeviceVector<void*>& listCodes,
        cudaStream_t stream) {
    idx_t threads =
            std::min(listIds.getSize(0), (idx_t)getMaxThreadsCurrentDevice());
    idx_t blocks = utils::divUp(listIds.getSize(0), threads);

    ivfpqAppend<<<threads, blocks, 0, stream>>>(
            listIds, listOffset, encodings, listCodes.data());

    CUDA_TEST_ERROR();
}

//
// IVF interleaved append
//

// Scalar encode a vector to Codec::EncodeT word-sized values; previously this
// was fused into a single append kernel but was refactored so that Flat, SQ and
// PQ all use the same arbitrary bitwidth append kernel
template <typename Codec>
__global__ void sqEncode(
        Tensor<float, 2, true> vecs,
        Tensor<typename Codec::EncodeT, 2, true> encodedVecs,
        Codec codec) {
    idx_t vec = blockIdx.x;

    for (idx_t d = threadIdx.x; d < vecs.getSize(1); d += blockDim.x) {
        encodedVecs[vec][d] = codec.encodeNew(d, vecs[vec][d]);
    }
}

template <typename Codec>
void runSQEncode(
        Tensor<float, 2, true>& vecs,
        Tensor<typename Codec::EncodeT, 2, true>& encodedVecs,
        Codec codec,
        cudaStream_t stream) {
    idx_t threads =
            std::min(vecs.getSize(1), (idx_t)getMaxThreadsCurrentDevice());
    idx_t blocks = vecs.getSize(0);

    sqEncode<<<blocks, threads, 0, stream>>>(vecs, encodedVecs, codec);
}

// Handles appending encoded vectors (one per EncodeT word) packed into
// EncodeBits interleaved by kWarpSize vectors.
// This is used by Flat, SQ and PQ code for the interleaved format.
template <typename EncodeT, int EncodeBits>
__global__ void ivfInterleavedAppend(
        // the IDs (offset in listData) of the unique lists
        // being added to
        Tensor<idx_t, 1, true> uniqueLists,
        // For each of the list IDs in uniqueLists, the start
        // offset in vectorsByUniqueList for the vectors that
        // we are adding to that list
        Tensor<idx_t, 1, true> uniqueListVectorStart,
        // IDs in vecs of the vectors being added to each
        // unique list
        // The vectors (offset in vecs) added to
        // uniqueLists[i] is:
        // {vBUL[uLVS[i]], ..., vBUL[uLVS[i+1] - 1]}
        Tensor<idx_t, 1, true> vectorsByUniqueList,
        // For each of the list IDs in uniqueLists, the start
        // offset (by vector) within that list where we begin
        // appending
        Tensor<idx_t, 1, true> uniqueListStartOffset,
        // The EncodeT-sized encoded vectors
        Tensor<EncodeT, 2, true> encodedVecs,
        // The set of addresses for each of the lists
        void** listData) {
    // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
    int laneId = threadIdx.x % kWarpSize;
    int warpId = threadIdx.x / kWarpSize;
    int warpsPerBlock = blockDim.x / kWarpSize;

    // Each block is dedicated to a separate list
    idx_t listId = uniqueLists[blockIdx.x];

    // The vecs we add to the list are at indices [vBUL[vecIdStart],
    // vBUL[vecIdEnd])
    idx_t vecIdStart = uniqueListVectorStart[blockIdx.x];
    // uLVS is explicitly terminated for us with one more than the number of
    // blocks that we have
    idx_t vecIdEnd = uniqueListVectorStart[blockIdx.x + 1];

    // How many vectors we are adding to this list
    auto numVecsAdding = vecIdEnd - vecIdStart;

    // The first vector we are updating within the list
    auto listVecStart = uniqueListStartOffset[blockIdx.x];

    // These are the actual vec IDs that we are adding (in vecs)
    auto listVecIds = vectorsByUniqueList[vecIdStart].data();

    // All data is written by groups of kWarpSize vectors (to mirror the warp).
    // listVecStart could be in the middle of this, or even, for sub-byte
    // encodings, mean that the first vector piece of data that we need to
    // update is in the high part of a byte.
    //
    // WarpPackedBits allows writing of arbitrary bit packed data in groups of
    // kWarpSize, but we ensure that it only operates on the group of kWarpSize
    // vectors. In order to do this we need to actually start updating vectors
    // at the next lower multiple of kWarpSize from listVecStart.
    auto alignedListVecStart = utils::roundDown(listVecStart, kWarpSize);

    // Each block of kWarpSize vectors fully encodes into this many bytes
    constexpr int bytesPerVectorBlockDim = EncodeBits * kWarpSize / 8;
    constexpr int wordsPerVectorBlockDim =
            bytesPerVectorBlockDim / sizeof(EncodeT);
    auto wordsPerVectorBlock = wordsPerVectorBlockDim * encodedVecs.getSize(1);

    EncodeT* listStart = ((EncodeT*)listData[listId]);

    // Each warp within the block handles a different chunk of kWarpSize
    auto warpVec = alignedListVecStart +
            (faiss::gpu::Tensor<long, 1, true>::DataType)warpId * kWarpSize;

    // The warp data starts here
    EncodeT* warpData = listStart + (warpVec / kWarpSize) * wordsPerVectorBlock;

    // Each warp encodes a single block
    for (; warpVec < listVecStart + numVecsAdding;
         // but block stride
         warpVec += blockDim.x,
         // the new warp data base strides by how many vector blocks we are
         // encoding, which is one per warp
         warpData += warpsPerBlock * wordsPerVectorBlock) {
        // This lane is adding this vec (if it is within bounds)
        auto laneVec = warpVec + laneId;

        // Which vector does this correspond to in the set of vectors that we
        // need to add? If this is < 0, then this particular thread is not
        // encoding / appending a new vector
        auto laneVecAdding = laneVec - listVecStart;

        // We are actually adding a new vector if this is within range
        bool valid = laneVecAdding >= 0 && laneVecAdding < numVecsAdding;

        // Now, which actual vector in vecs is this?
        auto vecId = valid ? listVecIds[laneVecAdding] : 0;

        // Each warp that has some vector data available needs to write out the
        // vector components
        EncodeT* data = warpData;

        for (int dim = 0; dim < encodedVecs.getSize(1); ++dim) {
            EncodeT enc = valid ? encodedVecs[vecId][dim] : (EncodeT)0;
            WarpPackedBits<EncodeT, EncodeBits>::write(
                    laneId, enc, valid, data);

            data += wordsPerVectorBlockDim;
        }
    }
}

void runIVFFlatInterleavedAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& uniqueLists,
        Tensor<idx_t, 1, true>& vectorsByUniqueList,
        Tensor<idx_t, 1, true>& uniqueListVectorStart,
        Tensor<idx_t, 1, true>& uniqueListStartOffset,
        Tensor<float, 2, true>& vecs,
        GpuScalarQuantizer* scalarQ,
        DeviceVector<void*>& listData,
        GpuResources* res,
        cudaStream_t stream) {
    auto dim = vecs.getSize(1);

#define RUN_APPEND(ENCODE_T, ENCODE_BITS, DATA)     \
    do {                                            \
        dim3 grid(uniqueLists.getSize(0));          \
        dim3 block(128);                            \
        ivfInterleavedAppend<ENCODE_T, ENCODE_BITS> \
                <<<grid, block, 0, stream>>>(       \
                        uniqueLists,                \
                        uniqueListVectorStart,      \
                        vectorsByUniqueList,        \
                        uniqueListStartOffset,      \
                        DATA,                       \
                        listData.data());           \
    } while (0)

    if (!scalarQ) {
        // No encoding is needed, we just append directly
        RUN_APPEND(float, 32, vecs);
        return;
    }

    // only implemented at the moment
    FAISS_ASSERT(scalarQ->bits == 16 || scalarQ->bits <= 8);

    if (scalarQ->bits == 16) {
        FAISS_ASSERT(scalarQ->qtype == ScalarQuantizer::QuantizerType::QT_fp16);

        using CodecT = Codec<ScalarQuantizer::QuantizerType::QT_fp16, 1>;
        CodecT codec(scalarQ->qtype);

        DeviceTensor<half, 2, true> encodedVecs(
                res,
                makeTempAlloc(AllocType::Other, stream),
                {vecs.getSize(0), vecs.getSize(1)});

        runSQEncode(vecs, encodedVecs, codec, stream);
        RUN_APPEND(CodecT::EncodeT, CodecT::kEncodeBits, encodedVecs);

    } else if (scalarQ->bits <= 8) {
        DeviceTensor<uint8_t, 2, true> encodedVecs(
                res,
                makeTempAlloc(AllocType::Other, stream),
                {vecs.getSize(0), vecs.getSize(1)});

        switch (scalarQ->qtype) {
            case ScalarQuantizer::QuantizerType::QT_8bit: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1>;
                CodecT codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);

                runSQEncode(vecs, encodedVecs, codec, stream);
                RUN_APPEND(CodecT::EncodeT, CodecT::kEncodeBits, encodedVecs);
            } break;
            case ScalarQuantizer::QuantizerType::QT_8bit_uniform: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_8bit_uniform,
                              1>;
                CodecT codec(
                        scalarQ->code_size,
                        scalarQ->trained[0],
                        scalarQ->trained[1]);

                runSQEncode(vecs, encodedVecs, codec, stream);
                RUN_APPEND(CodecT::EncodeT, CodecT::kEncodeBits, encodedVecs);
            } break;
            case ScalarQuantizer::QuantizerType::QT_8bit_direct: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct,
                              1>;
                CodecT codec(scalarQ->code_size);

                runSQEncode(vecs, encodedVecs, codec, stream);
                RUN_APPEND(CodecT::EncodeT, CodecT::kEncodeBits, encodedVecs);
            } break;
            case ScalarQuantizer::QuantizerType::QT_6bit: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_6bit, 1>;
                CodecT codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);

                runSQEncode(vecs, encodedVecs, codec, stream);
                RUN_APPEND(CodecT::EncodeT, CodecT::kEncodeBits, encodedVecs);
            } break;
            case ScalarQuantizer::QuantizerType::QT_4bit: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1>;
                CodecT codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);

                runSQEncode(vecs, encodedVecs, codec, stream);
                RUN_APPEND(CodecT::EncodeT, CodecT::kEncodeBits, encodedVecs);
            } break;
            case ScalarQuantizer::QuantizerType::QT_4bit_uniform: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform,
                              1>;
                CodecT codec(
                        scalarQ->code_size,
                        scalarQ->trained[0],
                        scalarQ->trained[1]);

                runSQEncode(vecs, encodedVecs, codec, stream);
                RUN_APPEND(CodecT::EncodeT, CodecT::kEncodeBits, encodedVecs);
            } break;
            default:
                // unimplemented, should be handled at a higher level
                FAISS_ASSERT(false);
        }
    }

#undef RUN_APPEND
    CUDA_TEST_ERROR();
}

void runIVFPQInterleavedAppend(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        Tensor<idx_t, 1, true>& uniqueLists,
        Tensor<idx_t, 1, true>& vectorsByUniqueList,
        Tensor<idx_t, 1, true>& uniqueListVectorStart,
        Tensor<idx_t, 1, true>& uniqueListStartOffset,
        int bitsPerCode,
        Tensor<uint8_t, 2, true>& encodings,
        DeviceVector<void*>& listCodes,
        cudaStream_t stream) {
    // limitation for now
    FAISS_ASSERT(bitsPerCode <= 8);

#define RUN_APPEND(ENCODE_T, ENCODE_BITS)           \
    do {                                            \
        dim3 grid(uniqueLists.getSize(0));          \
        dim3 block(128);                            \
                                                    \
        ivfInterleavedAppend<ENCODE_T, ENCODE_BITS> \
                <<<grid, block, 0, stream>>>(       \
                        uniqueLists,                \
                        uniqueListVectorStart,      \
                        vectorsByUniqueList,        \
                        uniqueListStartOffset,      \
                        encodings,                  \
                        listCodes.data());          \
    } while (0)

    switch (bitsPerCode) {
        case 4: {
            RUN_APPEND(uint8_t, 4);
            break;
        }
        case 5: {
            RUN_APPEND(uint8_t, 5);
            break;
        }
        case 6: {
            RUN_APPEND(uint8_t, 6);
            break;
        }
        case 8: {
            RUN_APPEND(uint8_t, 8);
            break;
        }
        default:
            // unhandled
            FAISS_ASSERT(false);
            break;
    }

#undef RUN_APPEND
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
