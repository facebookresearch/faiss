/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/WarpPackedBits.cuh>
#include <faiss/gpu/utils/WarpShuffles.cuh>

namespace faiss { namespace gpu {

//
// IVF list length update
//

// Updates the device-size array of list start pointers for codes and indices
__global__ void
runUpdateListPointers(Tensor<int, 1, true> listIds,
                      Tensor<int, 1, true> newListLength,
                      Tensor<void*, 1, true> newCodePointers,
                      Tensor<void*, 1, true> newIndexPointers,
                      int* listLengths,
                      void** listCodes,
                      void** listIndices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < listIds.getSize(0)) {
    int listId = listIds[i];
    listLengths[listId] = newListLength[i];
    listCodes[listId] = newCodePointers[i];
    listIndices[listId] = newIndexPointers[i];
  }
}

void
runUpdateListPointers(Tensor<int, 1, true>& listIds,
                      Tensor<int, 1, true>& newListLength,
                      Tensor<void*, 1, true>& newCodePointers,
                      Tensor<void*, 1, true>& newIndexPointers,
                      thrust::device_vector<int>& listLengths,
                      thrust::device_vector<void*>& listCodes,
                      thrust::device_vector<void*>& listIndices,
                      cudaStream_t stream) {
  int numThreads = std::min(listIds.getSize(0), getMaxThreadsCurrentDevice());
  int numBlocks = utils::divUp(listIds.getSize(0), numThreads);

  dim3 grid(numBlocks);
  dim3 block(numThreads);

  runUpdateListPointers<<<grid, block, 0, stream>>>(
    listIds, newListLength, newCodePointers, newIndexPointers,
    listLengths.data().get(),
    listCodes.data().get(),
    listIndices.data().get());

  CUDA_TEST_ERROR();
}

// Appends new indices for vectors being added to the IVF indices lists
__global__ void
ivfIndicesAppend(Tensor<int, 1, true> listIds,
                 Tensor<int, 1, true> listOffset,
                 Tensor<Index::idx_t, 1, true> indices,
                 IndicesOptions opt,
                 void** listIndices) {
  int vec = blockIdx.x * blockDim.x + threadIdx.x;

  if (vec >= listIds.getSize(0)) {
    return;
  }

  int listId = listIds[vec];
  int offset = listOffset[vec];

  // Add vector could be invalid (contains NaNs etc)
  if (listId == -1 || offset == -1) {
    return;
  }

  auto index = indices[vec];

  if (opt == INDICES_32_BIT) {
    // FIXME: there could be overflow here, but where should we check this?
    ((int*) listIndices[listId])[offset] = (int) index;
  } else if (opt == INDICES_64_BIT) {
    ((Index::idx_t*) listIndices[listId])[offset] = index;
  }
}

void
runIVFIndicesAppend(Tensor<int, 1, true>& listIds,
                    Tensor<int, 1, true>& listOffset,
                    Tensor<Index::idx_t, 1, true>& indices,
                    IndicesOptions opt,
                    thrust::device_vector<void*>& listIndices,
                    cudaStream_t stream) {
  FAISS_ASSERT(opt == INDICES_CPU ||
               opt == INDICES_IVF ||
               opt == INDICES_32_BIT ||
               opt == INDICES_64_BIT);

  if (opt != INDICES_CPU && opt != INDICES_IVF) {
    int num = listIds.getSize(0);
    int threads = std::min(num, getMaxThreadsCurrentDevice());
    int blocks = utils::divUp(num, threads);

    ivfIndicesAppend<<<blocks, threads, 0, stream>>>(
      listIds,
      listOffset,
      indices,
      opt,
      listIndices.data().get());

    CUDA_TEST_ERROR();
  }
}

//
// IVF PQ append
//

__global__ void
ivfpqInvertedListAppend(Tensor<int, 1, true> listIds,
                        Tensor<int, 1, true> listOffset,
                        Tensor<int, 2, true> encodings,
                        bool layoutBy32,
                        void** listCodes) {
  int encodingToAdd = blockIdx.x * blockDim.x + threadIdx.x;

  if (encodingToAdd >= listIds.getSize(0)) {
    return;
  }

  int listId = listIds[encodingToAdd];
  int vectorNumInList = listOffset[encodingToAdd];

  // Add vector could be invalid (contains NaNs etc)
  if (listId == -1 || vectorNumInList == -1) {
    return;
  }

  auto encoding = encodings[encodingToAdd];

  if (layoutBy32) {
    // Layout with 32 vectors interleaved
    int fullBlockSize = 32 * encodings.getSize(1);

    int blockStart = (vectorNumInList / 32) * fullBlockSize;
    int start = blockStart + (vectorNumInList % 32);

    uint8_t* codeStart = ((uint8_t*) listCodes[listId]) + start;

    // This is actually properly warp coalesced, as each thread handles a
    // different vector
    for (int i = 0; i < encodings.getSize(1); ++i) {
      codeStart[i * 32] = (uint8_t) encoding[i];
    }
  } else {
    // Layout with dimensions innermost
    uint8_t* codeStart =
      ((uint8_t*) listCodes[listId]) +
      vectorNumInList * encodings.getSize(1);

    // FIXME: slow
    for (int i = 0; i < encodings.getSize(1); ++i) {
      codeStart[i] = (uint8_t) encoding[i];
    }
  }
}

void
runIVFPQInvertedListAppend(Tensor<int, 1, true>& listIds,
                           Tensor<int, 1, true>& listOffset,
                           Tensor<int, 2, true>& encodings,
                           Tensor<Index::idx_t, 1, true>& indices,
                           bool layoutBy32,
                           thrust::device_vector<void*>& listCodes,
                           thrust::device_vector<void*>& listIndices,
                           IndicesOptions indicesOptions,
                           cudaStream_t stream) {
  // Append the indices that we're about to add, if any
  runIVFIndicesAppend(listIds, listOffset, indices,
                      indicesOptions, listIndices, stream);

  // Append the PQ codes
  int threads = std::min(listIds.getSize(0), getMaxThreadsCurrentDevice());
  int blocks = utils::divUp(listIds.getSize(0), threads);

  ivfpqInvertedListAppend<<<threads, blocks, 0, stream>>>(
    listIds, listOffset, encodings, layoutBy32, listCodes.data().get());

  CUDA_TEST_ERROR();
}

//
// IVF flat append
//

template <typename Codec>
__global__ void
ivfFlatInterleavedAppend(// the IDs (offset in listData) of the unique lists
                         // being added to
                         Tensor<int, 1, true> uniqueLists,
                         // For each of the list IDs in uniqueLists, the start
                         // offset in vectorsByUniqueList for the vectors that
                         // we are adding to that list
                         Tensor<int, 1, true> uniqueListVectorStart,
                         // IDs in vecs of the vectors being added to each
                         // unique list
                         // The vectors (offset in vecs) added to
                         // uniqueLists[i] is:
                         // {vBUL[uLVS[i]], ..., vBUL[uLVS[i+1] - 1]}
                         Tensor<int, 1, true> vectorsByUniqueList,
                         // For each of the list IDs in uniqueLists, the start
                         // offset (by vector) within that list where we begin
                         // appending
                         Tensor<int, 1, true> uniqueListStartOffset,
                         // The vectors being added
                         Tensor<float, 2, true> vecs,
                         // The set of addresses for each of the lists
                         void** listData,
                         Codec codec) {
  using EncodeT = typename Codec::EncodeT;

  // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
  int laneId = threadIdx.x % kWarpSize;
  int warpId = threadIdx.x / kWarpSize;
  int warpsPerBlock = blockDim.x / kWarpSize;

  // Each block is dedicated to a separate list
  int listId = uniqueLists[blockIdx.x];

  // The vecs we add to the list are at indices [vBUL[vecIdStart], vBUL[vecIdEnd])
  int vecIdStart = uniqueListVectorStart[blockIdx.x];
  // uLVS is explicitly terminated for us with one more than the number of
  // blocks that we have
  int vecIdEnd = uniqueListVectorStart[blockIdx.x + 1];

  // How many vectors we are adding to this list
  int numVecsAdding = vecIdEnd - vecIdStart;

  // The first vector we are updating within the list
  auto listVecStart = uniqueListStartOffset[blockIdx.x];

  // These are the actual vec IDs that we are adding (in vecs)
  int* listVecIds = vectorsByUniqueList[vecIdStart].data();

  // All data is written by groups of 32 vectors (to mirror the warp).
  // listVecStart could be in the middle of this, or even, for sub-byte
  // encodings, mean that the first vector piece of data that we need to update
  // is in the high part of a byte.
  //
  // WarpPackedBits allows writing of arbitrary bit packed data in groups of 32,
  // but we ensure that it only operates on the group of 32 vectors.
  // In order to do this we need to actually start updating vectors at the next
  // lower multiple of 32 from listVecStart.
  int alignedListVecStart = utils::roundDown(listVecStart, 32);

  // Each block of 32 vectors fully encodes into this many bytes
  constexpr int bytesPerVectorBlockDim = Codec::kEncodeBits * 32 / 8;
  constexpr int wordsPerVectorBlockDim = bytesPerVectorBlockDim / sizeof(EncodeT);
  int wordsPerVectorBlock = wordsPerVectorBlockDim * vecs.getSize(1);

  EncodeT* listStart = ((EncodeT*) listData[listId]);

  // Each warp within the block handles a different chunk of 32
  int warpVec = alignedListVecStart + warpId * 32;

  // The warp data starts here
  EncodeT* warpData = listStart + (warpVec / 32) * wordsPerVectorBlock;

  // Each warp encodes a single block
  for (; warpVec < listVecStart + numVecsAdding;
       // but block stride
       warpVec += blockDim.x,
       // the new warp data base strides by how many vector blocks we are
       // encoding, which is one per warp
       warpData += warpsPerBlock * wordsPerVectorBlock) {
    // This lane is adding this vec (if it is within bounds)
    int laneVec = warpVec + laneId;

    // Which vector does this correspond to in the set of vectors that we need
    // to add?
    // If this is < 0, then this particular thread is not encoding / appending a
    // new vector
    int laneVecAdding = laneVec - listVecStart;

    // We are actually adding a new vector if this is within range
    bool valid = laneVecAdding >= 0 && laneVecAdding < numVecsAdding;

    // Now, which actual vector in vecs is this?
    int vecId = valid ? listVecIds[laneVecAdding] : 0;

    // Each warp that has some vector data available needs to write out the
    // vector components
    EncodeT* data = warpData;

    for (int dim = 0;
         dim < vecs.getSize(1); ++dim,
           data += wordsPerVectorBlockDim) {
      EncodeT enc = 0;
      if (valid) {
        enc = codec.encodeNew(dim, vecs[vecId][dim]);
      }

      WarpPackedBits<EncodeT, Codec::kEncodeBits>::
        write(laneId, enc, valid, data);
    }
  }
}

template <typename Codec>
__global__ void
ivfFlatAppend(Tensor<int, 1, true> listIds,
              Tensor<int, 1, true> listOffset,
              Tensor<float, 2, true> vecs,
              void** listData,
              Codec codec) {
  int vec = blockIdx.x;

  int listId = listIds[vec];
  int offset = listOffset[vec];

  // Add vector could be invalid (contains NaNs etc)
  if (listId == -1 || offset == -1) {
    return;
  }

  // Handle whole encoding (only thread 0 will handle the remainder)
  int limit = utils::divDown(vecs.getSize(1), Codec::kDimPerIter);

  int i;
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

        codec.encodePartial(listData[listId], offset, i, remaining, toEncode);
      }
    }
  }
}

void
runIVFFlatInterleavedAppend(Tensor<int, 1, true>& listIds,
                            Tensor<int, 1, true>& listOffset,
                            Tensor<int, 1, true>& uniqueLists,
                            Tensor<int, 1, true>& vectorsByUniqueList,
                            Tensor<int, 1, true>& uniqueListVectorStart,
                            Tensor<int, 1, true>& uniqueListStartOffset,
                            Tensor<float, 2, true>& vecs,
                            Tensor<Index::idx_t, 1, true>& indices,
                            GpuScalarQuantizer* scalarQ,
                            thrust::device_vector<void*>& listData,
                            thrust::device_vector<void*>& listIndices,
                            IndicesOptions indicesOptions,
                            cudaStream_t stream) {
  int dim = vecs.getSize(1);

  // First, append the indices that we're about to add, if any
  runIVFIndicesAppend(listIds, listOffset, indices,
                      indicesOptions, listIndices, stream);

#define RUN_APPEND                                                      \
  do {                                                                  \
    dim3 grid(uniqueLists.getSize(0));                                  \
    dim3 block(128);                                                    \
    ivfFlatInterleavedAppend                                            \
      <<<grid, block, 0, stream>>>(                                     \
        uniqueLists,                                                    \
        uniqueListVectorStart,                                          \
        vectorsByUniqueList,                                            \
        uniqueListStartOffset,                                          \
        vecs,                                                           \
        listData.data().get(),                                          \
        codec);                                                         \
  } while (0)

  if (!scalarQ) {
    CodecFloat codec(dim * sizeof(float));
    RUN_APPEND;
  } else {
    switch (scalarQ->qtype) {
      case ScalarQuantizer::QuantizerType::QT_8bit:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1>
          codec(scalarQ->code_size,
                scalarQ->gpuTrained.data(),
                scalarQ->gpuTrained.data() + dim);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_8bit_uniform:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_8bit_uniform, 1>
          codec(scalarQ->code_size, scalarQ->trained[0], scalarQ->trained[1]);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_fp16:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_fp16, 1>
          codec(scalarQ->code_size);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_8bit_direct:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct, 1>
          codec(scalarQ->code_size);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_6bit:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_6bit, 1>
          codec(scalarQ->code_size,
                scalarQ->gpuTrained.data(),
                scalarQ->gpuTrained.data() + dim);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_4bit:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1>
          codec(scalarQ->code_size,
                scalarQ->gpuTrained.data(),
                scalarQ->gpuTrained.data() + dim);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_4bit_uniform:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform, 1>
          codec(scalarQ->code_size, scalarQ->trained[0], scalarQ->trained[1]);
        RUN_APPEND;
      }
      break;
      default:
        // unimplemented, should be handled at a higher level
        FAISS_ASSERT(false);
    }
  }

  CUDA_TEST_ERROR();

#undef RUN_APPEND
}

void
runIVFFlatAppend(Tensor<int, 1, true>& listIds,
                 Tensor<int, 1, true>& listOffset,
                 Tensor<float, 2, true>& vecs,
                 Tensor<Index::idx_t, 1, true>& indices,
                 GpuScalarQuantizer* scalarQ,
                 thrust::device_vector<void*>& listData,
                 thrust::device_vector<void*>& listIndices,
                 IndicesOptions indicesOptions,
                 cudaStream_t stream) {
  int dim = vecs.getSize(1);
  int maxThreads = getMaxThreadsCurrentDevice();

  // First, append the indices that we're about to add, if any
  runIVFIndicesAppend(listIds, listOffset, indices,
                      indicesOptions, listIndices, stream);

  // Each block will handle appending a single vector
#define RUN_APPEND                                                      \
  do {                                                                  \
    dim3 grid(vecs.getSize(0));                                         \
    dim3 block(std::min(dim / codec.kDimPerIter, maxThreads));          \
    ivfFlatAppend                                                       \
      <<<grid, block, 0, stream>>>(                                     \
        listIds,                                                        \
        listOffset,                                                     \
        vecs,                                                           \
        listData.data().get(),                                          \
        codec);                                                         \
  } while (0)

  if (!scalarQ) {
    CodecFloat codec(dim * sizeof(float));
    RUN_APPEND;
  } else {
    switch (scalarQ->qtype) {
      case ScalarQuantizer::QuantizerType::QT_8bit:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1>
          codec(scalarQ->code_size,
                scalarQ->gpuTrained.data(),
                scalarQ->gpuTrained.data() + dim);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_8bit_uniform:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_8bit_uniform, 1>
          codec(scalarQ->code_size, scalarQ->trained[0], scalarQ->trained[1]);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_fp16:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_fp16, 1>
          codec(scalarQ->code_size);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_8bit_direct:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct, 1>
          codec(scalarQ->code_size);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_4bit:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1>
          codec(scalarQ->code_size,
                scalarQ->gpuTrained.data(),
                scalarQ->gpuTrained.data() + dim);
        RUN_APPEND;
      }
      break;
      case ScalarQuantizer::QuantizerType::QT_4bit_uniform:
      {
        Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform, 1>
          codec(scalarQ->code_size, scalarQ->trained[0], scalarQ->trained[1]);
        RUN_APPEND;
      }
      break;
      default:
        // unimplemented, should be handled at a higher level
        FAISS_ASSERT(false);
    }
  }

  CUDA_TEST_ERROR();

#undef RUN_APPEND
}


} } // namespace
