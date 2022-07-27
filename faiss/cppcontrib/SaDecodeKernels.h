// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// This file contains a custom fast implementation of faiss::Index::sa_decode()
//   function for the following index families:
//   * IVF256,PQ[1]x8np
//   * Residual[1]x8,PQ[2]x8
//
// The goal was to achieve the maximum performance, so the template version it
// is. The provided index families share the same code for sa_decode.
// The front-end code looks the following:
//   {
//     template <intptr_t DIM, intptr_t COARSE_SIZE, intptr_t FINE_SIZE>
//     struct Index2LevelDecoder { /*...*/ };
//   }
// * DIM is the dimensionality of data
// * COARSE_SIZE is the dimensionality of the coarse quantizer (IVF, Residual)
// * FINE_SIZE is the dimensionality of the ProductQuantizer dsq
// For example, "IVF256,PQ8np" for 160-dim data translates into
//   Index2LevelDecoder<160,160,20>
// For example, "Residual4x8,PQ16" for 256-dim data translates into
//   Index2LevelDecoder<256,64,16>
//
// Unlike the general purpose version in faiss::Index::sa_decode(),
//   this version provides the following functions:
// * ::store, which is similar to sa_decode(1, input, output),
//   The method signature is the following:
//   {
//     void store(
//       const float* const __restrict pqCoarseCentroids,
//       const float* const __restrict pqFineCentroids,
//       const uint8_t* const __restrict code,
//       float* const __restrict outputStore);
//   }
// * ::accum, which is used to create a linear combination
//   of decoded vectors:
//   {
//     faiss::Index* index;
//     float weight;
//
//     std::vector<float> buffer(d, 0);
//
//     index->sa_decode(1, input, buffer.data());
//     for (size_t iDim = 0; iDim < d; iDim++)
//       output[iDim] += weight * input[iDim];
//   }
//   The method signature is the following:
//   {
//    static void accum(
//      const float* const __restrict pqCoarseCentroids,
//      const float* const __restrict pqFineCentroids,
//      const uint8_t* const __restrict code,
//      const float weight,
//      float* const __restrict outputAccum);
//   }
// * There is an additional overload for ::accum that decodes two vectors
//   per call. This provides an additional speedup because of a CPU
//   superscalar architecture. Doing more vectors per call is less attractive
//   because of the possible lack of available CPU registers, but it is still
//   doable.
//   The method signature is the following:
//   {
//    static void accum(
//      const float* const __restrict pqCoarseCentroids0,
//      const float* const __restrict pqFineCentroids0,
//      const uint8_t* const __restrict code0,
//      const float weight0,
//      const float* const __restrict pqCoarseCentroids1,
//      const float* const __restrict pqFineCentroids1,
//      const uint8_t* const __restrict code1,
//      const float weight1,
//      float* const __restrict outputAccum);
//   }
// The provided version is not multithreaded.
//
// Currently, an AVX2+FMA implementation is available. AVX512 version is also
//   doable, but it was found to be slower than AVX2 for real world applications
//   that I needed.

#ifdef __AVX2__
#include <faiss/cppcontrib/SaDecodeKernels-avx2-inl.h>
#elif defined(__ARM_NEON)
#include <faiss/cppcontrib/SaDecodeKernels-neon-inl.h>
#else
#include <faiss/cppcontrib/SaDecodeKernels-inl.h>
#endif
