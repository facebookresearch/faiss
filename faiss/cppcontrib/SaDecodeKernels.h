// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// This file contains a custom fast implementation of faiss::Index::sa_decode()
//   function for the following index families:
//   * IVF256,PQ[1]x8np
//   * Residual[1]x8,PQ[2]x8
//   * IVF[2^9-2^16 bit],PQ[1]x8 (such as IVF1024,PQ16np)
//   * Residual1x[9-16 bit],PQ[1]x8 (such as Residual1x9,PQ8)
//   * PQ[1]x8
// Additionally, AVX2 and ARM versions support
//   * Residual[1]x8,PQ[2]x10
//   * Residual[1]x8,PQ[2]x16
//   * Residual[1]x10,PQ[2]x10
//   * Residual[1]x10,PQ[2]x16
//   * Residual[1]x16,PQ[2]x10
//   * Residual[1]x16,PQ[2]x16
//   * Residual1x[9-16 bit],PQ[1]x10 (such as Residual1x9,PQ16x10)
//   * * (use with COARSE_BITS=16)
//   * Residual1x[9-16 bit],PQ[1]x16 (such as Residual1x9,PQ16x16)
//   * * (use with COARSE_BITS=16)
//   * PQ[1]x10
//   * PQ[1]x16
// Unfortunately, currently Faiss does not support something like
//   IVF256,PQ16x10np
//
// The goal was to achieve the maximum performance, so the template version it
// is. The provided index families share the same code for sa_decode.
//
// The front-end code provides two high-level structures.
//
// First one:
//   {
//     template <
//        intptr_t DIM,
//        intptr_t COARSE_SIZE,
//        intptr_t FINE_SIZE,
//        intptr_t COARSE_BITS = 8
//        intptr_t FINE_BITS = 8>
//     struct Index2LevelDecoder { /*...*/ };
//   }
// * DIM is the dimensionality of data
// * COARSE_SIZE is the dimensionality of the coarse quantizer (IVF, Residual)
// * FINE_SIZE is the dimensionality of the ProductQuantizer dsq
// * COARSE_BITS is the number of bits that are needed to represent a coarse
//   quantizer code.
// * FINE_BITS is the number of bits that are needed to represent a fine
//   quantizer code.
// For example, "IVF256,PQ8np" for 160-dim data translates into
//   Index2LevelDecoder<160,160,20,8>
// For example, "Residual4x8,PQ16" for 256-dim data translates into
//   Index2LevelDecoder<256,64,1,8>
// For example, "IVF1024,PQ16np" for 256-dim data translates into
//   Index2LevelDecoder<256,256,16,10>. But as there are only 1 coarse code
//   element, Index2LevelDecoder<256,256,16,16> can be used as a faster
//   decoder.
// For example, "Residual4x10,PQ16x10np" for 256-dim data translates into
//   Index2LevelDecoder<256,64,16,10,10>
//
// Additional supported values for COARSE_BITS and FINE_BITS may be added later.
//
// Second one:
//   {
//     template <
//        intptr_t DIM,
//        intptr_t FINE_SIZE,
//        intptr_t FINE_BITS = 8>
//     struct IndexPQDecoder { /*...*/ };
//   }
// * DIM is the dimensionality of data
// * FINE_SIZE is the dimensionality of the ProductQuantizer dsq
// * FINE_BITS is the number of bits that are needed to represent a fine
//   quantizer code.
// For example, "PQ8np" for 160-dim data translates into
//   IndexPQDecoder<160,20>
//
// Unlike the general purpose version in faiss::Index::sa_decode(),
//   this version provides the following functions (please note that
//   pqCoarseCentroids params are not available for IndexPQDecoder,
//   but the functionality is the same as for Index2LevelDecoder):
//
// * ::store(), which is similar to sa_decode(1, input, output),
//   The method signature is the following:
//   {
//     void store(
//       const float* const __restrict pqCoarseCentroids,
//       const float* const __restrict pqFineCentroids,
//       const uint8_t* const __restrict code,
//       float* const __restrict outputStore);
//   }
//
// * ::accum(), which is used to create a linear combination
//   of decoded vectors:
//   {
//     const faiss::Index* const index;
//     const uint8_t* const input;
//     float weight;
//
//     std::vector<float> buffer(d, 0);
//
//     index->sa_decode(1, input, buffer.data());
//     for (size_t iDim = 0; iDim < d; iDim++)
//       output[iDim] += weight * buffer[iDim];
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
//
// * There is an additional overload for ::accum() that decodes two vectors
//   per call. This provides an additional speedup because of a CPU
//   superscalar architecture:
//   {
//     const faiss::Index* const index;
//     const uint8_t* const input0;
//     float weight0;
//     const uint8_t* const input1;
//     float weight1;
//
//     std::vector<float> buffer(d, 0);
//
//     index->sa_decode(1, input0, buffer.data());
//     for (size_t iDim = 0; iDim < d; iDim++)
//       output[iDim] += weight0 * buffer[iDim];
//
//     index->sa_decode(1, input1, buffer.data());
//     for (size_t iDim = 0; iDim < d; iDim++)
//       output[iDim] += weight1 * buffer[iDim];
//   }
//   If each code uses its own coarse quantizer centroids table and its own fine
//   quantizer centroids table, then the following overload can be used:
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
//   If codes share the coarse quantizer centroids table and also share
//   the fine quantizer centroids table, then the following overload can be
//   used:
//   {
//    static void accum(
//      const float* const __restrict pqCoarseCentroids,
//      const float* const __restrict pqFineCentroids,
//      const uint8_t* const __restrict code0,
//      const float weight0,
//      const uint8_t* const __restrict code1,
//      const float weight1,
//      float* const __restrict outputAccum);
//   }
//
// * And one more overload for ::accum() that decodes and accumulates
//   three vectors per call.
//   {
//     const faiss::Index* const index;
//     const uint8_t* const input0;
//     float weight0;
//     const uint8_t* const input1;
//     float weight1;
//     const uint8_t* const input2;
//     float weight2;
//
//     std::vector<float> buffer(d, 0);
//
//     index->sa_decode(1, input0, buffer.data());
//     for (size_t iDim = 0; iDim < d; iDim++)
//       output[iDim] += weight0 * buffer[iDim];
//
//     index->sa_decode(1, input1, buffer.data());
//     for (size_t iDim = 0; iDim < d; iDim++)
//       output[iDim] += weight1 * buffer[iDim];
//
//     index->sa_decode(1, input2, buffer.data());
//     for (size_t iDim = 0; iDim < d; iDim++)
//       output[iDim] += weight2 * buffer[iDim];
//   }
//
//   If each code uses its own coarse quantizer centroids table and its own fine
//   quantizer centroids table, then the following overload can be used:
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
//      const float* const __restrict pqCoarseCentroids2,
//      const float* const __restrict pqFineCentroids2,
//      const uint8_t* const __restrict code2,
//      const float weight2,
//      float* const __restrict outputAccum);
//   }
//   If codes share the coarse quantizer centroids table and also share
//   the fine quantizer centroids table, then the following overload can be
//   used:
//   {
//    static void accum(
//      const float* const __restrict pqCoarseCentroids,
//      const float* const __restrict pqFineCentroids,
//      const uint8_t* const __restrict code0,
//      const float weight0,
//      const uint8_t* const __restrict code1,
//      const float weight1,
//      const uint8_t* const __restrict code2,
//      const float weight2,
//      float* const __restrict outputAccum);
//   }
//
// The provided version is not multithreaded.
//
// Currently, an AVX2+FMA implementation is available. AVX512 version is also
//   doable, but it was found to be slower than AVX2 for real world applications
//   that I needed.
//
////////////////////////////////////////////////////////////////////////////////////
//
// It is possible to use an additional index wrapper on top of IVFPQ /
// Residual+PQ, known as IndexRowwiseMinMax / IndexRowwiseMinMaxFP16. Index
// wrapper that performs rowwise normalization to [0,1], preserving the
// coefficients. This is a vector codec index only.
// For more details please refer to the description in
// faiss/IndexRowwiseMinMax.h file.
//
// If such a wrapper is used, then the quantizer will look like, say,
//    MinMaxFP16,IVF256,PQ32np
//  or
//    MinMax,PQ16np
// In this case, please use the following contruction for the decoding,
// basically, wrapping a kernel in a kernel:
//   {
//      using SubT = faiss::cppcontrib::Index2LevelDecoder<128, 128, 2>;
//      using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
//      // do T::store(...) or T::accum(...)
//   }
//
// T::accum(...) contains an additional function variable which is
// used for accumulating scaling. Thus, the code pattern is the following:
//   {
//     const float* const __restrict pqCoarseCentroidsQ;
//     const float* const __restrict pqFineCentroidsQ;
//     const uint8_t* const __restrict input;
//     const float* const __restrict weights;
//     float* const __restrict output;
//     float outputAccumMin = 0;
//
//     for (size_t i = 0; i < n; i++) {
//         T::accum(
//                 pqCoarseCentroidsQ,
//                 pqFineCentroidsQ,
//                 input + i * code_size,
//                 weights[i],
//                 output,
//                 outputAccumMin);
//     }
//     for (size_t j = 0; j < d; j++)
//         output[j] += outputAccumMin;
//   }
// This is similar to the following regular pseudo-code:
//   {
//     const faiss::Index* const index;
//     const uint8_t* const __restrict input;
//     const float* const __restrict weights;
//     float* const __restrict output;
//
//     for (size_t i = 0; i < n; i++) {
//       std::vector<float> buffer(d, 0);
//
//       index->sa_decode(1, input + i * code_size, buffer.data());
//       for (size_t j = 0; j < d; j++)
//         output[j] += weights[i] * buffer[j];
//     }

#include <faiss/cppcontrib/sa_decode/MinMax-inl.h>
#include <faiss/cppcontrib/sa_decode/MinMaxFP16-inl.h>

#ifdef __AVX2__
#include <faiss/cppcontrib/sa_decode/Level2-avx2-inl.h>
#include <faiss/cppcontrib/sa_decode/PQ-avx2-inl.h>
#elif defined(__ARM_NEON)
#include <faiss/cppcontrib/sa_decode/Level2-neon-inl.h>
#include <faiss/cppcontrib/sa_decode/PQ-neon-inl.h>
#else
#include <faiss/cppcontrib/sa_decode/Level2-inl.h>
#include <faiss/cppcontrib/sa_decode/PQ-inl.h>
#endif
