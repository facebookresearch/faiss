/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PQ_AVX2_INL_H
#define PQ_AVX2_INL_H

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <faiss/cppcontrib/detail/UintReader.h>

namespace faiss {
namespace cppcontrib {

////////////////////////////////////////////////////////////////////////////////////
/// IndexPQDecoder
////////////////////////////////////////////////////////////////////////////////////

namespace {

// Despite the following functions are somewhat redundant, I'd like to keep the
// overall basic blocks similar to ones from Index2LevelDecoder.
// A compiler will optimize away the redundant code.

// Processes 8 float values.
// Returns {
//   [0..1] = *fine0[0..1];
//   [2..3] = *fine1[0..1];
//   [4..5] = *fine2[0..1];
//   [6..7] = *fine3[0..1];
// }
inline __m256 elementaryBlock2x4b(
        const float* const __restrict fine0,
        const float* const __restrict fine1,
        const float* const __restrict fine2,
        const float* const __restrict fine3) {
    // load fine
    const __m256 fineValue = _mm256_castpd_ps(_mm256_setr_pd(
            *reinterpret_cast<const double*>(fine0),
            *reinterpret_cast<const double*>(fine1),
            *reinterpret_cast<const double*>(fine2),
            *reinterpret_cast<const double*>(fine3)));

    // add coarse and fine
    return fineValue;
}

// Processes 8 float values.
// Returns {
//   [0..1] = existingValue[0..1] + weight * (*fine0[0..1]);
//   [2..3] = existingValue[0..1] + weight * (*fine1[0..1]);
//   [4..5] = existingValue[0..1] + weight * (*fine2[0..1]);
//   [6..7] = existingValue[0..1] + weight * (*fine3[0..1]);
// }
inline __m256 elementaryBlock2x4bAccum(
        const float* const __restrict fine0,
        const float* const __restrict fine1,
        const float* const __restrict fine2,
        const float* const __restrict fine3,
        const float weight,
        const __m256 existingValue) {
    // add coarse and fine
    const __m256 fineValue = elementaryBlock2x4b(fine0, fine1, fine2, fine3);

    // this operation is expected to be optimized by a compiler
    const __m256 weightAvx2 = _mm256_set1_ps(weight);
    // do fma
    return _mm256_fmadd_ps(fineValue, weightAvx2, existingValue);
}

// Processes 4 float values.
// Returns {
//   [0..3] = *fine[0..3];
// }
inline __m128 elementaryBlock4x1b(const float* const __restrict fine) {
    // load fine
    const __m128 fineValue = _mm_loadu_ps(fine);
    return fineValue;
}

// Processes 4 float values.
// Returns {
//   [0..3] = existingValue[0..3] + weight * (*fine[0..3]);
// }
inline __m128 elementaryBlock4x1bAccum(
        const float* const __restrict fine,
        const float weight,
        const __m128 existingValue) {
    const __m128 fineValue = elementaryBlock4x1b(fine);

    // this operation is expected to be optimized by a compiler
    const __m128 weightAvx = _mm_set1_ps(weight);
    // do fma
    return _mm_fmadd_ps(fineValue, weightAvx, existingValue);
}

// Processes 8 float values.
// Returns {
//   [0..3] = *fine0[0..3];
//   [4..7] = *fine1[0..3];
// }
inline __m256 elementaryBlock4x2b(
        const float* const __restrict fine0,
        const float* const __restrict fine1) {
    // load fine
    const __m128 fineValue0 = _mm_loadu_ps(fine0);
    const __m128 fineValue1 = _mm_loadu_ps(fine1);

    // combine two 4b into a single 8b
    const __m256 combinedFineValue = _mm256_set_m128(fineValue1, fineValue0);
    return combinedFineValue;
}

// Processes 8 float values.
// Returns {
//   [0..3] = existingValue[0..3] + weight * (*fine0[0..3]);
//   [4..7] = existingValue[4..7] + weight * (*fine1[0..3]);
// }
inline __m256 elementaryBlock4x2bAccum(
        const float* const __restrict fine0,
        const float* const __restrict fine1,
        const float weight,
        const __m256 existingValue) {
    const __m256 fineValue = elementaryBlock4x2b(fine0, fine1);

    // this operation is expected to be optimized by a compiler
    const __m256 weightAvx2 = _mm256_set1_ps(weight);
    // do fma
    return _mm256_fmadd_ps(fineValue, weightAvx2, existingValue);
}

// Processes 8 float values.
// Returns {
//   [0..7] = *fine[0..7];
// }
inline __m256 elementaryBlock8x1b(const float* const __restrict fine) {
    // load fine
    const __m256 fineValue = _mm256_loadu_ps(fine);
    return fineValue;
}

// Processes 8 float values.
// Returns {
//   [0..7] = existingValue[0..7] + weight * (*fine[0..7]);
// }
inline __m256 elementaryBlock8x1bAccum(
        const float* const __restrict fine,
        const float weight,
        const __m256 existingValue) {
    const __m256 fineValue = elementaryBlock8x1b(fine);

    // this operation is expected to be optimized by a compiler
    const __m256 weightAvx2 = _mm256_set1_ps(weight);
    // do fma
    return _mm256_fmadd_ps(fineValue, weightAvx2, existingValue);
}

// The following code uses template-based for-loop unrolling,
//   because the compiler does not do that on its own as needed.
// The idea is the following:
//   template<int I, int MAX>
//   struct Foo {
//     static void bar() {
//       doSomething(I);
//       Foo<I + 1, MAX>::bar();
//     }
//   };
//
//   template<int MAX>
//   struct Foo<MAX, MAX> {
//     static void bar() {}
//   };
//
//   Initiate the loop:
//     Foo<0, MAX>::bar();

template <
        intptr_t DIM,
        intptr_t FINE_SIZE,
        intptr_t FINE_BITS,
        intptr_t CPOS,
        bool FINE_SIZE_EQ_2 = FINE_SIZE == 2,
        bool FINE_SIZE_EQ_4 = FINE_SIZE == 4,
        bool QPOS_LEFT_GE_8 = (FINE_SIZE - CPOS % FINE_SIZE >= 8),
        bool QPOS_LEFT_GE_4 = (FINE_SIZE - CPOS % FINE_SIZE >= 4),
        bool DIM_EQ_CPOS = DIM == CPOS>
struct IndexPQDecoderImpl;

template <
        intptr_t DIM,
        intptr_t FINE_BITS,
        intptr_t CPOS,
        bool QPOS_LEFT_GE_8,
        bool QPOS_LEFT_GE_4>
struct IndexPQDecoderImpl<
        DIM,
        2,
        FINE_BITS,
        CPOS,
        true,
        false,
        QPOS_LEFT_GE_8,
        QPOS_LEFT_GE_4,
        false> {
    static constexpr intptr_t FINE_SIZE = 2;

    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    static constexpr intptr_t FINE_TABLE_BYTES = (1 << FINE_BITS);

    // process 1 sample
    static void store(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode0c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine0);
        const intptr_t fineCode0d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine0);

        const __m256 storeValue = elementaryBlock2x4b(
              pqFineCentroids0 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode0c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode0d) * FINE_SIZE + fineCentroidOffset);

        _mm256_storeu_ps(outputStore + CPOS, storeValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::store(
              pqFineCentroids0, code0, outputStore);

        // clang-format on
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode0c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine0);
        const intptr_t fineCode0d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine0);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids0 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode0c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode0d) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0, outputAccum);

        // clang-format on
    }

    // Process 2 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode0c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine0);
        const intptr_t fineCode0d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine0);
        const intptr_t fineCode1a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine1);
        const intptr_t fineCode1b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine1);
        const intptr_t fineCode1c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine1);
        const intptr_t fineCode1d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine1);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids0 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode0c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode0d) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids1 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode1a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids1 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode1b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids1 + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode1c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids1 + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode1d) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0,
              pqFineCentroids1, code1, weight1,
              outputAccum);

        // clang-format on
    }

    // Process 2 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode0c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine0);
        const intptr_t fineCode0d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine0);
        const intptr_t fineCode1a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine1);
        const intptr_t fineCode1b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine1);
        const intptr_t fineCode1c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine1);
        const intptr_t fineCode1d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine1);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode0c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode0d) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode1a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode1b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode1c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode1d) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids,
              code0, weight0,
              code1, weight1,
              outputAccum);

        // clang-format on
    }

    // Process 3 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            const float* const __restrict pqFineCentroids2,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode0c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine0);
        const intptr_t fineCode0d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine0);
        const intptr_t fineCode1a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine1);
        const intptr_t fineCode1b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine1);
        const intptr_t fineCode1c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine1);
        const intptr_t fineCode1d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine1);
        const intptr_t fineCode2a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine2);
        const intptr_t fineCode2b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine2);
        const intptr_t fineCode2c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine2);
        const intptr_t fineCode2d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine2);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids0 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode0c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode0d) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids1 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode1a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids1 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode1b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids1 + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode1c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids1 + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode1d) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids2 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode2a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids2 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode2b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids2 + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode2c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids2 + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode2d) * FINE_SIZE + fineCentroidOffset,
              weight2,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0,
              pqFineCentroids1, code1, weight1,
              pqFineCentroids2, code2, weight2,
              outputAccum);

        // clang-format on
    }

    // Process 3 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode0c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine0);
        const intptr_t fineCode0d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine0);
        const intptr_t fineCode1a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine1);
        const intptr_t fineCode1b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine1);
        const intptr_t fineCode1c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine1);
        const intptr_t fineCode1d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine1);
        const intptr_t fineCode2a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine2);
        const intptr_t fineCode2b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine2);
        const intptr_t fineCode2c = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 2>::get(fine2);
        const intptr_t fineCode2d = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 3>::get(fine2);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode0c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode0d) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode1a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode1b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode1c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode1d) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        existingValue = elementaryBlock2x4bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode2a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode2b) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 2) * FINE_TABLE_BYTES + fineCode2c) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 3) * FINE_TABLE_BYTES + fineCode2d) * FINE_SIZE + fineCentroidOffset,
              weight2,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids,
              code0, weight0,
              code1, weight1,
              code2, weight2,
              outputAccum);

        // clang-format on
    }
};

template <
        intptr_t DIM,
        intptr_t FINE_BITS,
        intptr_t CPOS,
        bool QPOS_LEFT_GE_8,
        bool QPOS_LEFT_GE_4>
struct IndexPQDecoderImpl<
        DIM,
        4,
        FINE_BITS,
        CPOS,
        false,
        true,
        QPOS_LEFT_GE_8,
        QPOS_LEFT_GE_4,
        false> {
    static constexpr intptr_t FINE_SIZE = 4;

    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    static constexpr intptr_t FINE_TABLE_BYTES = (1 << FINE_BITS);

    // process 1 sample
    static void store(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);

        const __m256 storeValue = elementaryBlock4x2b(
              pqFineCentroids0 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset);

        _mm256_storeu_ps(outputStore + CPOS, storeValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::store(
              pqFineCentroids0, code0, outputStore);

        // clang-format on
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids0 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0, outputAccum);

        // clang-format on
    }

    // Process 2 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode1a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine1);
        const intptr_t fineCode1b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine1);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids0 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids1 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode1a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids1 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode1b) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0,
              pqFineCentroids1, code1, weight1,
              outputAccum);

        // clang-format on
    }

    // Process 2 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode1a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine1);
        const intptr_t fineCode1b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine1);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode1a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode1b) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids,
              code0, weight0,
              code1, weight1,
              outputAccum);

        // clang-format on
    }

    // Process 3 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            const float* const __restrict pqFineCentroids2,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode1a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine1);
        const intptr_t fineCode1b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine1);
        const intptr_t fineCode2a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine2);
        const intptr_t fineCode2b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine2);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids0 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids0 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids1 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode1a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids1 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode1b) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids2 + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode2a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids2 + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode2b) * FINE_SIZE + fineCentroidOffset,
              weight2,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0,
              pqFineCentroids1, code1, weight1,
              pqFineCentroids2, code2, weight2,
              outputAccum);

        // clang-format on
    }

    // Process 3 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

        // clang-format off

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine0);
        const intptr_t fineCode0b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine0);
        const intptr_t fineCode1a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine1);
        const intptr_t fineCode1b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine1);
        const intptr_t fineCode2a = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(fine2);
        const intptr_t fineCode2b = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(fine2);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode0a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode0b) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode1a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode1b) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        existingValue = elementaryBlock4x2bAccum(
              pqFineCentroids + ((fineCentroidIdx + 0) * FINE_TABLE_BYTES + fineCode2a) * FINE_SIZE + fineCentroidOffset,
              pqFineCentroids + ((fineCentroidIdx + 1) * FINE_TABLE_BYTES + fineCode2b) * FINE_SIZE + fineCentroidOffset,
              weight2,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids,
              code0, weight0,
              code1, weight1,
              code2, weight2,
              outputAccum);

        // clang-format on
    }
};

template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t FINE_BITS, intptr_t CPOS>
struct IndexPQDecoderImpl<
        DIM,
        FINE_SIZE,
        FINE_BITS,
        CPOS,
        false,
        false,
        true,
        true,
        false> {
    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    static constexpr intptr_t FINE_TABLE_BYTES = (1 << FINE_BITS);

    // process 1 sample
    static void store(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // clang-format off

        // process chunks, 8 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);

        const __m256 storeValue = elementaryBlock8x1b(
              pqFineCentroids0 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset);

        _mm256_storeu_ps(outputStore + CPOS, storeValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::store(
              pqFineCentroids0, code0, outputStore);

        // clang-format on
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // clang-format off

        // process chunks, 8 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids0 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0, outputAccum);

        // clang-format on
    }

    // Process 2 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;

        // clang-format off

        // process chunks, 8 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);
        const intptr_t fineCode1 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine1);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids0 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids1 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0,
              pqFineCentroids1, code1, weight1,
              outputAccum);

        // clang-format on
    }

    // Process 2 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;

        // clang-format off

        // process chunks, 8 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);
        const intptr_t fineCode1 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine1);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids,
              code0, weight0,
              code1, weight1,
              outputAccum);

        // clang-format on
    }

    // Process 3 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            const float* const __restrict pqFineCentroids2,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

        // clang-format off

        // process chunks, 8 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);
        const intptr_t fineCode1 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine1);
        const intptr_t fineCode2 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine2);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids0 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids1 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids2 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) * FINE_SIZE + fineCentroidOffset,
              weight2,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids0, code0, weight0,
              pqFineCentroids1, code1, weight1,
              pqFineCentroids2, code2, weight2,
              outputAccum);

        // clang-format on
    }

    // Process 3 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

        // clang-format off

        // process chunks, 8 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);
        const intptr_t fineCode1 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine1);
        const intptr_t fineCode2 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine2);

        __m256 existingValue = _mm256_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        existingValue = elementaryBlock8x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) * FINE_SIZE + fineCentroidOffset,
              weight2,
              existingValue);

        _mm256_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
              pqFineCentroids,
              code0, weight0,
              code1, weight1,
              code2, weight2,
              outputAccum);

        // clang-format on
    }
};

template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t FINE_BITS, intptr_t CPOS>
struct IndexPQDecoderImpl<
        DIM,
        FINE_SIZE,
        FINE_BITS,
        CPOS,
        false,
        false,
        false,
        true,
        false> {
    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    static constexpr intptr_t FINE_TABLE_BYTES = (1 << FINE_BITS);

    // process 1 sample
    static void store(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // clang-format off

        // process chunks, 4 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);

        const __m128 storeValue = elementaryBlock4x1b(
              pqFineCentroids0 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset);

        _mm_storeu_ps(outputStore + CPOS, storeValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::store(
              pqFineCentroids0, code0, outputStore);

        // clang-format on
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // clang-format off

        // process chunks, 4 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);

        __m128 existingValue = _mm_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids0 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        _mm_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
              pqFineCentroids0, code0, weight0, outputAccum);

        // clang-format on
    }

    // Process 2 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;

        // clang-format off

        // process chunks, 4 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);
        const intptr_t fineCode1 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine1);

        __m128 existingValue = _mm_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids0 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids1 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        _mm_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
              pqFineCentroids0, code0, weight0,
              pqFineCentroids1, code1, weight1,
              outputAccum);

        // clang-format on
    }

    // Process 2 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;

        // clang-format off

        // process chunks, 4 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);
        const intptr_t fineCode1 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine1);

        __m128 existingValue = _mm_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        _mm_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
              pqFineCentroids,
              code0, weight0,
              code1, weight1,
              outputAccum);

        // clang-format on
    }

    // Process 3 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            const float* const __restrict pqFineCentroids2,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

        // clang-format off

        // process chunks, 4 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);
        const intptr_t fineCode1 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine1);
        const intptr_t fineCode2 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine2);

        __m128 existingValue = _mm_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids0 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids1 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids2 + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) * FINE_SIZE + fineCentroidOffset,
              weight2,
              existingValue);

        _mm_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
              pqFineCentroids0, code0, weight0,
              pqFineCentroids1, code1, weight1,
              pqFineCentroids2, code2, weight2,
              outputAccum);

        // clang-format on
    }

    // Process 3 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

        // clang-format off

        // process chunks, 4 float

        const intptr_t fineCode0 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine0);
        const intptr_t fineCode1 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine1);
        const intptr_t fineCode2 = detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::get(fine2);

        __m128 existingValue = _mm_loadu_ps(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE + fineCentroidOffset,
              weight0,
              existingValue);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) * FINE_SIZE + fineCentroidOffset,
              weight1,
              existingValue);

        existingValue = elementaryBlock4x1bAccum(
              pqFineCentroids + (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) * FINE_SIZE + fineCentroidOffset,
              weight2,
              existingValue);

        _mm_storeu_ps(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
              pqFineCentroids, code0, weight0,
              code1, weight1,
              code2, weight2,
              outputAccum);

        // clang-format on
    }
};

// This partial specialization is expected to do nothing.
template <
        intptr_t DIM,
        intptr_t FINE_SIZE,
        intptr_t FINE_BITS,
        bool FINE_SIZE_EQ_2,
        bool FINE_SIZE_EQ_4,
        bool QPOS_LEFT_GE_8,
        bool QPOS_LEFT_GE_4>
struct IndexPQDecoderImpl<
        DIM,
        FINE_SIZE,
        FINE_BITS,
        DIM,
        FINE_SIZE_EQ_2,
        FINE_SIZE_EQ_4,
        QPOS_LEFT_GE_8,
        QPOS_LEFT_GE_4,
        true> {
    // clang-format off

    // process 1 sample
    static void store(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {}

    // process 1 sample
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {}

    // Process 2 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {}

    // Process 2 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {}

    // Process 3 samples.
    // Each code uses its own fine pq centroids table.
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            const float* const __restrict pqFineCentroids2,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {}

    // Process 3 samples.
    // Fine pq centroids table is shared among codes.
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {}

    // clang-format on
};

} // namespace

// Suitable for PQ[1]x8
// Suitable for PQ[1]x10
// Suitable for PQ[1]x12
// Suitable for PQ[1]x16
template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t FINE_BITS = 8>
struct IndexPQDecoder {
    static_assert(
            FINE_BITS == 8 || FINE_BITS == 10 || FINE_BITS == 12 ||
                    FINE_BITS == 16,
            "Only 8, 10, 12 or 16 bits are currently supported for FINE_BITS");

    static constexpr intptr_t dim = DIM;
    static constexpr intptr_t fineSize = FINE_SIZE;
    static constexpr intptr_t fineBits = FINE_BITS;

    // Process 1 sample.
    static void store(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, 0>::store(
                pqFineCentroids, code, outputStore);
    }

    // Process 1 sample.
    // Performs outputAccum += weight * decoded(code)
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            const float weight,
            float* const __restrict outputAccum) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, 0>::accum(
                pqFineCentroids, code, weight, outputAccum);
    }

    // Process 2 samples.
    // Each code uses its own fine pq centroids table.
    //
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    //   decoded(code1)
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, 0>::accum(
                pqFineCentroids0,
                code0,
                weight0,
                pqFineCentroids1,
                code1,
                weight1,
                outputAccum);
    }

    // Process 2 samples.
    // Fine pq centroids table is shared among codes.
    //
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    //   decoded(code1)
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, 0>::accum(
                pqFineCentroids, code0, weight0, code1, weight1, outputAccum);
    }

    // Process 3 samples.
    // Each code uses its own fine pq centroids table.
    //
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    //   decoded(code1) + weight2 * decoded(code2)
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            const float* const __restrict pqFineCentroids2,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, 0>::accum(
                pqFineCentroids0,
                code0,
                weight0,
                pqFineCentroids1,
                code1,
                weight1,
                pqFineCentroids2,
                code2,
                weight2,
                outputAccum);
    }

    // Process 3 samples.
    // Fine pq centroids table is shared among codes.
    //
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    //   decoded(code1) + weight2 * decoded(code2)
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, 0>::accum(
                pqFineCentroids,
                code0,
                weight0,
                code1,
                weight1,
                code2,
                weight2,
                outputAccum);
    }
};

} // namespace cppcontrib
} // namespace faiss
#endif // PQ_AVX2_INL_H
