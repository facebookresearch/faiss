/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PQ_NEON_INL_H
#define PQ_NEON_INL_H

#include <arm_neon.h>

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

// Processes 4 float values.
// Returns {
//   [0..3] = *fine[0..3];
// }
inline float32x4_t elementaryBlock4x1b(const float* const __restrict fine) {
    // load fine
    const auto fineValue = vld1q_f32(fine);
    return fineValue;
}

// Processes 4 float values.
// Returns {
//   [0..3] = existingValue[0..3] + weight * (*fine[0..3]);
// }
inline float32x4_t elementaryBlock4x1bAccum(
        const float* const __restrict fine,
        const float weight,
        const float32x4_t existingValue) {
    const auto fineValue = elementaryBlock4x1b(fine);

    // this operation is expected to be optimized by a compiler
    const auto weightNeon = vdupq_n_f32(weight);
    // do fma
    return vfmaq_f32(existingValue, weightNeon, fineValue);
}

// Processes 8 float values.
// Returns {
//   [0..3] = *fine0[0..3];
//   [4..7] = *fine1[0..3];
// }
inline float32x4x2_t elementaryBlock4x2b(
        const float* const __restrict fine0,
        const float* const __restrict fine1) {
    // load fine
    const auto fineValue0 = vld1q_f32(fine0);
    const auto fineValue1 = vld1q_f32(fine1);

    return {fineValue0, fineValue1};
}

// Processes 8 float values.
// Returns {
//   [0..3] = existingValue[0..3] + weight * (*fine0[0..3]);
//   [4..7] = existingValue[4..7] + weight * (*fine1[0..3]);
// }
inline float32x4x2_t elementaryBlock4x2bAccum(
        const float* const __restrict fine0,
        const float* const __restrict fine1,
        const float weight,
        const float32x4x2_t existingValue) {
    const auto fineValue = elementaryBlock4x2b(fine0, fine1);

    // this operation is expected to be optimized by a compiler
    const auto weightNeon = vdupq_n_f32(weight);
    // do fma
    const auto result0 =
            vfmaq_f32(existingValue.val[0], weightNeon, fineValue.val[0]);
    const auto result1 =
            vfmaq_f32(existingValue.val[1], weightNeon, fineValue.val[1]);
    return {result0, result1};
}

// Processes 8 float values.
// Returns {
//   [0..7] = *fine[0..7];
// }
inline float32x4x2_t elementaryBlock8x1b(const float* const __restrict fine) {
    // load fine
    const auto fineValue0 = vld1q_f32(fine);
    const auto fineValue1 = vld1q_f32(fine + 4);
    return {fineValue0, fineValue1};
}

// Processes 8 float values.
// Returns {
//   [0..7] = existingValue[0..7] + weight * (*fine[0..7]);
// }
inline float32x4x2_t elementaryBlock8x1bAccum(
        const float* const __restrict fine,
        const float weight,
        const float32x4x2_t existingValue) {
    const auto fineValue = elementaryBlock8x1b(fine);

    // this operation is expected to be optimized by a compiler
    const auto weightNeon = vdupq_n_f32(weight);
    // do fma
    const auto result0 =
            vfmaq_f32(existingValue.val[0], weightNeon, fineValue.val[0]);
    const auto result1 =
            vfmaq_f32(existingValue.val[1], weightNeon, fineValue.val[1]);
    return {result0, result1};
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
        bool FINE_SIZE_EQ_4 = FINE_SIZE == 4,
        bool QPOS_LEFT_GE_8 = (FINE_SIZE - CPOS % FINE_SIZE >= 8),
        bool QPOS_LEFT_GE_4 = (FINE_SIZE - CPOS % FINE_SIZE >= 4),
        bool DIM_EQ_CPOS = DIM == CPOS>
struct IndexPQDecoderImpl;

template <
        intptr_t DIM,
        intptr_t CPOS,
        intptr_t FINE_BITS,
        bool QPOS_LEFT_GE_8,
        bool QPOS_LEFT_GE_4>
struct IndexPQDecoderImpl<
        DIM,
        4,
        FINE_BITS,
        CPOS,
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

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine0);

        const auto storeValue = elementaryBlock4x2b(
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode0a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode0b) *
                                FINE_SIZE +
                        fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue.val[0]);
        vst1q_f32(outputStore + CPOS + 4, storeValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::store(
                pqFineCentroids0, code0, outputStore);
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine0);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode0a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode0b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
                pqFineCentroids0, code0, weight0, outputAccum);
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

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine0);
        const intptr_t fineCode1a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine1);
        const intptr_t fineCode1b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine1);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode0a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode0b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids1 +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode1a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids1 +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode1b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
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

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine0);
        const intptr_t fineCode1a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine1);
        const intptr_t fineCode1b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine1);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode0a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode0b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode1a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode1b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
                pqFineCentroids, code0, weight0, code1, weight1, outputAccum);
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

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine0);
        const intptr_t fineCode1a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine1);
        const intptr_t fineCode1b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine1);
        const intptr_t fineCode2a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine2);
        const intptr_t fineCode2b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine2);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode0a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode0b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids1 +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode1a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids1 +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode1b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids2 +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode2a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids2 +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode2b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight2,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
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

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine0);
        const intptr_t fineCode1a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine1);
        const intptr_t fineCode1b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine1);
        const intptr_t fineCode2a = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 0>::get(
                        fine2);
        const intptr_t fineCode2b = detail::
                UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx + 1>::get(
                        fine2);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode0a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode0b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode1a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode1b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids +
                        ((fineCentroidIdx + 0) * FINE_TABLE_BYTES +
                         fineCode2a) *
                                FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids +
                        ((fineCentroidIdx + 1) * FINE_TABLE_BYTES +
                         fineCode2b) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight2,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
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

template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t FINE_BITS, intptr_t CPOS>
struct IndexPQDecoderImpl<
        DIM,
        FINE_SIZE,
        FINE_BITS,
        CPOS,
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

        // process chunks, 8 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);

        const auto storeValue = elementaryBlock8x1b(
                pqFineCentroids0 +
                (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE +
                fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue.val[0]);
        vst1q_f32(outputStore + CPOS + 4, storeValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::store(
                pqFineCentroids0, code0, outputStore);
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // process chunks, 8 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        const auto existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
                pqFineCentroids0, code0, weight0, outputAccum);
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

        // process chunks, 8 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine1);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids1 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
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

        // process chunks, 8 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine1);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
                pqFineCentroids, code0, weight0, code1, weight1, outputAccum);
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

        // process chunks, 8 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine1);
        const intptr_t fineCode2 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine2);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids1 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids2 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight2,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
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

        // process chunks, 8 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine1);
        const intptr_t fineCode2 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine2);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight2,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 8>::accum(
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

template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t FINE_BITS, intptr_t CPOS>
struct IndexPQDecoderImpl<
        DIM,
        FINE_SIZE,
        FINE_BITS,
        CPOS,
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

        // process chunks, 4 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);

        const auto storeValue = elementaryBlock4x1b(
                pqFineCentroids0 +
                (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) * FINE_SIZE +
                fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::store(
                pqFineCentroids0, code0, outputStore);
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // process chunks, 4 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
                pqFineCentroids0, code0, weight0, outputAccum);
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

        // process chunks, 4 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine1);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids1 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
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

        // process chunks, 4 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine1);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
                pqFineCentroids, code0, weight0, code1, weight1, outputAccum);
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

        // process chunks, 4 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine1);
        const intptr_t fineCode2 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine2);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids1 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids2 +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight2,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
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

        // process chunks, 4 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine1);
        const intptr_t fineCode2 =
                detail::UintReader<DIM, FINE_SIZE, FINE_BITS, fineCentroidIdx>::
                        get(fine2);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids +
                        (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) *
                                FINE_SIZE +
                        fineCentroidOffset,
                weight2,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, FINE_BITS, CPOS + 4>::accum(
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

// This partial specialization is expected to do nothing.
template <
        intptr_t DIM,
        intptr_t FINE_SIZE,
        intptr_t FINE_BITS,
        bool FINE_SIZE_EQ_4,
        bool QPOS_LEFT_GE_8,
        bool QPOS_LEFT_GE_4>
struct IndexPQDecoderImpl<
        DIM,
        FINE_SIZE,
        FINE_BITS,
        DIM,
        FINE_SIZE_EQ_4,
        QPOS_LEFT_GE_8,
        QPOS_LEFT_GE_4,
        true> {
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
#endif // PQ_NEON_INL_H
