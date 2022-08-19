// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include <faiss/cppcontrib/detail/CoarseBitType.h>
#include <faiss/cppcontrib/detail/UintReader.h>

namespace faiss {
namespace cppcontrib {

namespace {

// Processes 4 float values.
// Returns {
//   [0..3] = *coarse[0..3] + *fine[0..3];
// }
inline float32x4_t elementaryBlock4x1b(
        const float* const __restrict coarse,
        const float* const __restrict fine) {
    // load fine
    const auto fineValue = vld1q_f32(fine);
    // load coarse
    const auto coarseValue = vld1q_f32(coarse);

    // add coarse and fine
    return vaddq_f32(fineValue, coarseValue);
}

// Processes 4 float values.
// Returns {
//   [0..3] = existingValue[0..3] + weight * (*coarse[0..3] + *fine[0..3]);
// }
inline float32x4_t elementaryBlock4x1bAccum(
        const float* const __restrict coarse,
        const float* const __restrict fine,
        const float weight,
        const float32x4_t existingValue) {
    // add coarse and fine
    const auto combinedValue = elementaryBlock4x1b(coarse, fine);

    // this operation is expected to be optimized by a compiler
    const auto weightNeon = vdupq_n_f32(weight);
    // do fma
    return vfmaq_f32(existingValue, weightNeon, combinedValue);
}

// Processes 8 float values.
// Returns {
//   [0..3] = *coarse[0..3] + *fine0[0..3];
//   [4..7] = *coarse[4..7] + *fine1[0..3];
// }
inline float32x4x2_t elementaryBlock4x2b(
        const float* const __restrict coarse,
        const float* const __restrict fine0,
        const float* const __restrict fine1) {
    // load fine
    const auto fineValue0 = vld1q_f32(fine0);
    const auto fineValue1 = vld1q_f32(fine1);
    // load coarse
    const auto coarseValue0 = vld1q_f32(coarse);
    const auto coarseValue1 = vld1q_f32(coarse + 4);

    // add coarse and fine
    const auto result0 = vaddq_f32(fineValue0, coarseValue0);
    const auto result1 = vaddq_f32(fineValue1, coarseValue1);

    return {result0, result1};
}

// Processes 8 float values.
// Returns {
//   [0..3] = existingValue[0..3] + weight * (*coarse[0..3] + *fine0[0..3]);
//   [4..7] = existingValue[4..7] + weight * (*coarse[4..7] + *fine1[0..3]);
// }
inline float32x4x2_t elementaryBlock4x2bAccum(
        const float* const __restrict coarse,
        const float* const __restrict fine0,
        const float* const __restrict fine1,
        const float weight,
        const float32x4x2_t existingValue) {
    // add coarse and fine
    const auto combinedValue = elementaryBlock4x2b(coarse, fine0, fine1);

    // this operation is expected to be optimized by a compiler
    const auto weightNeon = vdupq_n_f32(weight);
    // do fma
    const auto result0 =
            vfmaq_f32(existingValue.val[0], weightNeon, combinedValue.val[0]);
    const auto result1 =
            vfmaq_f32(existingValue.val[1], weightNeon, combinedValue.val[1]);
    return {result0, result1};
}

// Processes 8 float values.
// Returns {
//   [0..7] = *coarse[0..7] + *fine[0..7];
// }
inline float32x4x2_t elementaryBlock8x1b(
        const float* const __restrict coarse,
        const float* const __restrict fine) {
    // load fine
    const auto fineValue0 = vld1q_f32(fine);
    const auto fineValue1 = vld1q_f32(fine + 4);
    // load coarse
    const auto coarseValue0 = vld1q_f32(coarse);
    const auto coarseValue1 = vld1q_f32(coarse + 4);

    // add coarse and fine
    return {vaddq_f32(fineValue0, coarseValue0),
            vaddq_f32(fineValue1, coarseValue1)};
}

// Processes 8 float values.
// Returns {
//   [0..7] = existingValue[0..7] + weight * (*coarse[0..7] + *fine[0..7]);
// }
inline float32x4x2_t elementaryBlock8x1bAccum(
        const float* const __restrict coarse,
        const float* const __restrict fine,
        const float weight,
        const float32x4x2_t existingValue) {
    // add coarse and fine
    const auto combinedValue = elementaryBlock8x1b(coarse, fine);

    // this operation is expected to be optimized by a compiler
    const auto weightNeon = vdupq_n_f32(weight);
    // do fma
    const auto result0 =
            vfmaq_f32(existingValue.val[0], weightNeon, combinedValue.val[0]);
    const auto result1 =
            vfmaq_f32(existingValue.val[1], weightNeon, combinedValue.val[1]);
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
        intptr_t COARSE_SIZE,
        intptr_t FINE_SIZE,
        intptr_t COARSE_BITS,
        intptr_t CPOS,
        bool FINE_SIZE_EQ_4 = FINE_SIZE == 4,
        bool QPOS_LEFT_GE_8 = (FINE_SIZE - CPOS % FINE_SIZE >= 8),
        bool QPOS_LEFT_GE_4 = (FINE_SIZE - CPOS % FINE_SIZE >= 4),
        bool DIM_EQ_CPOS = DIM == CPOS>
struct Index2LevelDecoderImpl;

template <
        intptr_t DIM,
        intptr_t COARSE_SIZE,
        intptr_t COARSE_BITS,
        intptr_t CPOS,
        bool QPOS_LEFT_GE_8,
        bool QPOS_LEFT_GE_4>
struct Index2LevelDecoderImpl<
        DIM,
        COARSE_SIZE,
        4,
        COARSE_BITS,
        CPOS,
        true,
        QPOS_LEFT_GE_8,
        QPOS_LEFT_GE_4,
        false> {
    static constexpr intptr_t FINE_SIZE = 4;

    static constexpr intptr_t coarseCentroidIdx = CPOS / COARSE_SIZE;
    static constexpr intptr_t coarseCentroidOffset = CPOS % COARSE_SIZE;
    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    // coarse quantizer storage
    using coarse_storage_type =
            typename detail::CoarseBitType<COARSE_BITS>::bit_type;
    static constexpr intptr_t COARSE_TABLE_BYTES = (1 << COARSE_BITS);

    // process 1 sample
    static void store(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0a =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 1>::get(
                        fine0);

        const auto storeValue = elementaryBlock4x2b(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * 256 + fineCode0a) * FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * 256 + fineCode0b) * FINE_SIZE +
                        fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue.val[0]);
        vst1q_f32(outputStore + CPOS + 4, storeValue.val[1]);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 8>::
                store(pqCoarseCentroids0, pqFineCentroids0, code0, outputStore);
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0a =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 1>::get(
                        fine0);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * 256 + fineCode0a) * FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * 256 + fineCode0b) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 8>::
                accum(pqCoarseCentroids0,
                      pqFineCentroids0,
                      code0,
                      weight0,
                      outputAccum);
    }

    // process 2 samples
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqCoarseCentroids1,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;
        const uint8_t* const __restrict coarse1 = code1;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine1 =
                code1 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0a =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 1>::get(
                        fine0);
        const intptr_t coarseCode1 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse1);
        const intptr_t fineCode1a =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 0>::get(
                        fine1);
        const intptr_t fineCode1b =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 1>::get(
                        fine1);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * 256 + fineCode0a) * FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * 256 + fineCode0b) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock4x2bAccum(
                pqCoarseCentroids1 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode1) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids1 +
                        ((fineCentroidIdx + 0) * 256 + fineCode1a) * FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids1 +
                        ((fineCentroidIdx + 1) * 256 + fineCode1b) * FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 8>::
                accum(pqCoarseCentroids0,
                      pqFineCentroids0,
                      code0,
                      weight0,
                      pqCoarseCentroids1,
                      pqFineCentroids1,
                      code1,
                      weight1,
                      outputAccum);
    }
};

template <
        intptr_t DIM,
        intptr_t COARSE_SIZE,
        intptr_t FINE_SIZE,
        intptr_t COARSE_BITS,
        intptr_t CPOS>
struct Index2LevelDecoderImpl<
        DIM,
        COARSE_SIZE,
        FINE_SIZE,
        COARSE_BITS,
        CPOS,
        false,
        true,
        true,
        false> {
    static constexpr intptr_t coarseCentroidIdx = CPOS / COARSE_SIZE;
    static constexpr intptr_t coarseCentroidOffset = CPOS % COARSE_SIZE;
    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    // coarse quantizer storage
    using coarse_storage_type =
            typename detail::CoarseBitType<COARSE_BITS>::bit_type;
    static constexpr intptr_t COARSE_TABLE_BYTES = (1 << COARSE_BITS);

    // process 1 sample
    static void store(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 8 float

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);

        const auto storeValue = elementaryBlock8x1b(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue.val[0]);
        vst1q_f32(outputStore + CPOS + 4, storeValue.val[1]);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 8>::
                store(pqCoarseCentroids0, pqFineCentroids0, code0, outputStore);
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 8 float

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        const auto existingValue = elementaryBlock8x1bAccum(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 8>::
                accum(pqCoarseCentroids0,
                      pqFineCentroids0,
                      code0,
                      weight0,
                      outputAccum);
    }

    // process 2 samples
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqCoarseCentroids1,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;
        const uint8_t* const __restrict coarse1 = code1;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine1 =
                code1 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 8 float

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);
        const intptr_t coarseCode1 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse1);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine1);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock8x1bAccum(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock8x1bAccum(
                pqCoarseCentroids1 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode1) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids1 +
                        (fineCentroidIdx * 256 + fineCode1) * FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 8>::
                accum(pqCoarseCentroids0,
                      pqFineCentroids0,
                      code0,
                      weight0,
                      pqCoarseCentroids1,
                      pqFineCentroids1,
                      code1,
                      weight1,
                      outputAccum);
    }
};

template <
        intptr_t DIM,
        intptr_t COARSE_SIZE,
        intptr_t FINE_SIZE,
        intptr_t COARSE_BITS,
        intptr_t CPOS>
struct Index2LevelDecoderImpl<
        DIM,
        COARSE_SIZE,
        FINE_SIZE,
        COARSE_BITS,
        CPOS,
        false,
        false,
        true,
        false> {
    static constexpr intptr_t coarseCentroidIdx = CPOS / COARSE_SIZE;
    static constexpr intptr_t coarseCentroidOffset = CPOS % COARSE_SIZE;
    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    // coarse quantizer storage
    using coarse_storage_type =
            typename detail::CoarseBitType<COARSE_BITS>::bit_type;
    static constexpr intptr_t COARSE_TABLE_BYTES = (1 << COARSE_BITS);

    // process 1 sample
    static void store(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 4 float

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);

        const auto storeValue = elementaryBlock4x1b(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 4>::
                store(pqCoarseCentroids0, pqFineCentroids0, code0, outputStore);
    }

    // process 1 sample
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 4 float

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 4>::
                accum(pqCoarseCentroids0,
                      pqFineCentroids0,
                      code0,
                      weight0,
                      outputAccum);
    }

    // process 2 samples
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqCoarseCentroids1,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const uint8_t* const __restrict coarse0 = code0;
        const uint8_t* const __restrict coarse1 = code1;

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine1 =
                code1 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

        // process chunks, 4 float

        const intptr_t coarseCode0 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse0);
        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);
        const intptr_t coarseCode1 = detail::
                UintReader<DIM, COARSE_SIZE, COARSE_BITS, coarseCentroidIdx>::
                        get(coarse1);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine1);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqCoarseCentroids0 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        existingValue = elementaryBlock4x1bAccum(
                pqCoarseCentroids1 +
                        (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode1) *
                                COARSE_SIZE +
                        coarseCentroidOffset,
                pqFineCentroids1 +
                        (fineCentroidIdx * 256 + fineCode1) * FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        Index2LevelDecoderImpl<
                DIM,
                COARSE_SIZE,
                FINE_SIZE,
                COARSE_BITS,
                CPOS + 4>::
                accum(pqCoarseCentroids0,
                      pqFineCentroids0,
                      code0,
                      weight0,
                      pqCoarseCentroids1,
                      pqFineCentroids1,
                      code1,
                      weight1,
                      outputAccum);
    }
};

// This partial specialization is expected to do nothing.
template <
        intptr_t DIM,
        intptr_t COARSE_SIZE,
        intptr_t FINE_SIZE,
        intptr_t COARSE_BITS,
        bool FINE_SIZE_EQ_4,
        bool QPOS_LEFT_GE_8,
        bool QPOS_LEFT_GE_4>
struct Index2LevelDecoderImpl<
        DIM,
        COARSE_SIZE,
        FINE_SIZE,
        COARSE_BITS,
        DIM,
        FINE_SIZE_EQ_4,
        QPOS_LEFT_GE_8,
        QPOS_LEFT_GE_4,
        true> {
    // process 1 sample
    static void store(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {}

    // process 1 sample
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            float* const __restrict outputAccum) {}

    // process 2 samples
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqCoarseCentroids1,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {}
};
} // namespace

// Suitable for IVF256,PQ[1]x8
// Suitable for Residual[1]x8,PQ[2]x8
// Suitable for IVF[9-16 bit],PQ[1]x8 (such as IVF1024,PQ16np)
// Suitable for Residual1x[9-16 bit],PQ[1]x8 (such as Residual1x9,PQ8)
template <
        intptr_t DIM,
        intptr_t COARSE_SIZE,
        intptr_t FINE_SIZE,
        intptr_t COARSE_BITS = 8>
struct Index2LevelDecoder {
    static_assert(
            COARSE_BITS == 8 || COARSE_BITS == 16,
            "Only 8 or 16 bits are currently supported for COARSE_BITS");

    // Process 1 sample.
    static void store(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        Index2LevelDecoderImpl<DIM, COARSE_SIZE, FINE_SIZE, COARSE_BITS, 0>::
                store(pqCoarseCentroids, pqFineCentroids, code, outputStore);
    }

    // Process 1 sample.
    // Performs outputAccum += weight * decoded(code)
    static void accum(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            const float weight,
            float* const __restrict outputAccum) {
        Index2LevelDecoderImpl<DIM, COARSE_SIZE, FINE_SIZE, COARSE_BITS, 0>::
                accum(pqCoarseCentroids,
                      pqFineCentroids,
                      code,
                      weight,
                      outputAccum);
    }

    // process 2 samples
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    // decoded(code1)
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqCoarseCentroids1,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        Index2LevelDecoderImpl<DIM, COARSE_SIZE, FINE_SIZE, COARSE_BITS, 0>::
                accum(pqCoarseCentroids0,
                      pqFineCentroids0,
                      code0,
                      weight0,
                      pqCoarseCentroids1,
                      pqFineCentroids1,
                      code1,
                      weight1,
                      outputAccum);
    }
};

} // namespace cppcontrib
} // namespace faiss
