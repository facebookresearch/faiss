// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

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

    // add coarse and fine
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
        intptr_t CPOS,
        bool FINE_SIZE_EQ_4 = FINE_SIZE == 4,
        bool QPOS_LEFT_GE_8 = (FINE_SIZE - CPOS % FINE_SIZE >= 8),
        bool QPOS_LEFT_GE_4 = (FINE_SIZE - CPOS % FINE_SIZE >= 4),
        bool DIM_EQ_CPOS = DIM == CPOS>
struct IndexPQDecoderImpl;

template <intptr_t DIM, intptr_t CPOS, bool QPOS_LEFT_GE_8, bool QPOS_LEFT_GE_4>
struct IndexPQDecoderImpl<
        DIM,
        4,
        CPOS,
        true,
        QPOS_LEFT_GE_8,
        QPOS_LEFT_GE_4,
        false> {
    static constexpr intptr_t FINE_SIZE = 4;

    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    // process 1 sample
    static void store(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // process chunks, 4 float
        // but 8 floats per loop

        const intptr_t fineCode0a =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 1>::get(
                        fine0);

        const auto storeValue = elementaryBlock4x2b(
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * 256 + fineCode0a) * FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * 256 + fineCode0b) * FINE_SIZE +
                        fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue.val[0]);
        vst1q_f32(outputStore + CPOS + 4, storeValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 8>::store(
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

        const intptr_t fineCode0a =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 1>::get(
                        fine0);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
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
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 8>::accum(
                pqFineCentroids0, code0, weight0, outputAccum);
    }

    // process 2 samples
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

        const intptr_t fineCode0a =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 0>::get(
                        fine0);
        const intptr_t fineCode0b =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 1>::get(
                        fine0);
        const intptr_t fineCode1a =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 0>::get(
                        fine1);
        const intptr_t fineCode1b =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx + 1>::get(
                        fine1);

        auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock4x2bAccum(
                pqFineCentroids0 +
                        ((fineCentroidIdx + 0) * 256 + fineCode0a) * FINE_SIZE +
                        fineCentroidOffset,
                pqFineCentroids0 +
                        ((fineCentroidIdx + 1) * 256 + fineCode0b) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock4x2bAccum(
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
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 8>::accum(
                pqFineCentroids0,
                code0,
                weight0,
                pqFineCentroids1,
                code1,
                weight1,
                outputAccum);
    }
};

template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t CPOS>
struct IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS, false, true, true, false> {
    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    // process 1 sample
    static void store(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // process chunks, 8 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);

        const auto storeValue = elementaryBlock8x1b(
                pqFineCentroids0 +
                (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue.val[0]);
        vst1q_f32(outputStore + CPOS + 4, storeValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 8>::store(
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
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        const auto existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 8>::accum(
                pqFineCentroids0, code0, weight0, outputAccum);
    }

    // process 2 samples
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
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine1);

        const auto existingValue0 = vld1q_f32(outputAccum + CPOS);
        const auto existingValue1 = vld1q_f32(outputAccum + CPOS + 4);

        auto existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                {existingValue0, existingValue1});

        existingValue = elementaryBlock8x1bAccum(
                pqFineCentroids1 +
                        (fineCentroidIdx * 256 + fineCode1) * FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue.val[0]);
        vst1q_f32(outputAccum + CPOS + 4, existingValue.val[1]);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 8>::accum(
                pqFineCentroids0,
                code0,
                weight0,
                pqFineCentroids1,
                code1,
                weight1,
                outputAccum);
    }
};

template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t CPOS>
struct IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS, false, false, true, false> {
    static constexpr intptr_t fineCentroidIdx = CPOS / FINE_SIZE;
    static constexpr intptr_t fineCentroidOffset = CPOS % FINE_SIZE;

    static constexpr intptr_t QPOS_LEFT = FINE_SIZE - fineCentroidOffset;

    // process 1 sample
    static void store(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            float* const __restrict outputStore) {
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;

        // process chunks, 4 float

        const intptr_t fineCode0 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);

        const auto storeValue = elementaryBlock4x1b(
                pqFineCentroids0 +
                (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                fineCentroidOffset);

        vst1q_f32(outputStore + CPOS, storeValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 4>::store(
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
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 4>::accum(
                pqFineCentroids0, code0, weight0, outputAccum);
    }

    // process 2 samples
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
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine0);
        const intptr_t fineCode1 =
                detail::UintReader<DIM, FINE_SIZE, 8, fineCentroidIdx>::get(
                        fine1);

        auto existingValue = vld1q_f32(outputAccum + CPOS);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids0 +
                        (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                        fineCentroidOffset,
                weight0,
                existingValue);

        existingValue = elementaryBlock4x1bAccum(
                pqFineCentroids1 +
                        (fineCentroidIdx * 256 + fineCode1) * FINE_SIZE +
                        fineCentroidOffset,
                weight1,
                existingValue);

        vst1q_f32(outputAccum + CPOS, existingValue);

        // next
        IndexPQDecoderImpl<DIM, FINE_SIZE, CPOS + 4>::accum(
                pqFineCentroids0,
                code0,
                weight0,
                pqFineCentroids1,
                code1,
                weight1,
                outputAccum);
    }
};

// This partial specialization is expected to do nothing.
template <
        intptr_t DIM,
        intptr_t FINE_SIZE,
        bool FINE_SIZE_EQ_4,
        bool QPOS_LEFT_GE_8,
        bool QPOS_LEFT_GE_4>
struct IndexPQDecoderImpl<
        DIM,
        FINE_SIZE,
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

    // process 2 samples
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {}
};
} // namespace

// Suitable for PQ[1]x8
template <intptr_t DIM, intptr_t FINE_SIZE>
struct IndexPQDecoder {
    // Process 1 sample.
    static void store(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, 0>::store(
                pqFineCentroids, code, outputStore);
    }

    // Process 1 sample.
    // Performs outputAccum += weight * decoded(code)
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            const float weight,
            float* const __restrict outputAccum) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, 0>::accum(
                pqFineCentroids, code, weight, outputAccum);
    }

    // process 2 samples
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    // decoded(code1)
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        IndexPQDecoderImpl<DIM, FINE_SIZE, 0>::accum(
                pqFineCentroids0,
                code0,
                weight0,
                pqFineCentroids1,
                code1,
                weight1,
                outputAccum);
    }
};

} // namespace cppcontrib
} // namespace faiss
