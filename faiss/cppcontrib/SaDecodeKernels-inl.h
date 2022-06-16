// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace cppcontrib {

// Suitable for IVF256,PQ[1]x8
// Suitable for Residual[1]x8,PQ[2]x8
template <intptr_t DIM, intptr_t COARSE_SIZE, intptr_t FINE_SIZE>
struct Index2LevelDecoder {
    // Process 1 sample.
    // Performs outputStore = decoded(code)
    static void store(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        // coarse quantizer
        const uint8_t* const __restrict coarse = code;

        // fine quantizer
        const uint8_t* const __restrict fine = code + (DIM / COARSE_SIZE);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t coarseCode = coarse[coarseCentroidIdx];
            const intptr_t fineCode = fine[fineCentroidIdx];

            const float* const __restrict coarsePtr = pqCoarseCentroids +
                    (coarseCentroidIdx * 256 + coarseCode) * COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr = pqFineCentroids +
                    (fineCentroidIdx * 256 + fineCode) * FINE_SIZE +
                    fineCentroidOffset;

            outputStore[i] = *coarsePtr + *finePtr;
        }
    }

    // Process 1 sample.
    // Performs outputAccum += weight * decoded(code)
    static void accum(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            const float weight,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const uint8_t* const __restrict coarse = code;

        // fine quantizer
        const uint8_t* const __restrict fine = code + (DIM / COARSE_SIZE);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t coarseCode = coarse[coarseCentroidIdx];
            const intptr_t fineCode = fine[fineCentroidIdx];

            const float* const __restrict coarsePtr = pqCoarseCentroids +
                    (coarseCentroidIdx * 256 + coarseCode) * COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr = pqFineCentroids +
                    (fineCentroidIdx * 256 + fineCode) * FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight * (*coarsePtr + *finePtr);
        }
    }

    // process 2 samples
    // Performs
    //  outputAccum += weight0 * decoded(code0) + weight1 * decoded(code1)
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
        const uint8_t* const __restrict fine0 = code0 + (DIM / COARSE_SIZE);
        const uint8_t* const __restrict fine1 = code1 + (DIM / COARSE_SIZE);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t coarseCode0 = coarse0[coarseCentroidIdx];
            const intptr_t fineCode0 = fine0[fineCentroidIdx];
            const intptr_t coarseCode1 = coarse1[coarseCentroidIdx];
            const intptr_t fineCode1 = fine1[fineCentroidIdx];

            const float* const __restrict coarsePtr0 = pqCoarseCentroids0 +
                    (coarseCentroidIdx * 256 + coarseCode0) * COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr0 = pqFineCentroids0 +
                    (fineCentroidIdx * 256 + fineCode0) * FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict coarsePtr1 = pqCoarseCentroids1 +
                    (coarseCentroidIdx * 256 + coarseCode1) * COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids1 +
                    (fineCentroidIdx * 256 + fineCode1) * FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*coarsePtr0 + *finePtr0) +
                    weight1 * (*coarsePtr1 + *finePtr1);
        }
    }
};

} // namespace cppcontrib
} // namespace faiss
