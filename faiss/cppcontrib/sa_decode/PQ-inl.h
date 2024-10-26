/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PQ_INL_H
#define PQ_INL_H

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace cppcontrib {

////////////////////////////////////////////////////////////////////////////////////
/// IndexPQDecoder
////////////////////////////////////////////////////////////////////////////////////

// Suitable for PQ[1]x8
template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t FINE_BITS = 8>
struct IndexPQDecoder {
    static_assert(
            FINE_BITS == 8,
            "Only 8 bits is currently supported for FINE_BITS");

    static constexpr intptr_t dim = DIM;
    static constexpr intptr_t fineSize = FINE_SIZE;
    static constexpr intptr_t fineBits = FINE_BITS;

    static constexpr intptr_t FINE_TABLE_BYTES = (1 << FINE_BITS);

    // Process 1 sample.
    // Performs outputStore = decoded(code)
    static void store(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        // fine quantizer
        const uint8_t* const __restrict fine = code;

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t fineCode = fine[fineCentroidIdx];

            const float* const __restrict finePtr = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputStore[i] = *finePtr;
        }
    }

    // Process 1 sample.
    // Performs outputAccum += weight * decoded(code)
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            const float weight,
            float* const __restrict outputAccum) {
        // fine quantizer
        const uint8_t* const __restrict fine = code;

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t fineCode = fine[fineCentroidIdx];

            const float* const __restrict finePtr = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight * (*finePtr);
        }
    }

    // Process 2 samples.
    // Each code uses its own fine pq centroids table.
    //
    // Performs
    //  outputAccum += weight0 * decoded(code0) + weight1 * decoded(code1)
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

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t fineCode0 = fine0[fineCentroidIdx];
            const intptr_t fineCode1 = fine1[fineCentroidIdx];

            const float* const __restrict finePtr0 = pqFineCentroids0 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids1 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*finePtr0) + weight1 * (*finePtr1);
        }
    }

    // Process 2 samples.
    // Fine pq centroids table is shared among codes.
    //
    // Performs
    //  outputAccum += weight0 * decoded(code0) + weight1 * decoded(code1)
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

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t fineCode0 = fine0[fineCentroidIdx];
            const intptr_t fineCode1 = fine1[fineCentroidIdx];

            const float* const __restrict finePtr0 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*finePtr0) + weight1 * (*finePtr1);
        }
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
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t fineCode0 = fine0[fineCentroidIdx];
            const intptr_t fineCode1 = fine1[fineCentroidIdx];
            const intptr_t fineCode2 = fine2[fineCentroidIdx];

            const float* const __restrict finePtr0 = pqFineCentroids0 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids1 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict finePtr2 = pqFineCentroids2 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*finePtr0) + weight1 * (*finePtr1) +
                    weight2 * (*finePtr2);
        }
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
        // fine quantizer
        const uint8_t* const __restrict fine0 = code0;
        const uint8_t* const __restrict fine1 = code1;
        const uint8_t* const __restrict fine2 = code2;

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            const intptr_t fineCode0 = fine0[fineCentroidIdx];
            const intptr_t fineCode1 = fine1[fineCentroidIdx];
            const intptr_t fineCode2 = fine2[fineCentroidIdx];

            const float* const __restrict finePtr0 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict finePtr2 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*finePtr0) + weight1 * (*finePtr1) +
                    weight2 * (*finePtr2);
        }
    }
};

} // namespace cppcontrib
} // namespace faiss
#endif // PQ_INL_H
