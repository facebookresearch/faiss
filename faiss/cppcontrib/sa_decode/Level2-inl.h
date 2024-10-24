/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef LEVEL2_INL_H
#define LEVEL2_INL_H

#include <cstddef>
#include <cstdint>

#include <faiss/cppcontrib/detail/CoarseBitType.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {
namespace cppcontrib {

bool isBigEndian() {
#ifdef FAISS_BIG_ENDIAN
    return true;
#else
    return false;
#endif
}

////////////////////////////////////////////////////////////////////////////////////
/// Index2LevelDecoder
////////////////////////////////////////////////////////////////////////////////////

// Suitable for IVF256,PQ[1]x8
// Suitable for Residual[1]x8,PQ[2]x8
// Suitable for IVF[9-16 bit],PQ[1]x8 (such as IVF1024,PQ16np)
// Suitable for Residual1x[9-16 bit],PQ[1]x8 (such as Residual1x9,PQ8)
template <
        intptr_t DIM,
        intptr_t COARSE_SIZE,
        intptr_t FINE_SIZE,
        intptr_t COARSE_BITS = 8,
        intptr_t FINE_BITS = 8>
struct Index2LevelDecoder {
    static_assert(
            COARSE_BITS == 8 || COARSE_BITS == 16,
            "Only 8 or 16 bits are currently supported for COARSE_BITS");
    static_assert(
            FINE_BITS == 8,
            "Only 8 bits is currently supported for FINE_BITS");

    static constexpr intptr_t dim = DIM;
    static constexpr intptr_t coarseSize = COARSE_SIZE;
    static constexpr intptr_t fineSize = FINE_SIZE;
    static constexpr intptr_t coarseBits = COARSE_BITS;
    static constexpr intptr_t fineBits = FINE_BITS;

    // coarse quantizer storage
    using coarse_storage_type =
            typename detail::CoarseBitType<COARSE_BITS>::bit_type;
    static constexpr intptr_t COARSE_TABLE_BYTES = (1 << COARSE_BITS);

    static constexpr intptr_t FINE_TABLE_BYTES = (1 << FINE_BITS);

    // Process 1 sample.
    // Performs outputStore = decoded(code)
    static void store(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        // coarse quantizer
        const coarse_storage_type* const __restrict coarse =
                reinterpret_cast<const coarse_storage_type*>(code);

        // fine quantizer
        const uint8_t* const __restrict fine =
                code + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;
            intptr_t coarseCode, fineCode;
            if (isBigEndian() && sizeof(coarse_storage_type) == 2) {
                coarseCode = Swap2Bytes(coarse[coarseCentroidIdx]);
                fineCode = Swap2Bytes(fine[fineCentroidIdx]);
            } else {
                coarseCode = coarse[coarseCentroidIdx];
                fineCode = fine[fineCentroidIdx];
            }

            const float* const __restrict coarsePtr = pqCoarseCentroids +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode) *
                            FINE_SIZE +
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
        const coarse_storage_type* const __restrict coarse =
                reinterpret_cast<const coarse_storage_type*>(code);

        // fine quantizer
        const uint8_t* const __restrict fine =
                code + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;

            intptr_t coarseCode, fineCode;
            if (isBigEndian() && sizeof(coarse_storage_type) == 2) {
                coarseCode = Swap2Bytes(coarse[coarseCentroidIdx]);
                fineCode = Swap2Bytes(fine[fineCentroidIdx]);
            } else {
                coarseCode = coarse[coarseCentroidIdx];
                fineCode = fine[fineCentroidIdx];
            }
            const float* const __restrict coarsePtr = pqCoarseCentroids +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight * (*coarsePtr + *finePtr);
        }
    }

    // Process 2 samples.
    // Each code uses its own coarse pq centroids table and fine pq centroids
    // table.
    //
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    //   decoded(code1).
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
        const coarse_storage_type* const __restrict coarse0 =
                reinterpret_cast<const coarse_storage_type*>(code0);
        const coarse_storage_type* const __restrict coarse1 =
                reinterpret_cast<const coarse_storage_type*>(code1);

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine1 =
                code1 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;
            intptr_t coarseCode0, coarseCode1, fineCode0, fineCode1;
            if (isBigEndian() && sizeof(coarse_storage_type) == 2) {
                coarseCode0 = Swap2Bytes(coarse0[coarseCentroidIdx]);
                fineCode0 = Swap2Bytes(fine0[fineCentroidIdx]);
                coarseCode1 = Swap2Bytes(coarse1[coarseCentroidIdx]);
                fineCode1 = Swap2Bytes(fine1[fineCentroidIdx]);
            } else {
                coarseCode0 = coarse0[coarseCentroidIdx];
                fineCode0 = fine0[fineCentroidIdx];
                coarseCode1 = coarse1[coarseCentroidIdx];
                fineCode1 = fine1[fineCentroidIdx];
            }

            const float* const __restrict coarsePtr0 = pqCoarseCentroids0 +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr0 = pqFineCentroids0 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict coarsePtr1 = pqCoarseCentroids1 +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode1) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids1 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*coarsePtr0 + *finePtr0) +
                    weight1 * (*coarsePtr1 + *finePtr1);
        }
    }

    // Process 2 samples.
    // Coarse pq centroids table and fine pq centroids table are shared among
    // codes.
    //
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    //   decoded(code1)
    static void accum(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const coarse_storage_type* const __restrict coarse0 =
                reinterpret_cast<const coarse_storage_type*>(code0);
        const coarse_storage_type* const __restrict coarse1 =
                reinterpret_cast<const coarse_storage_type*>(code1);

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine1 =
                code1 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;
            intptr_t coarseCode0, coarseCode1, fineCode0, fineCode1;
            if (isBigEndian() && sizeof(coarse_storage_type) == 2) {
                coarseCode0 = Swap2Bytes(coarse0[coarseCentroidIdx]);
                fineCode0 = Swap2Bytes(fine0[fineCentroidIdx]);
                coarseCode1 = Swap2Bytes(coarse1[coarseCentroidIdx]);
                fineCode1 = Swap2Bytes(fine1[fineCentroidIdx]);
            } else {
                coarseCode0 = coarse0[coarseCentroidIdx];
                fineCode0 = fine0[fineCentroidIdx];
                coarseCode1 = coarse1[coarseCentroidIdx];
                fineCode1 = fine1[fineCentroidIdx];
            }

            const float* const __restrict coarsePtr0 = pqCoarseCentroids +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr0 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict coarsePtr1 = pqCoarseCentroids +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode1) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*coarsePtr0 + *finePtr0) +
                    weight1 * (*coarsePtr1 + *finePtr1);
        }
    }

    // Process 3 samples.
    // Each code uses its own coarse pq centroids table and fine pq centroids
    // table.
    //
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    //   decoded(code1) + weight2 * decoded(code2)
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqCoarseCentroids1,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            const float* const __restrict pqCoarseCentroids2,
            const float* const __restrict pqFineCentroids2,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const coarse_storage_type* const __restrict coarse0 =
                reinterpret_cast<const coarse_storage_type*>(code0);
        const coarse_storage_type* const __restrict coarse1 =
                reinterpret_cast<const coarse_storage_type*>(code1);
        const coarse_storage_type* const __restrict coarse2 =
                reinterpret_cast<const coarse_storage_type*>(code2);

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine1 =
                code1 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine2 =
                code2 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;
            intptr_t coarseCode0, coarseCode1, fineCode0, fineCode1;
            intptr_t coarseCode2, fineCode2;
            if (isBigEndian() && sizeof(coarse_storage_type) == 2) {
                coarseCode0 = Swap2Bytes(coarse0[coarseCentroidIdx]);
                fineCode0 = Swap2Bytes(fine0[fineCentroidIdx]);
                coarseCode1 = Swap2Bytes(coarse1[coarseCentroidIdx]);
                fineCode1 = Swap2Bytes(fine1[fineCentroidIdx]);
                coarseCode2 = Swap2Bytes(coarse2[coarseCentroidIdx]);
                fineCode2 = Swap2Bytes(fine2[fineCentroidIdx]);
            } else {
                coarseCode0 = coarse0[coarseCentroidIdx];
                fineCode0 = fine0[fineCentroidIdx];
                coarseCode1 = coarse1[coarseCentroidIdx];
                fineCode1 = fine1[fineCentroidIdx];
                coarseCode2 = coarse2[coarseCentroidIdx];
                fineCode2 = fine2[fineCentroidIdx];
            }

            const float* const __restrict coarsePtr0 = pqCoarseCentroids0 +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr0 = pqFineCentroids0 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict coarsePtr1 = pqCoarseCentroids1 +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode1) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids1 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict coarsePtr2 = pqCoarseCentroids2 +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode2) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr2 = pqFineCentroids2 +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*coarsePtr0 + *finePtr0) +
                    weight1 * (*coarsePtr1 + *finePtr1) +
                    weight2 * (*coarsePtr2 + *finePtr2);
        }
    }

    // Process 3 samples.
    // Coarse pq centroids table and fine pq centroids table are shared among
    // codes.
    //
    // Performs outputAccum += weight0 * decoded(code0) + weight1 *
    //   decoded(code1) + weight2 * decoded(code2)
    static void accum(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum) {
        // coarse quantizer
        const coarse_storage_type* const __restrict coarse0 =
                reinterpret_cast<const coarse_storage_type*>(code0);
        const coarse_storage_type* const __restrict coarse1 =
                reinterpret_cast<const coarse_storage_type*>(code1);
        const coarse_storage_type* const __restrict coarse2 =
                reinterpret_cast<const coarse_storage_type*>(code2);

        // fine quantizer
        const uint8_t* const __restrict fine0 =
                code0 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine1 =
                code1 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);
        const uint8_t* const __restrict fine2 =
                code2 + (DIM / COARSE_SIZE) * sizeof(coarse_storage_type);

#pragma unroll
        for (intptr_t i = 0; i < DIM; i++) {
            const intptr_t coarseCentroidIdx = i / COARSE_SIZE;
            const intptr_t coarseCentroidOffset = i % COARSE_SIZE;
            const intptr_t fineCentroidIdx = i / FINE_SIZE;
            const intptr_t fineCentroidOffset = i % FINE_SIZE;
            intptr_t coarseCode0, fineCode0, coarseCode1, fineCode1;
            intptr_t coarseCode2, fineCode2;
            if (isBigEndian() && sizeof(coarse_storage_type) == 2) {
                coarseCode0 = Swap2Bytes(coarse0[coarseCentroidIdx]);
                fineCode0 = Swap2Bytes(fine0[fineCentroidIdx]);
                coarseCode1 = Swap2Bytes(coarse1[coarseCentroidIdx]);
                fineCode1 = Swap2Bytes(fine1[fineCentroidIdx]);
                coarseCode2 = Swap2Bytes(coarse2[coarseCentroidIdx]);
                fineCode2 = Swap2Bytes(fine2[fineCentroidIdx]);
            } else {
                coarseCode0 = coarse0[coarseCentroidIdx];
                fineCode0 = fine0[fineCentroidIdx];
                coarseCode1 = coarse1[coarseCentroidIdx];
                fineCode1 = fine1[fineCentroidIdx];
                coarseCode2 = coarse2[coarseCentroidIdx];
                fineCode2 = fine2[fineCentroidIdx];
            }

            const float* const __restrict coarsePtr0 = pqCoarseCentroids +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode0) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr0 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode0) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict coarsePtr1 = pqCoarseCentroids +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode1) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr1 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode1) *
                            FINE_SIZE +
                    fineCentroidOffset;
            const float* const __restrict coarsePtr2 = pqCoarseCentroids +
                    (coarseCentroidIdx * COARSE_TABLE_BYTES + coarseCode2) *
                            COARSE_SIZE +
                    coarseCentroidOffset;
            const float* const __restrict finePtr2 = pqFineCentroids +
                    (fineCentroidIdx * FINE_TABLE_BYTES + fineCode2) *
                            FINE_SIZE +
                    fineCentroidOffset;

            outputAccum[i] += weight0 * (*coarsePtr0 + *finePtr0) +
                    weight1 * (*coarsePtr1 + *finePtr1) +
                    weight2 * (*coarsePtr2 + *finePtr2);
        }
    }
};

} // namespace cppcontrib
} // namespace faiss
#endif // LEVEL2_INL_H
