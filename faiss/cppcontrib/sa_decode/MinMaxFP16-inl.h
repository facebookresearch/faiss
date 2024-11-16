/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/utils/fp16.h>

namespace faiss {
namespace cppcontrib {

template <typename SubIndexT>
struct IndexMinMaxFP16Decoder {
    static constexpr intptr_t dim = SubIndexT::dim;

    // Process 1 sample.
    // Performs outputStore = scaler * decoded(code) + minv
    static void store(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        const uint16_t* const __restrict codeFP16 =
                reinterpret_cast<const uint16_t*>(code);
        const float scaler = faiss::decode_fp16(codeFP16[0]);
        const float minv = faiss::decode_fp16(codeFP16[1]);

        SubIndexT::store(
                pqCoarseCentroids,
                pqFineCentroids,
                code + 2 * sizeof(uint16_t),
                outputStore);
        for (intptr_t i = 0; i < SubIndexT::dim; i++) {
            outputStore[i] = outputStore[i] * scaler + minv;
        }
    }

    // Process 1 sample.
    // Performs outputStore = scaler * decoded(code) + minv
    static void store(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        const uint16_t* const __restrict codeFP16 =
                reinterpret_cast<const uint16_t*>(code);
        const float scaler = faiss::decode_fp16(codeFP16[0]);
        const float minv = faiss::decode_fp16(codeFP16[1]);

        SubIndexT::store(
                pqFineCentroids, code + 2 * sizeof(uint16_t), outputStore);
        for (intptr_t i = 0; i < SubIndexT::dim; i++) {
            outputStore[i] = outputStore[i] * scaler + minv;
        }
    }

    // Process 1 sample.
    // Performs
    //  * outputAccum += weight * scaler * decoded(code)
    //  * minvAccum += weight * minv
    static void accum(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            const float weight,
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict codeFP16 =
                reinterpret_cast<const uint16_t*>(code);
        const float scaler = faiss::decode_fp16(codeFP16[0]) * weight;
        const float minv = faiss::decode_fp16(codeFP16[1]) * weight;

        SubIndexT::accum(
                pqCoarseCentroids,
                pqFineCentroids,
                code + 2 * sizeof(uint16_t),
                scaler,
                outputAccum);

        minvAccum += minv;
    }

    // Process 1 sample.
    // Performs
    //  * outputAccum += weight * scaler * decoded(code)
    //  * minvAccum += weight * minv
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            const float weight,
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict codeFP16 =
                reinterpret_cast<const uint16_t*>(code);
        const float scaler = faiss::decode_fp16(codeFP16[0]) * weight;
        const float minv = faiss::decode_fp16(codeFP16[1]) * weight;

        SubIndexT::accum(
                pqFineCentroids,
                code + 2 * sizeof(uint16_t),
                scaler,
                outputAccum);

        minvAccum += minv;
    }

    // Process 2 samples.
    // Each code uses its own coarse pq centroids table and fine pq centroids
    // table.
    //
    // Performs
    //  * outputAccum += weight0 * scaler0 * decoded(code0)
    //                 + weight1 * scaler1 * decoded(code1)
    //  * minvAccum += weight0 * minv0 + weight1 * minv1
    static void accum(
            const float* const __restrict pqCoarseCentroids0,
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqCoarseCentroids1,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict code0FP16 =
                reinterpret_cast<const uint16_t*>(code0);
        const float scaler0 = faiss::decode_fp16(code0FP16[0]) * weight0;
        const float minv0 = faiss::decode_fp16(code0FP16[1]) * weight0;

        const uint16_t* const __restrict code1FP16 =
                reinterpret_cast<const uint16_t*>(code1);
        const float scaler1 = faiss::decode_fp16(code1FP16[0]) * weight1;
        const float minv1 = faiss::decode_fp16(code1FP16[1]) * weight1;

        SubIndexT::accum(
                pqCoarseCentroids0,
                pqFineCentroids0,
                code0 + 2 * sizeof(uint16_t),
                scaler0,
                pqCoarseCentroids1,
                pqFineCentroids1,
                code1 + 2 * sizeof(uint16_t),
                scaler1,
                outputAccum);

        minvAccum += minv0 + minv1;
    }

    // Process 2 samples.
    // Coarse pq centroids table and fine pq centroids table are shared among
    // codes.
    //
    // Performs
    //  * outputAccum += weight0 * scaler0 * decoded(code0)
    //                 + weight1 * scaler1 * decoded(code1)
    //  * minvAccum += weight0 * minv0 + weight1 * minv1
    static void accum(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict code0FP16 =
                reinterpret_cast<const uint16_t*>(code0);
        const float scaler0 = faiss::decode_fp16(code0FP16[0]) * weight0;
        const float minv0 = faiss::decode_fp16(code0FP16[1]) * weight0;

        const uint16_t* const __restrict code1FP16 =
                reinterpret_cast<const uint16_t*>(code1);
        const float scaler1 = faiss::decode_fp16(code1FP16[0]) * weight1;
        const float minv1 = faiss::decode_fp16(code1FP16[1]) * weight1;

        SubIndexT::accum(
                pqCoarseCentroids,
                pqFineCentroids,
                code0 + 2 * sizeof(uint16_t),
                scaler0,
                code1 + 2 * sizeof(uint16_t),
                scaler1,
                outputAccum);

        minvAccum += minv0 + minv1;
    }

    // Process 2 samples.
    // Each code uses its own fine pq centroids table.
    //
    // Performs
    //  * outputAccum += weight0 * scaler0 * decoded(code0)
    //                 + weight1 * scaler1 * decoded(code1)
    //  * minvAccum += weight0 * minv0 + weight1 * minv1
    static void accum(
            const float* const __restrict pqFineCentroids0,
            const uint8_t* const __restrict code0,
            const float weight0,
            const float* const __restrict pqFineCentroids1,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict code0FP16 =
                reinterpret_cast<const uint16_t*>(code0);
        const float scaler0 = faiss::decode_fp16(code0FP16[0]) * weight0;
        const float minv0 = faiss::decode_fp16(code0FP16[1]) * weight0;

        const uint16_t* const __restrict code1FP16 =
                reinterpret_cast<const uint16_t*>(code1);
        const float scaler1 = faiss::decode_fp16(code1FP16[0]) * weight1;
        const float minv1 = faiss::decode_fp16(code1FP16[1]) * weight1;

        SubIndexT::accum(
                pqFineCentroids0,
                code0 + 2 * sizeof(uint16_t),
                scaler0,
                pqFineCentroids1,
                code1 + 2 * sizeof(uint16_t),
                scaler1,
                outputAccum);

        minvAccum += minv0 + minv1;
    }

    // Process 2 samples.
    // Fine pq centroids table is shared among codes.
    //
    // Performs
    //  * outputAccum += weight0 * scaler0 * decoded(code0)
    //                 + weight1 * scaler1 * decoded(code1)
    //  * minvAccum += weight0 * minv0 + weight1 * minv1
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict code0FP16 =
                reinterpret_cast<const uint16_t*>(code0);
        const float scaler0 = faiss::decode_fp16(code0FP16[0]) * weight0;
        const float minv0 = faiss::decode_fp16(code0FP16[1]) * weight0;

        const uint16_t* const __restrict code1FP16 =
                reinterpret_cast<const uint16_t*>(code1);
        const float scaler1 = faiss::decode_fp16(code1FP16[0]) * weight1;
        const float minv1 = faiss::decode_fp16(code1FP16[1]) * weight1;

        SubIndexT::accum(
                pqFineCentroids,
                code0 + 2 * sizeof(uint16_t),
                scaler0,
                code1 + 2 * sizeof(uint16_t),
                scaler1,
                outputAccum);

        minvAccum += minv0 + minv1;
    }

    // Process 3 samples.
    // Each code uses its own coarse pq centroids table and fine pq centroids
    // table.
    //
    // Performs
    //  * outputAccum += weight0 * scaler0 * decoded(code0)
    //                 + weight1 * scaler1 * decoded(code1)
    //                 + weight2 * scaler2 * decoded(code2)
    //  * minvAccum += weight0 * minv0 + weight1 * minv1 + weight2 * minv2
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
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict code0FP16 =
                reinterpret_cast<const uint16_t*>(code0);
        const float scaler0 = faiss::decode_fp16(code0FP16[0]) * weight0;
        const float minv0 = faiss::decode_fp16(code0FP16[1]) * weight0;

        const uint16_t* const __restrict code1FP16 =
                reinterpret_cast<const uint16_t*>(code1);
        const float scaler1 = faiss::decode_fp16(code1FP16[0]) * weight1;
        const float minv1 = faiss::decode_fp16(code1FP16[1]) * weight1;

        const uint16_t* const __restrict code2FP16 =
                reinterpret_cast<const uint16_t*>(code2);
        const float scaler2 = faiss::decode_fp16(code2FP16[0]) * weight2;
        const float minv2 = faiss::decode_fp16(code2FP16[1]) * weight2;

        SubIndexT::accum(
                pqCoarseCentroids0,
                pqFineCentroids0,
                code0 + 2 * sizeof(uint16_t),
                scaler0,
                pqCoarseCentroids1,
                pqFineCentroids1,
                code1 + 2 * sizeof(uint16_t),
                scaler1,
                pqCoarseCentroids2,
                pqFineCentroids2,
                code2 + 2 * sizeof(uint16_t),
                scaler2,
                outputAccum);

        minvAccum += minv0 + minv1 + minv2;
    }

    // Process 3 samples.
    // Coarse pq centroids table and fine pq centroids table are shared among
    // codes.
    //
    // Performs
    //  * outputAccum += weight0 * scaler0 * decoded(code0)
    //                 + weight1 * scaler1 * decoded(code1)
    //                 + weight2 * scaler2 * decoded(code2)
    //  * minvAccum += weight0 * minv0 + weight1 * minv1 + weight2 * minv2
    static void accum(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict code0FP16 =
                reinterpret_cast<const uint16_t*>(code0);
        const float scaler0 = faiss::decode_fp16(code0FP16[0]) * weight0;
        const float minv0 = faiss::decode_fp16(code0FP16[1]) * weight0;

        const uint16_t* const __restrict code1FP16 =
                reinterpret_cast<const uint16_t*>(code1);
        const float scaler1 = faiss::decode_fp16(code1FP16[0]) * weight1;
        const float minv1 = faiss::decode_fp16(code1FP16[1]) * weight1;

        const uint16_t* const __restrict code2FP16 =
                reinterpret_cast<const uint16_t*>(code2);
        const float scaler2 = faiss::decode_fp16(code2FP16[0]) * weight2;
        const float minv2 = faiss::decode_fp16(code2FP16[1]) * weight2;

        SubIndexT::accum(
                pqCoarseCentroids,
                pqFineCentroids,
                code0 + 2 * sizeof(uint16_t),
                scaler0,
                code1 + 2 * sizeof(uint16_t),
                scaler1,
                code2 + 2 * sizeof(uint16_t),
                scaler2,
                outputAccum);

        minvAccum += minv0 + minv1 + minv2;
    }

    // Process 3 samples.
    // Each code uses its own fine pq centroids table.
    //
    // Performs
    //  * outputAccum += weight0 * scaler0 * decoded(code0)
    //                 + weight1 * scaler1 * decoded(code1)
    //                 + weight2 * scaler2 * decoded(code2)
    //  * minvAccum += weight0 * minv0 + weight1 * minv1 + weight2 * minv2
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
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict code0FP16 =
                reinterpret_cast<const uint16_t*>(code0);
        const float scaler0 = faiss::decode_fp16(code0FP16[0]) * weight0;
        const float minv0 = faiss::decode_fp16(code0FP16[1]) * weight0;

        const uint16_t* const __restrict code1FP16 =
                reinterpret_cast<const uint16_t*>(code1);
        const float scaler1 = faiss::decode_fp16(code1FP16[0]) * weight1;
        const float minv1 = faiss::decode_fp16(code1FP16[1]) * weight1;

        const uint16_t* const __restrict code2FP16 =
                reinterpret_cast<const uint16_t*>(code2);
        const float scaler2 = faiss::decode_fp16(code2FP16[0]) * weight2;
        const float minv2 = faiss::decode_fp16(code2FP16[1]) * weight2;

        SubIndexT::accum(
                pqFineCentroids0,
                code0 + 2 * sizeof(uint16_t),
                scaler0,
                pqFineCentroids1,
                code1 + 2 * sizeof(uint16_t),
                scaler1,
                pqFineCentroids2,
                code2 + 2 * sizeof(uint16_t),
                scaler2,
                outputAccum);

        minvAccum += minv0 + minv1 + minv2;
    }

    // Process 3 samples.
    // Fine pq centroids table is shared among codes.
    //
    // Performs
    //  * outputAccum += weight0 * scaler0 * decoded(code0)
    //                 + weight1 * scaler1 * decoded(code1)
    //                 + weight2 * scaler2 * decoded(code2)
    //  * minvAccum += weight0 * minv0 + weight1 * minv1 + weight2 * minv2
    static void accum(
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code0,
            const float weight0,
            const uint8_t* const __restrict code1,
            const float weight1,
            const uint8_t* const __restrict code2,
            const float weight2,
            float* const __restrict outputAccum,
            float& minvAccum) {
        const uint16_t* const __restrict code0FP16 =
                reinterpret_cast<const uint16_t*>(code0);
        const float scaler0 = faiss::decode_fp16(code0FP16[0]) * weight0;
        const float minv0 = faiss::decode_fp16(code0FP16[1]) * weight0;

        const uint16_t* const __restrict code1FP16 =
                reinterpret_cast<const uint16_t*>(code1);
        const float scaler1 = faiss::decode_fp16(code1FP16[0]) * weight1;
        const float minv1 = faiss::decode_fp16(code1FP16[1]) * weight1;

        const uint16_t* const __restrict code2FP16 =
                reinterpret_cast<const uint16_t*>(code2);
        const float scaler2 = faiss::decode_fp16(code2FP16[0]) * weight2;
        const float minv2 = faiss::decode_fp16(code2FP16[1]) * weight2;

        SubIndexT::accum(
                pqFineCentroids,
                code0 + 2 * sizeof(uint16_t),
                scaler0,
                code1 + 2 * sizeof(uint16_t),
                scaler1,
                code2 + 2 * sizeof(uint16_t),
                scaler2,
                outputAccum);

        minvAccum += minv0 + minv1 + minv2;
    }
};

} // namespace cppcontrib
} // namespace faiss
