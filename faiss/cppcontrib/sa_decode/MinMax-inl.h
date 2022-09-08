#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace cppcontrib {

template <typename SubIndexT>
struct IndexMinMaxDecoder {
    static constexpr intptr_t dim = SubIndexT::dim;

    // Process 1 sample.
    // Performs outputStore = scaler * decoded(code) + minv
    static void store(
            const float* const __restrict pqCoarseCentroids,
            const float* const __restrict pqFineCentroids,
            const uint8_t* const __restrict code,
            float* const __restrict outputStore) {
        const float* const __restrict codeFloat =
                reinterpret_cast<const float*>(code);
        const float scaler = codeFloat[0];
        const float minv = codeFloat[1];

        SubIndexT::store(
                pqCoarseCentroids,
                pqFineCentroids,
                code + 2 * sizeof(float),
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
        const float* const __restrict codeFloat =
                reinterpret_cast<const float*>(code);
        const float scaler = codeFloat[0];
        const float minv = codeFloat[1];

        SubIndexT::store(
                pqFineCentroids, code + 2 * sizeof(float), outputStore);
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
        const float* const __restrict codeFloat =
                reinterpret_cast<const float*>(code);
        const float scaler = codeFloat[0] * weight;
        const float minv = codeFloat[1] * weight;

        SubIndexT::accum(
                pqCoarseCentroids,
                pqFineCentroids,
                code + 2 * sizeof(float),
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
        const float* const __restrict codeFloat =
                reinterpret_cast<const float*>(code);
        const float scaler = codeFloat[0] * weight;
        const float minv = codeFloat[1] * weight;

        SubIndexT::accum(
                pqFineCentroids, code + 2 * sizeof(float), scaler, outputAccum);

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
        const float* const __restrict code0Float =
                reinterpret_cast<const float*>(code0);
        const float scaler0 = code0Float[0] * weight0;
        const float minv0 = code0Float[1] * weight0;

        const float* const __restrict code1Float =
                reinterpret_cast<const float*>(code1);
        const float scaler1 = code1Float[0] * weight1;
        const float minv1 = code1Float[1] * weight1;

        SubIndexT::accum(
                pqCoarseCentroids0,
                pqFineCentroids0,
                code0 + 2 * sizeof(float),
                scaler0,
                pqCoarseCentroids1,
                pqFineCentroids1,
                code1 + 2 * sizeof(float),
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
        const float* const __restrict code0Float =
                reinterpret_cast<const float*>(code0);
        const float scaler0 = code0Float[0] * weight0;
        const float minv0 = code0Float[1] * weight0;

        const float* const __restrict code1Float =
                reinterpret_cast<const float*>(code1);
        const float scaler1 = code1Float[0] * weight1;
        const float minv1 = code1Float[1] * weight1;

        SubIndexT::accum(
                pqCoarseCentroids,
                pqFineCentroids,
                code0 + 2 * sizeof(float),
                scaler0,
                code1 + 2 * sizeof(float),
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
        const float* const __restrict code0Float =
                reinterpret_cast<const float*>(code0);
        const float scaler0 = code0Float[0] * weight0;
        const float minv0 = code0Float[1] * weight0;

        const float* const __restrict code1Float =
                reinterpret_cast<const float*>(code1);
        const float scaler1 = code1Float[0] * weight1;
        const float minv1 = code1Float[1] * weight1;

        SubIndexT::accum(
                pqFineCentroids0,
                code0 + 2 * sizeof(float),
                scaler0,
                pqFineCentroids1,
                code1 + 2 * sizeof(float),
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
        const float* const __restrict code0Float =
                reinterpret_cast<const float*>(code0);
        const float scaler0 = code0Float[0] * weight0;
        const float minv0 = code0Float[1] * weight0;

        const float* const __restrict code1Float =
                reinterpret_cast<const float*>(code1);
        const float scaler1 = code1Float[0] * weight1;
        const float minv1 = code1Float[1] * weight1;

        SubIndexT::accum(
                pqFineCentroids,
                code0 + 2 * sizeof(float),
                scaler0,
                code1 + 2 * sizeof(float),
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
        const float* const __restrict code0Float =
                reinterpret_cast<const float*>(code0);
        const float scaler0 = code0Float[0] * weight0;
        const float minv0 = code0Float[1] * weight0;

        const float* const __restrict code1Float =
                reinterpret_cast<const float*>(code1);
        const float scaler1 = code1Float[0] * weight1;
        const float minv1 = code1Float[1] * weight1;

        const float* const __restrict code2Float =
                reinterpret_cast<const float*>(code2);
        const float scaler2 = code2Float[0] * weight2;
        const float minv2 = code2Float[1] * weight2;

        SubIndexT::accum(
                pqCoarseCentroids0,
                pqFineCentroids0,
                code0 + 2 * sizeof(float),
                scaler0,
                pqCoarseCentroids1,
                pqFineCentroids1,
                code1 + 2 * sizeof(float),
                scaler1,
                pqCoarseCentroids2,
                pqFineCentroids2,
                code2 + 2 * sizeof(float),
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
        const float* const __restrict code0Float =
                reinterpret_cast<const float*>(code0);
        const float scaler0 = code0Float[0] * weight0;
        const float minv0 = code0Float[1] * weight0;

        const float* const __restrict code1Float =
                reinterpret_cast<const float*>(code1);
        const float scaler1 = code1Float[0] * weight1;
        const float minv1 = code1Float[1] * weight1;

        const float* const __restrict code2Float =
                reinterpret_cast<const float*>(code2);
        const float scaler2 = code2Float[0] * weight2;
        const float minv2 = code2Float[1] * weight2;

        SubIndexT::accum(
                pqCoarseCentroids,
                pqFineCentroids,
                code0 + 2 * sizeof(float),
                scaler0,
                code1 + 2 * sizeof(float),
                scaler1,
                code2 + 2 * sizeof(float),
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
        const float* const __restrict code0Float =
                reinterpret_cast<const float*>(code0);
        const float scaler0 = code0Float[0] * weight0;
        const float minv0 = code0Float[1] * weight0;

        const float* const __restrict code1Float =
                reinterpret_cast<const float*>(code1);
        const float scaler1 = code1Float[0] * weight1;
        const float minv1 = code1Float[1] * weight1;

        const float* const __restrict code2Float =
                reinterpret_cast<const float*>(code2);
        const float scaler2 = code2Float[0] * weight2;
        const float minv2 = code2Float[1] * weight2;

        SubIndexT::accum(
                pqFineCentroids0,
                code0 + 2 * sizeof(float),
                scaler0,
                pqFineCentroids1,
                code1 + 2 * sizeof(float),
                scaler1,
                pqFineCentroids2,
                code2 + 2 * sizeof(float),
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
        const float* const __restrict code0Float =
                reinterpret_cast<const float*>(code0);
        const float scaler0 = code0Float[0] * weight0;
        const float minv0 = code0Float[1] * weight0;

        const float* const __restrict code1Float =
                reinterpret_cast<const float*>(code1);
        const float scaler1 = code1Float[0] * weight1;
        const float minv1 = code1Float[1] * weight1;

        const float* const __restrict code2Float =
                reinterpret_cast<const float*>(code2);
        const float scaler2 = code2Float[0] * weight2;
        const float minv2 = code2Float[1] * weight2;

        SubIndexT::accum(
                pqFineCentroids,
                code0 + 2 * sizeof(float),
                scaler0,
                code1 + 2 * sizeof(float),
                scaler1,
                code2 + 2 * sizeof(float),
                scaler2,
                outputAccum);

        minvAccum += minv0 + minv1 + minv2;
    }
};

} // namespace cppcontrib
} // namespace faiss
