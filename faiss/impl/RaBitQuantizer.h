#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/Quantizer.h>

namespace faiss {

// the reference implementation of the https://arxiv.org/pdf/2405.12497
//   Jianyang Gao, Cheng Long, "RaBitQ: Quantizing High-Dimensional Vectors
//   with a Theoretical Error Bound for Approximate Nearest Neighbor Search".
//
// It is assumed that the Random Matrix Rotation is performed externally.
struct RaBitQuantizer : Quantizer {
    // all RaBitQ operations are provided against a centroid, which needs
    //   to be provided Externally (!). Nullptr value implies that the centroid
    //   consists of zero values.
    // This is the default value that can be customized using XYZ_core() calls.
    //   Such a customization is needed for IVF calls.
    //
    // This particular pointer will NOT be serialized.
    float* centroid = nullptr;

    RaBitQuantizer(size_t d = 0);

    void train(size_t n, const float* x) override;

    // every vector is expected to take (d + 7) / 8 + sizeof(FactorsData) bytes,
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

    void compute_codes_core(
            const float* x,
            uint8_t* codes,
            size_t n,
            const float* centroid_in) const;

    // The decode output is Heavily geared towards maintaining the IP, not L2.
    // This means that the reconstructed codes maybe less accurate than one may
    //   expect, if one computes an L2 distance between a reconstructed code and
    //   the corresponding original vector.
    // But value of the dot product between a query and the original vector
    //   might be very close to the value of the dot product between a query and
    //   the reconstructed code.
    // Basically, it seems to be related to the distributions of values, not
    //   values.
    void decode(const uint8_t* codes, float* x, size_t n) const override;

    void decode_core(
            const uint8_t* codes,
            float* x,
            size_t n,
            const float* centroid_in) const;

    struct RaBitDistanceComputer : FlatCodesDistanceComputer {
        // dimensionality
        size_t d = 0;
        // a centroid to use
        const float* centroid = nullptr;

        // the metric
        MetricType metric = MetricType::METRIC_L2;

        RaBitDistanceComputer();

        float distance_to_code(const uint8_t* code) override;

        virtual float distance_to_code_IP(const uint8_t* code) const = 0;
        virtual float distance_to_code_L2(const uint8_t* code) const = 0;

        float symmetric_dis(idx_t i, idx_t j) override;
    };

    // returns the distance computer.
    // specify qb = 0 to get an DC that does not quantize a query
    // specify qb > 0 to have SQ qb-bits query
    RaBitDistanceComputer* get_distance_computer(
            uint8_t qb,
            MetricType metric,
            const float* centroid_in = nullptr) const;
};

} // namespace faiss