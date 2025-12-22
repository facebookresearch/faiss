/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

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

    // RaBitQ codes computations are independent from a metric. But it is needed
    //   to store some additional fp32 constants together with a quantized code.
    //   A decision was made to make this quantizer as space efficient as
    //   possible. Thus, a quantizer has to introduce a metric.
    MetricType metric_type = MetricType::METRIC_L2;

    // Number of bits per dimension (1-9). Default is 1 for backward
    // compatibility.
    // - nb_bits = 1: standard 1-bit RaBitQ (sign bits only)
    // - nb_bits = 2-9: multi-bit RaBitQ (1 sign bit + ex_bits extra bits)
    size_t nb_bits = 1;

    RaBitQuantizer(
            size_t d = 0,
            MetricType metric = MetricType::METRIC_L2,
            size_t nb_bits = 1);

    // Compute code size based on dimensionality and number of bits
    // Returns: size in bytes for one encoded vector
    // - nb_bits=1: (d+7)/8 + 8 bytes (1-bit codes + base factors)
    // - nb_bits>1: (d+7)/8 + 8 + d*ex_bits/8 + 8 bytes
    //              (1-bit codes + base factors + ex-bit codes + ex factors)
    size_t compute_code_size(size_t d, size_t num_bits) const;

    void train(size_t n, const float* x) override;

    // every vector is expected to take (d + 7) / 8 + sizeof(SignBitFactors)
    // bytes,
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

    // returns the distance computer.
    // specify qb = 0 to get an DC that does not quantize a query
    // specify qb > 0 to have SQ qb-bits query
    FlatCodesDistanceComputer* get_distance_computer(
            uint8_t qb = 0,
            const float* centroid = nullptr,
            bool centered = false) const;
};

// RaBitQDistanceComputer: Base class for RaBitQ distance computers
//
// This intermediate class exists to provide a unified interface for
// two-stage multi-bit search. While most Faiss quantizers extend
// FlatCodesDistanceComputer directly, RaBitQ requires this additional
// abstraction layer due to its unique split encoding strategy
// (1 sign bit + magnitude bits) which enables:
//
// 1. distance_to_code_1bit() - Fast 1-bit filtering using only sign bits
// 2. distance_to_code_full() - Accurate multi-bit refinement using all bits
// 3. lower_bound_distance() - Error-bounded adaptive filtering
//                              (based on 1-bit estimator)
//
// These three methods implement RaBitQ's two-stage search pattern and are
// shared between the quantized (Q) and non-quantized (NotQ) query variants.
// The intermediate class allows two-stage search code to work with both
// variants via a single dynamic_cast.
struct RaBitQDistanceComputer : FlatCodesDistanceComputer {
    size_t d = 0;
    const float* centroid = nullptr;
    MetricType metric_type = MetricType::METRIC_L2;
    size_t nb_bits = 1;

    // Query norm for lower bound computation (g_error in rabitq-library)
    // This is the L2 norm of the rotated query: ||query - centroid||
    float g_error = 0.0f;

    float symmetric_dis(idx_t /*i*/, idx_t /*j*/) override {
        // Not used for RaBitQ
        FAISS_THROW_MSG("Not implemented");
    }

    // Compute 1-bit distance estimate (fast)
    virtual float distance_to_code_1bit(const uint8_t* code) = 0;

    // Compute full multi-bit distance (accurate)
    virtual float distance_to_code_full(const uint8_t* code) = 0;

    // Compute lower bound of distance using error bounds
    // Guarantees: actual_distance >= lower_bound_distance
    // Used for adaptive filtering in two-stage search
    virtual float lower_bound_distance(const uint8_t* code);

    // Override from FlatCodesDistanceComputer
    // Delegates to distance_to_code_full() for multi-bit distance computation
    float distance_to_code(const uint8_t* code) final {
        return distance_to_code_full(code);
    }
};

} // namespace faiss
