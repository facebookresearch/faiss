/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Reference:
// "Practical and asymptotically optimal quantization of high-dimensional
// vectors in euclidean space for approximate nearest neighbor search"
// Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, Raymond
// Chi-Wing Wong https://dl.acm.org/doi/pdf/10.1145/3725413
//
// Reference implementation: https://github.com/VectorDB-NTU/RaBitQ-Library
// NOTE: Parts of this implementation are adapted from
// rabitqlib/quantization/rabitq_impl.hpp in the above repository.

#pragma once

#include <faiss/MetricType.h>
#include <faiss/impl/RaBitQUtils.h>
#include <cstddef>
#include <cstdint>

namespace faiss {
namespace rabitq_multibit {

/**
 * Compute optimal scaling factor for ex-bits quantization.
 *
 * Uses priority queue-based search to find the scaling factor that
 * maximizes the inner product between quantized and original vectors.
 *
 * @param o_abs Normalized absolute residual vector (positive values)
 * @param d Dimensionality
 * @param nb_bits Number of bits per dimension (2-9)
 * @return Optimal scaling factor 't'
 */
float compute_optimal_scaling_factor(
        const float* o_abs,
        size_t d,
        size_t nb_bits);

/**
 * Pack multi-bit codes from integer array to byte array.
 *
 * @param tmp_code Integer codes (length d), values in [0, 2^ex_bits - 1]
 * @param ex_code Output packed byte array
 * @param d Dimensionality
 * @param nb_bits Number of bits per dimension (2-9)
 */
void pack_multibit_codes(
        const int* tmp_code,
        uint8_t* ex_code,
        size_t d,
        size_t nb_bits);

/**
 * Compute ex-bits factors for distance computation.
 *
 * @param residual Original residual vector (data - centroid)
 * @param centroid Centroid vector (can be nullptr for zero centroid)
 * @param tmp_code Quantized ex-bit codes (unpacked integers)
 * @param d Dimensionality
 * @param ex_bits Number of extra bits
 * @param norm L2 norm of residual
 * @param ipnorm Unnormalized inner product
 * @param ex_factors Output factors structure
 * @param metric_type Distance metric (L2 or IP)
 */
void compute_ex_factors(
        const float* residual,
        const float* centroid,
        const int* tmp_code,
        size_t d,
        size_t ex_bits,
        float norm,
        double ipnorm,
        rabitq_utils::ExFactorsData& ex_factors,
        MetricType metric_type);

/**
 * Main quantization function: quantize residual vector to ex-bits.
 *
 * Performs the complete multi-bit quantization pipeline:
 * 1. Normalize residual
 * 2. Take absolute value
 * 3. Find optimal scaling factor
 * 4. Quantize to ex_bits
 * 5. Handle negative dimensions by bit flipping
 * 6. Pack codes into byte array
 * 7. Compute factors for distance computation
 *
 * @param residual Input residual vector (data - centroid), length d
 * @param d Dimensionality
 * @param nb_bits Number of bits per dimension (2-9)
 * @param ex_code Output packed ex-bit codes
 * @param ex_factors Output ex-bits factors
 * @param metric_type Distance metric (L2 or Inner Product)
 * @param centroid Optional centroid vector (needed for IP metric)
 */
void quantize_ex_bits(
        const float* residual,
        size_t d,
        size_t nb_bits,
        uint8_t* ex_code,
        rabitq_utils::ExFactorsData& ex_factors,
        MetricType metric_type,
        const float* centroid = nullptr);

} // namespace rabitq_multibit
} // namespace faiss
