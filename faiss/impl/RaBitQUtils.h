/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/impl/platform_macros.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace faiss {
namespace rabitq_utils {

/** Base factors computed per database vector for RaBitQ distance computation.
 * Used by both 1-bit and multi-bit RaBitQ variants.
 * These can be stored either embedded in codes (IndexRaBitQ) or separately
 * (IndexRaBitQFastScan).
 *
 * For 1-bit mode only - contains the minimal factors needed for distance
 * estimation using just sign bits.
 */
FAISS_PACK_STRUCTS_BEGIN
struct FAISS_PACKED SignBitFactors {
    // ||or - c||^2 - ((metric==IP) ? ||or||^2 : 0)
    float or_minus_c_l2sqr = 0;
    float dp_multiplier = 0;
};

/** Extended factors for multi-bit RaBitQ (nb_bits > 1).
 * Includes error bound for lower bound computation in two-stage search.
 * Inherits base factors to maintain layout compatibility.
 *
 * Used in multi-bit mode - the error bound enables quick filtering of
 * unlikely candidates in the first stage of two-stage search.
 */
struct FAISS_PACKED SignBitFactorsWithError : SignBitFactors {
    // Error bound for lower bound computation in two-stage search
    // Used in formula: lower_bound = est_distance - f_error * g_error
    // Only allocated when nb_bits > 1
    float f_error = 0;
};

/** Additional factors for multi-bit RaBitQ (nb_bits > 1).
 * Used to store normalization and scaling factors for the refinement bits
 * that encode additional precision beyond the sign bit.
 */
struct FAISS_PACKED ExtraBitsFactors {
    // Additive correction factor for refinement bit reconstruction
    float f_add_ex = 0;
    // Scaling/rescaling factor for refinement bit reconstruction
    float f_rescale_ex = 0;
};
FAISS_PACK_STRUCTS_END

/** Query-specific factors computed during search for RaBitQ distance
 * computation. Used by both IndexRaBitQ and IndexRaBitQFastScan
 * implementations.
 */
struct QueryFactorsData {
    float c1 = 0;
    float c2 = 0;
    float c34 = 0;

    float qr_to_c_L2sqr = 0;
    float qr_norm_L2sqr = 0;

    float int_dot_scale = 1;

    float g_error = 0;
    std::vector<float> rotated_q;
};

/** Ideal quantizer radii for quantizers of 1..8 bits, optimized to minimize
 * L2 reconstruction error. Shared between all RaBitQ implementations.
 */
FAISS_API extern const float Z_MAX_BY_QB[8];

/** Compute factors for a single database vector using RaBitQ algorithm.
 * This function consolidates the mathematical logic that was duplicated
 * between IndexRaBitQ and IndexRaBitQFastScan.
 *
 * @param x             input vector (d dimensions)
 * @param d             dimensionality
 * @param centroid      database centroid (nullptr if not used)
 * @param metric_type   distance metric (L2 or Inner Product)
 * @param compute_error whether to compute f_error (false for 1-bit mode)
 * @return              computed factors for distance computation
 */
SignBitFactorsWithError compute_vector_factors(
        const float* x,
        size_t d,
        const float* centroid,
        MetricType metric_type,
        bool compute_error = true);

/** Compute intermediate values needed for vector factor computation.
 * Separated out to allow different bit packing strategies while sharing
 * the core mathematical computation.
 *
 * @param x             input vector (d dimensions)
 * @param d             dimensionality
 * @param centroid      database centroid (nullptr if not used)
 * @param norm_L2sqr    output: ||or - c||^2
 * @param or_L2sqr      output: ||or||^2
 * @param dp_oO         output: sum of |or_i - c_i| (absolute deviations)
 */
void compute_vector_intermediate_values(
        const float* x,
        size_t d,
        const float* centroid,
        float& norm_L2sqr,
        float& or_L2sqr,
        float& dp_oO);

/** Compute final factors from intermediate values.
 * @param norm_L2sqr    ||or - c||^2
 * @param or_L2sqr      ||or||^2
 * @param dp_oO         sum of |or_i - c_i|
 * @param d             dimensionality
 * @param metric_type   distance metric
 * @param compute_error whether to compute f_error (false for 1-bit mode)
 * @return              computed factors
 */
SignBitFactorsWithError compute_factors_from_intermediates(
        float norm_L2sqr,
        float or_L2sqr,
        float dp_oO,
        size_t d,
        MetricType metric_type,
        bool compute_error = true);

/** Compute query factors for RaBitQ distance computation.
 * This consolidates the query processing logic shared between implementations.
 *
 * @param query         query vector (d dimensions)
 * @param d             dimensionality
 * @param centroid      database centroid (nullptr if not used)
 * @param qb            number of quantization bits (1-8)
 * @param centered      whether to use centered quantization
 * @param metric_type   distance metric
 * @param rotated_q     output: query - centroid
 * @param rotated_qq    output: quantized query values
 * @return              computed query factors
 */
QueryFactorsData compute_query_factors(
        const float* query,
        size_t d,
        const float* centroid,
        uint8_t qb,
        bool centered,
        MetricType metric_type,
        std::vector<float>& rotated_q,
        std::vector<uint8_t>& rotated_qq);

/** Extract bit value from RaBitQ code in standard format.
 * Used by IndexRaBitQ which stores bits sequentially.
 *
 * @param code          RaBitQ code data
 * @param bit_index     which bit to extract (0 to d-1)
 * @return              bit value (true/false)
 */
bool extract_bit_standard(const uint8_t* code, size_t bit_index);

/** Extract bit value from FastScan code format.
 * Used by IndexRaBitQFastScan which packs bits into 4-bit sub-quantizers.
 *
 * @param code          FastScan code data
 * @param bit_index     which bit to extract (0 to d-1)
 * @return              bit value (true/false)
 */
bool extract_bit_fastscan(const uint8_t* code, size_t bit_index);

/** Set bit value in standard RaBitQ code format.
 * @param code          RaBitQ code data to modify
 * @param bit_index     which bit to set (0 to d-1)
 */
void set_bit_standard(uint8_t* code, size_t bit_index);

/** Set bit value in FastScan code format.
 * @param code          FastScan code data to modify
 * @param bit_index     which bit to set (0 to d-1)
 */
void set_bit_fastscan(uint8_t* code, size_t bit_index);

/** Compute adjusted 1-bit distance from normalized LUT distance.
 * This is the core distance formula shared by all RaBitQ handlers.
 *
 * @param normalized_distance  Distance from SIMD LUT lookup (after
 * normalization)
 * @param db_factors          Database vector factors (SignBitFactors or
 * SignBitFactorsWithError)
 * @param query_factors       Query factors computed during search
 * @param centered            Whether centered quantization is used
 * @param qb                  Number of quantization bits
 * @param d                   Dimensionality
 * @return                    Adjusted distance value
 */
inline float compute_1bit_adjusted_distance(
        float normalized_distance,
        const SignBitFactors& db_factors,
        const QueryFactorsData& query_factors,
        bool centered,
        size_t qb,
        size_t d) {
    float adjusted_distance;

    if (centered) {
        // For centered mode: normalized_distance contains the raw XOR
        // contribution. Apply the signed odd integer quantization formula:
        // int_dot = ((1 << qb) - 1) * d - 2 * xor_dot_product
        int64_t int_dot = ((1 << qb) - 1) * d;
        int_dot -= 2 * static_cast<int64_t>(normalized_distance);

        adjusted_distance = query_factors.qr_to_c_L2sqr +
                db_factors.or_minus_c_l2sqr -
                2 * db_factors.dp_multiplier * int_dot *
                        query_factors.int_dot_scale;
    } else {
        // For non-centered quantization: use traditional formula
        float final_dot = normalized_distance - query_factors.c34;
        adjusted_distance = db_factors.or_minus_c_l2sqr +
                query_factors.qr_to_c_L2sqr -
                2 * db_factors.dp_multiplier * final_dot;
    }

    // Apply inner product correction if needed
    if (query_factors.qr_norm_L2sqr != 0.0f) {
        adjusted_distance =
                -0.5f * (adjusted_distance - query_factors.qr_norm_L2sqr);
    } else {
        adjusted_distance = std::max(0.0f, adjusted_distance);
    }

    return adjusted_distance;
}

/** Extract multi-bit code on-the-fly from packed ex-bit codes.
 * This inline function extracts a single code value without unpacking the
 * entire array, enabling efficient on-the-fly decoding during distance
 * computation.
 *
 * @param ex_code       packed ex-bit codes
 * @param index         which code to extract (0 to d-1)
 * @param ex_bits       number of bits per code (1-8)
 * @return              extracted code value in range [0, 2^ex_bits - 1]
 */
inline int extract_code_inline(
        const uint8_t* ex_code,
        size_t index,
        size_t ex_bits) {
    size_t bit_pos = index * ex_bits;
    int code_value = 0;

    // Extract ex_bits bits starting at bit_pos
    for (size_t bit = 0; bit < ex_bits; bit++) {
        size_t byte_idx = bit_pos / 8;
        size_t bit_idx = bit_pos % 8;

        if (ex_code[byte_idx] & (1 << bit_idx)) {
            code_value |= (1 << bit);
        }

        bit_pos++;
    }

    return code_value;
}

/** Compute full multi-bit distance from sign bits and ex-bit codes.
 * This is the core distance computation shared by RaBitQFastScan handlers.
 *
 * The multi-bit distance combines the sign bit (1-bit) with additional
 * magnitude bits (ex_bits) to compute a more accurate distance estimate.
 *
 * @param sign_bits       unpacked sign bits (1-bit codes in standard format)
 * @param ex_code         packed ex-bit codes
 * @param ex_fac          ex-bit factors (f_add_ex, f_rescale_ex)
 * @param rotated_q       rotated query vector
 * @param qr_to_c_L2sqr   precomputed ||query_rotated - centroid||^2
 * @param qr_norm_L2sqr   precomputed ||query_rotated||^2 (0 for L2 metric)
 * @param d               dimensionality
 * @param ex_bits         number of extra bits (nb_bits - 1)
 * @param metric_type     distance metric (L2 or Inner Product)
 * @return                computed full multi-bit distance
 */
inline float compute_full_multibit_distance(
        const uint8_t* sign_bits,
        const uint8_t* ex_code,
        const ExtraBitsFactors& ex_fac,
        const float* rotated_q,
        float qr_to_c_L2sqr,
        float qr_norm_L2sqr,
        size_t d,
        size_t ex_bits,
        MetricType metric_type) {
    float ex_ip = 0.0f;
    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);

    for (size_t i = 0; i < d; i++) {
        const size_t byte_idx = i / 8;
        const size_t bit_offset = i % 8;
        const bool sign_bit = (sign_bits[byte_idx] >> bit_offset) & 1;

        int ex_code_val = extract_code_inline(ex_code, i, ex_bits);

        int total_code = (sign_bit ? 1 : 0) << ex_bits;
        total_code += ex_code_val;
        float reconstructed = static_cast<float>(total_code) + cb;

        ex_ip += rotated_q[i] * reconstructed;
    }

    float dist = qr_to_c_L2sqr + ex_fac.f_add_ex + ex_fac.f_rescale_ex * ex_ip;

    if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
        dist = -0.5f * (dist - qr_norm_L2sqr);
    } else {
        dist = std::max(0.0f, dist);
    }

    return dist;
}

} // namespace rabitq_utils
} // namespace faiss
