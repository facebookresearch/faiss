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

/** Factors computed per database vector for RaBitQ distance computation.
 * These can be stored either embedded in codes (IndexRaBitQ) or separately
 * (IndexRaBitQFastScan).
 */
struct FactorsData {
    // ||or - c||^2 - ((metric==IP) ? ||or||^2 : 0)
    float or_minus_c_l2sqr = 0;
    float dp_multiplier = 0;
};

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
 * @return              computed factors for distance computation
 */
FactorsData compute_vector_factors(
        const float* x,
        size_t d,
        const float* centroid,
        MetricType metric_type);

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
 * @return              computed factors
 */
FactorsData compute_factors_from_intermediates(
        float norm_L2sqr,
        float or_L2sqr,
        float dp_oO,
        size_t d,
        MetricType metric_type);

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

} // namespace rabitq_utils
} // namespace faiss
