/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQUtils.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {
namespace rabitq_utils {

// Verify no unexpected padding in structures used for per-vector storage.
// These checks ensure compute_per_vector_storage_size() remains accurate.
static_assert(
        sizeof(SignBitFactors) == 8,
        "SignBitFactors has unexpected padding");
static_assert(
        sizeof(SignBitFactorsWithError) == 12,
        "SignBitFactorsWithError has unexpected padding");
static_assert(
        sizeof(ExtraBitsFactors) == 8,
        "ExtraBitsFactors has unexpected padding");

// Ideal quantizer radii for quantizers of 1..8 bits, optimized to minimize
// L2 reconstruction error.
const float Z_MAX_BY_QB[8] = {
        0.79688, // qb = 1.
        1.49375,
        2.05078,
        2.50938,
        2.91250,
        3.26406,
        3.59844,
        3.91016, // qb = 8.
};

void compute_vector_intermediate_values(
        const float* x,
        size_t d,
        const float* centroid,
        float& norm_L2sqr,
        float& or_L2sqr,
        float& dp_oO) {
    norm_L2sqr = 0.0f;
    or_L2sqr = 0.0f;
    dp_oO = 0.0f;

    for (size_t j = 0; j < d; j++) {
        const float x_val = x[j];
        const float centroid_val = (centroid != nullptr) ? centroid[j] : 0.0f;
        const float or_minus_c = x_val - centroid_val;

        const float or_minus_c_sq = or_minus_c * or_minus_c;
        norm_L2sqr += or_minus_c_sq;
        or_L2sqr += x_val * x_val;

        const bool xb = (or_minus_c > 0.0f);
        dp_oO += xb ? or_minus_c : -or_minus_c;
    }
}

SignBitFactorsWithError compute_factors_from_intermediates(
        float norm_L2sqr,
        float or_L2sqr,
        float dp_oO,
        size_t d,
        MetricType metric_type,
        bool compute_error) {
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    constexpr float kConstEpsilon =
            1.9f; // Error bound constant from RaBitQ paper
    const float inv_d_sqrt =
            (d == 0) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(d)));

    const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
    const float inv_norm_L2 =
            (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);

    const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
    const float inv_dp_oO =
            (std::abs(normalized_dp) < epsilon) ? 1.0f : (1.0f / normalized_dp);

    SignBitFactorsWithError factors;
    factors.or_minus_c_l2sqr = (metric_type == MetricType::METRIC_INNER_PRODUCT)
            ? (norm_L2sqr - or_L2sqr)
            : norm_L2sqr;
    factors.dp_multiplier = inv_dp_oO * sqrt_norm_L2;

    // Compute error bound only if needed (skip for 1-bit mode)
    if (compute_error) {
        const float xu_cb_norm_sqr = static_cast<float>(d) * 0.25f;
        const float ip_resi_xucb = 0.5f * dp_oO;

        float tmp_error = 0.0f;
        if (std::abs(ip_resi_xucb) > epsilon) {
            const float ratio_sq = (norm_L2sqr * xu_cb_norm_sqr) /
                    (ip_resi_xucb * ip_resi_xucb);
            if (ratio_sq > 1.0f) {
                if (d == 1) {
                    tmp_error = sqrt_norm_L2 * kConstEpsilon *
                            std::sqrt(ratio_sq - 1.0f);
                } else {
                    tmp_error = sqrt_norm_L2 * kConstEpsilon *
                            std::sqrt((ratio_sq - 1.0f) /
                                      static_cast<float>(d - 1));
                }
            }
        }

        // Apply metric-specific multiplier
        if (metric_type == MetricType::METRIC_L2) {
            factors.f_error = 2.0f * tmp_error;
        } else if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
            factors.f_error = 1.0f * tmp_error;
        } else {
            factors.f_error = 0.0f;
        }
    }

    return factors;
}

SignBitFactorsWithError compute_vector_factors(
        const float* x,
        size_t d,
        const float* centroid,
        MetricType metric_type,
        bool compute_error) {
    float norm_L2sqr, or_L2sqr, dp_oO;
    compute_vector_intermediate_values(
            x, d, centroid, norm_L2sqr, or_L2sqr, dp_oO);
    return compute_factors_from_intermediates(
            norm_L2sqr, or_L2sqr, dp_oO, d, metric_type, compute_error);
}

QueryFactorsData compute_query_factors(
        const float* query,
        size_t d,
        const float* centroid,
        uint8_t qb,
        bool centered,
        MetricType metric_type,
        std::vector<float>& rotated_q,
        std::vector<uint8_t>& rotated_qq) {
    FAISS_THROW_IF_NOT(qb <= 8);
    FAISS_THROW_IF_NOT(qb > 0);

    QueryFactorsData query_factors;

    // Compute distance from query to centroid
    if (centroid != nullptr) {
        query_factors.qr_to_c_L2sqr = fvec_L2sqr(query, centroid, d);
    } else {
        query_factors.qr_to_c_L2sqr = fvec_norm_L2sqr(query, d);
    }
    query_factors.g_error = std::sqrt(query_factors.qr_to_c_L2sqr);

    // Rotate the query (subtract centroid)
    rotated_q.resize(d);
    for (size_t i = 0; i < d; i++) {
        if (i < rotated_q.size()) {
            rotated_q[i] =
                    query[i] - ((centroid == nullptr) ? 0.0f : centroid[i]);
        }
    }

    const float inv_d_sqrt =
            (d == 0) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(d)));

    // Compute quantization range
    float v_min = std::numeric_limits<float>::max();
    float v_max = std::numeric_limits<float>::lowest();

    if (centered) {
        float z_max = Z_MAX_BY_QB[qb - 1];
        float v_radius = z_max * std::sqrt(query_factors.qr_to_c_L2sqr / d);
        v_min = -v_radius;
        v_max = v_radius;
    } else {
        // Only compute min/max if we have dimensions to process
        if (d > 0 && !rotated_q.empty()) {
            for (size_t i = 0; i < d; i++) {
                const float v_q = rotated_q[i];
                v_min = std::min(v_min, v_q);
                v_max = std::max(v_max, v_q);
            }
        } else {
            // For empty dimensions, use default range
            v_min = 0.0f;
            v_max = 1.0f;
        }
    }

    // Quantize the query
    const uint8_t max_code = (1 << qb) - 1;
    const float delta = (v_max - v_min) / max_code;
    const float inv_delta = 1.0f / delta;

    rotated_qq.resize(d);
    size_t sum_qq = 0;
    int64_t sum2_signed_odd_int = 0;

    // Process arrays - throw error if they are unexpectedly empty
    if (d > 0 && !rotated_q.empty() && !rotated_qq.empty()) {
        for (size_t i = 0; i < d; i++) {
            const float v_q = rotated_q[i];
            // Non-randomized scalar quantization
            const uint8_t v_qq = std::clamp<float>(
                    std::round((v_q - v_min) * inv_delta), 0, max_code);
            rotated_qq[i] = v_qq;
            sum_qq += v_qq;

            if (centered) {
                int64_t signed_odd_int = int64_t(v_qq) * 2 - max_code;
                sum2_signed_odd_int += signed_odd_int * signed_odd_int;
            }
        }
    } else {
        FAISS_THROW_MSG(
                "Arrays unexpectedly empty when d=" + std::to_string(d) +
                "or d is incorrectly set");
    }

    // Compute query factors
    query_factors.c1 = 2 * delta * inv_d_sqrt;
    query_factors.c2 = 2 * v_min * inv_d_sqrt;
    query_factors.c34 = inv_d_sqrt * (delta * sum_qq + d * v_min);

    if (centered) {
        query_factors.int_dot_scale = std::sqrt(
                query_factors.qr_to_c_L2sqr / (sum2_signed_odd_int * d));
    } else {
        query_factors.int_dot_scale = 1.0f;
    }

    // Compute query norm for inner product metric
    query_factors.qr_norm_L2sqr = 0.0f;
    if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
        query_factors.qr_norm_L2sqr = fvec_norm_L2sqr(query, d);
    }

    return query_factors;
}

bool extract_bit_standard(const uint8_t* code, size_t bit_index) {
    const size_t byte_idx = bit_index / 8;
    const size_t bit_offset = bit_index % 8;
    return (code[byte_idx] >> bit_offset) & 1;
}

bool extract_bit_fastscan(const uint8_t* code, size_t bit_index) {
    const size_t m = bit_index / 4; // Sub-quantizer index
    const size_t dim_offset =
            bit_index % 4;         // Bit position within sub-quantizer
    const size_t byte_idx = m / 2; // Byte index (2 sub-quantizers per byte)
    const uint8_t bit_mask = static_cast<uint8_t>(1 << dim_offset);

    if (m % 2 == 0) {
        // Lower 4 bits of byte
        return (code[byte_idx] & bit_mask) != 0;
    } else {
        // Upper 4 bits of byte (shifted)
        return (code[byte_idx] & (bit_mask << 4)) != 0;
    }
}

void set_bit_standard(uint8_t* code, size_t bit_index) {
    const size_t byte_idx = bit_index / 8;
    const size_t bit_offset = bit_index % 8;
    code[byte_idx] |= (1 << bit_offset);
}

void set_bit_fastscan(uint8_t* code, size_t bit_index) {
    const size_t m = bit_index / 4;
    const size_t dim_offset = bit_index % 4;
    const uint8_t bit_mask = static_cast<uint8_t>(1 << dim_offset);
    const size_t byte_idx = m / 2;

    if (m % 2 == 0) {
        code[byte_idx] |= bit_mask;
    } else {
        code[byte_idx] |= (bit_mask << 4);
    }
}

} // namespace rabitq_utils
} // namespace faiss
