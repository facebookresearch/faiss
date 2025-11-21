/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: Parts of this implementation are adapted from:
// RaBitQ-Library/include/rabitqlib/quantization/rabitq_impl.hpp
// https://github.com/VectorDB-NTU/RaBitQ-Library

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <vector>

namespace faiss {
namespace rabitq_multibit {

using rabitq_utils::ExFactorsData;
using rabitq_utils::FactorsData;

constexpr float kTightStart[9] =
        {0.0f, 0.15f, 0.20f, 0.52f, 0.59f, 0.71f, 0.75f, 0.77f, 0.81f};

constexpr double kEps = 1e-5;

/**
 * Compute optimal scaling factor for ex-bits quantization using priority
 * queue-based search.
 *
 * This function finds the optimal scaling factor 't' that maximizes the
 * inner product between the normalized quantized vector and the normalized
 * absolute residual. The algorithm uses a priority queue to efficiently
 * explore different quantization levels.
 *
 *
 * @param o_abs Normalized absolute residual vector (must be positive, length
 * d)
 * @param d Dimensionality of the vector
 * @param nb_bits Number of bits per dimension (2-9)
 * @return Optimal scaling factor 't'
 */
float compute_optimal_scaling_factor(
        const float* o_abs,
        size_t d,
        size_t nb_bits) {
    const size_t ex_bits = nb_bits - 1;
    FAISS_THROW_IF_NOT_MSG(
            ex_bits >= 1 && ex_bits <= 8, "ex_bits must be in range [1, 8]");

    const int kNEnum = 10;
    const int max_code = (1 << ex_bits) - 1;

    float max_o = *std::max_element(o_abs, o_abs + d);

    // Determine search range [t_start, t_end]
    float t_end = static_cast<float>(max_code + kNEnum) / max_o;
    float t_start = t_end * kTightStart[ex_bits];

    std::vector<float> inv_o_abs(d);
    for (size_t i = 0; i < d; ++i) {
        inv_o_abs[i] = 1.0f / o_abs[i];
    }

    std::vector<int> cur_o_bar(d);
    float sqr_denominator = static_cast<float>(d) * 0.25f;
    float numerator = 0.0f;

    for (size_t i = 0; i < d; ++i) {
        int cur = static_cast<int>((t_start * o_abs[i]) + kEps);
        cur_o_bar[i] = cur;
        sqr_denominator += static_cast<float>(cur * cur + cur);
        numerator += (cur + 0.5f) * o_abs[i];
    }

    float inv_sqrt_denom = 1.0f / std::sqrt(sqr_denominator);

    // Pair: (next_t, dimension_index)
    // Maximum size is d (one entry per dimension), so reserve exactly d
    std::vector<std::pair<float, size_t>> pq_storage;
    pq_storage.reserve(d);
    std::priority_queue<
            std::pair<float, size_t>,
            std::vector<std::pair<float, size_t>>,
            std::greater<>>
            next_t(std::greater<>(), std::move(pq_storage));

    // Initialize queue with next quantization level for each dimension
    for (size_t i = 0; i < d; ++i) {
        float t_next = static_cast<float>(cur_o_bar[i] + 1) * inv_o_abs[i];
        if (t_next < t_end) {
            next_t.emplace(t_next, i);
        }
    }

    float max_ip = 0.0f;
    float t = 0.0f;

    while (!next_t.empty()) {
        float cur_t = next_t.top().first;
        size_t update_id = next_t.top().second;
        next_t.pop();

        cur_o_bar[update_id]++;
        int update_o_bar = cur_o_bar[update_id];

        float delta = 2.0f * update_o_bar;
        sqr_denominator += delta;
        numerator += o_abs[update_id];

        float old_denom = sqr_denominator - delta;
        inv_sqrt_denom = inv_sqrt_denom *
                (1.0f - 0.5f * delta / (old_denom + delta * 0.5f));

        float cur_ip = numerator * inv_sqrt_denom;

        if (cur_ip > max_ip) {
            max_ip = cur_ip;
            t = cur_t;
        }

        if (update_o_bar < max_code) {
            float t_next =
                    static_cast<float>(update_o_bar + 1) * inv_o_abs[update_id];
            if (t_next < t_end) {
                next_t.emplace(t_next, update_id);
            }
        }
    }

    return t;
}

/**
 * Pack multi-bit codes from integer array to byte array.
 *
 * @param tmp_code Integer codes (length d), each value in [0, 2^ex_bits - 1]
 * @param ex_code Output packed byte array
 * @param d Dimensionality
 * @param nb_bits Number of bits per dimension (2-9)
 */
void pack_multibit_codes(
        const int* tmp_code,
        uint8_t* ex_code,
        size_t d,
        size_t nb_bits) {
    const size_t ex_bits = nb_bits - 1;
    FAISS_THROW_IF_NOT_MSG(
            ex_bits >= 1 && ex_bits <= 8, "ex_bits must be in range [1, 8]");

    size_t total_bits = d * ex_bits;
    size_t output_size = (total_bits + 7) / 8;
    memset(ex_code, 0, output_size);

    size_t bit_pos = 0;
    for (size_t i = 0; i < d; i++) {
        int code_value = tmp_code[i];

        for (size_t bit = 0; bit < ex_bits; bit++) {
            size_t byte_idx = bit_pos / 8;
            size_t bit_idx = bit_pos % 8;

            if (code_value & (1 << bit)) {
                ex_code[byte_idx] |= (1 << bit_idx);
            }

            bit_pos++;
        }
    }
}

/**
 * Compute ex-bits factors for distance computation.
 *
 * @param residual Original residual vector (data - centroid)
 * @param centroid Centroid vector (can be nullptr for zero centroid)
 * @param tmp_code Quantized ex-bit codes (before packing, after bit flipping)
 * @param d Dimensionality
 * @param ex_bits Number of extra bits
 * @param norm L2 norm of residual
 * @param ipnorm Unnormalized inner product between quantized and normalized
 * residual
 * @param ex_factors Output factors structure
 * @param metric_type Distance metric (L2 or Inner Product)
 */
void compute_ex_factors(
        const float* residual,
        const float* centroid,
        const int* tmp_code,
        size_t d,
        size_t ex_bits,
        float norm,
        double ipnorm,
        ExFactorsData& ex_factors,
        MetricType metric_type) {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2 ||
                    metric_type == MetricType::METRIC_INNER_PRODUCT,
            "Unsupported metric type");

    // Compute ipnorm_inv = 1 / ipnorm
    float ipnorm_inv = static_cast<float>(1.0 / ipnorm);
    if (!std::isnormal(ipnorm_inv)) {
        ipnorm_inv = 1.0f;
    }

    // Reconstruct xu_cb from total_code
    // total_code was formed from: total_code[i] = (sign << ex_bits) +
    // ex_code[i] Reconstruction: xu_cb[i] = total_code[i] + cb
    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);
    std::vector<float> xu_cb(d);
    for (size_t i = 0; i < d; i++) {
        xu_cb[i] = static_cast<float>(tmp_code[i]) + cb;
    }

    // Compute inner products needed for factors
    float l2_sqr = norm * norm;
    float ip_resi_xucb = fvec_inner_product(residual, xu_cb.data(), d);

    // Compute factors
    if (metric_type == MetricType::METRIC_L2) {
        // For L2, no centroid correction needed in IVF setting
        // because residual = x - centroid, distance computed in residual space
        ex_factors.f_add_ex = l2_sqr;
        ex_factors.f_rescale_ex = ipnorm_inv * -2.0f * norm;
    } else {
        // For IP, centroid correction is needed
        float ip_resi_cent = 0;
        if (centroid != nullptr) {
            ip_resi_cent = fvec_inner_product(residual, centroid, d);
        }

        float ip_cent_xucb = 0;
        if (centroid != nullptr) {
            ip_cent_xucb = fvec_inner_product(centroid, xu_cb.data(), d);
        }

        // When ip_resi_xucb is zero, the correction term should be zero
        float correction_term = 0.0f;
        if (ip_resi_xucb != 0.0f) {
            correction_term = l2_sqr * ip_cent_xucb / ip_resi_xucb;
        }

        ex_factors.f_add_ex = 1 - ip_resi_cent + correction_term;
        ex_factors.f_rescale_ex = ipnorm_inv * -norm;
    }
}

/**
 * Quantize residual vector to ex-bits.
 *
 * This is the main quantization function that:
 * 1. Normalizes the residual
 * 2. Takes absolute value
 * 3. Finds optimal scaling factor
 * 4. Quantizes to ex_bits
 * 5. Handles negative dimensions by flipping bits
 * 6. Packs codes into byte array
 * 7. Computes factors for distance computation
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
        ExFactorsData& ex_factors,
        MetricType metric_type,
        const float* centroid) {
    const size_t ex_bits = nb_bits - 1;
    FAISS_THROW_IF_NOT_MSG(
            ex_bits >= 1 && ex_bits <= 8, "ex_bits must be in range [1, 8]");
    FAISS_THROW_IF_NOT_MSG(residual != nullptr, "residual cannot be null");
    FAISS_THROW_IF_NOT_MSG(ex_code != nullptr, "ex_code cannot be null");

    // Step 1: Compute L2 norm of residual
    float norm_sqr = fvec_norm_L2sqr(residual, d);
    float norm = std::sqrt(norm_sqr);

    // Handle degenerate case
    if (norm < 1e-10f) {
        size_t code_size = (d * ex_bits + 7) / 8;
        memset(ex_code, 0, code_size);
        ex_factors.f_add_ex = 0.0f;
        ex_factors.f_rescale_ex = 1.0f;
        return;
    }

    // Step 2: Normalize residual
    std::vector<float> normalized_residual(d);
    for (size_t i = 0; i < d; i++) {
        normalized_residual[i] = residual[i] / norm;
    }

    // Step 3: Take absolute value
    std::vector<float> o_abs(d);
    for (size_t i = 0; i < d; i++) {
        o_abs[i] = std::abs(normalized_residual[i]);
    }

    // Step 4: Find optimal scaling factor
    float t = compute_optimal_scaling_factor(o_abs.data(), d, nb_bits);

    // Step 5: Quantize to ex_bits
    std::vector<int> tmp_code(d);
    double ipnorm = 0;
    int max_code = (1 << ex_bits) - 1;

    for (size_t i = 0; i < d; i++) {
        tmp_code[i] = std::min(static_cast<int>(t * o_abs[i] + kEps), max_code);
        // Compute unnormalized inner product
        ipnorm += (tmp_code[i] + 0.5) * o_abs[i];
    }

    // Step 6: Handle negative dimensions (flip bits)
    // For negative residuals, flip all bits: code' = ~code & max_code
    for (size_t i = 0; i < d; i++) {
        if (residual[i] < 0) {
            tmp_code[i] = (~tmp_code[i]) & max_code;
        }
    }

    // Step 7: Pack codes into byte array
    pack_multibit_codes(tmp_code.data(), ex_code, d, nb_bits);

    // Step 8: Compute factors for distance computation
    // Reconstruct total_code for factor computation
    std::vector<int> total_code(d);
    for (size_t i = 0; i < d; i++) {
        // Form total_code = (sign << ex_bits) + ex_code
        bool sign_bit = (residual[i] >= 0);
        total_code[i] = tmp_code[i] + ((sign_bit ? 1 : 0) << ex_bits);
    }

    // Compute ex-factors; centroid is needed for IP metric correction
    compute_ex_factors(
            residual,
            centroid, // Pass centroid for IP metric factor computation
            total_code.data(),
            d,
            ex_bits,
            norm,
            ipnorm,
            ex_factors,
            metric_type);
}

} // namespace rabitq_multibit
} // namespace faiss
