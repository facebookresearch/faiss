/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQuantizer.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizerMultiBit.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::BaseFactorsData;
using rabitq_utils::ExFactorsData;
using rabitq_utils::FactorsData;
using rabitq_utils::QueryFactorsData;

RaBitQuantizer::RaBitQuantizer(size_t d, MetricType metric, size_t nb_bits)
        : Quantizer(d, 0), // code_size will be set below
          metric_type{metric},
          nb_bits{nb_bits} {
    // Validate nb_bits range
    FAISS_THROW_IF_NOT(nb_bits >= 1 && nb_bits <= 9);

    // Set code_size using compute_code_size
    code_size = compute_code_size(d, nb_bits);
}

size_t RaBitQuantizer::compute_code_size(size_t d, size_t num_bits) const {
    // Validate inputs
    FAISS_THROW_IF_NOT(num_bits >= 1 && num_bits <= 9);

    size_t ex_bits = num_bits - 1;

    // Base: 1-bit codes + base factors
    // Layout for 1-bit: [binary_code: (d+7)/8 bytes][BaseFactorsData: 8 bytes]
    //   base_factors = or_minus_c_l2sqr (4) + dp_multiplier (4)
    // Layout for multi-bit: [binary_code: (d+7)/8 bytes][FactorsData: 12 bytes]
    //   factors = or_minus_c_l2sqr (4) + dp_multiplier (4) + f_error (4)
    size_t base_size = (d + 7) / 8 +
            (ex_bits == 0 ? sizeof(BaseFactorsData) : sizeof(FactorsData));

    // Extra: ex-bit codes + ex factors (only if ex_bits > 0)
    // Layout: [ex_code: (d*ex_bits+7)/8 bytes][ex_factors: 8 bytes]
    size_t ex_size = 0;
    if (ex_bits > 0) {
        ex_size = (d * ex_bits + 7) / 8 + sizeof(ExFactorsData);
    }

    return base_size + ex_size;
}

void RaBitQuantizer::train(size_t n, const float* x) {
    // does nothing
}

void RaBitQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    compute_codes_core(x, codes, n, centroid);
}

void RaBitQuantizer::compute_codes_core(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid_in) const {
    FAISS_ASSERT(codes != nullptr);
    FAISS_ASSERT(x != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));

    if (n == 0) {
        return;
    }

    const size_t ex_bits = nb_bits - 1;

    // Compute codes
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        // Pointer to this vector's code
        uint8_t* code = codes + i * code_size;

        // Clear code memory
        memset(code, 0, code_size);

        const float* x_row = x + i * d;

        // Pointer arithmetic for code layout:
        // For 1-bit: [binary_code: (d+7)/8 bytes][BaseFactorsData: 8 bytes]
        // For multi-bit: [binary_code: (d+7)/8 bytes][FactorsData: 12 bytes]
        //                [ex_code: (d*ex_bits+7)/8 bytes][ex_factors: 8 bytes]
        uint8_t* binary_code = code;

        // Step 1: Compute 1-bit quantization and base factors
        // Store residual for potential ex-bits quantization
        std::vector<float> residual(d);

        // Use shared utilities for computing factors
        FactorsData factors_data = rabitq_utils::compute_vector_factors(
                x_row, d, centroid_in, metric_type);

        // Write appropriate factors based on nb_bits
        if (ex_bits == 0) {
            // For 1-bit: write only BaseFactorsData (8 bytes)
            BaseFactorsData* base_factors =
                    reinterpret_cast<BaseFactorsData*>(code + (d + 7) / 8);
            base_factors->or_minus_c_l2sqr = factors_data.or_minus_c_l2sqr;
            base_factors->dp_multiplier = factors_data.dp_multiplier;
        } else {
            // For multi-bit: write full FactorsData (12 bytes)
            FactorsData* full_factors =
                    reinterpret_cast<FactorsData*>(code + (d + 7) / 8);
            *full_factors = factors_data;
        }

        // Pack bits into standard RaBitQ format
        for (size_t j = 0; j < d; j++) {
            const float x_val = x_row[j];
            const float centroid_val =
                    (centroid_in == nullptr) ? 0.0f : centroid_in[j];
            const float or_minus_c = x_val - centroid_val;
            residual[j] = or_minus_c;

            const bool xb = (or_minus_c > 0.0f);

            // Store the 1-bit sign code
            if (xb) {
                rabitq_utils::set_bit_standard(binary_code, j);
            }
        }

        // Step 2: Compute ex-bits quantization (if nb_bits > 1)
        if (ex_bits > 0) {
            // Pointer to ex-bit code section
            uint8_t* ex_code = code + (d + 7) / 8 + sizeof(FactorsData);
            // Pointer to ex-factors section
            ExFactorsData* ex_factors = reinterpret_cast<ExFactorsData*>(
                    ex_code + (d * ex_bits + 7) / 8);

            // Quantize residual to ex-bits (pass centroid for IP metric)
            rabitq_multibit::quantize_ex_bits(
                    residual.data(),
                    d,
                    nb_bits,
                    ex_code,
                    *ex_factors,
                    metric_type,
                    centroid_in);
        }
    }
}

void RaBitQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    decode_core(codes, x, n, centroid);
}

void RaBitQuantizer::decode_core(
        const uint8_t* codes,
        float* x,
        size_t n,
        const float* centroid_in) const {
    FAISS_ASSERT(codes != nullptr);
    FAISS_ASSERT(x != nullptr);

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));
    const size_t ex_bits = nb_bits - 1;

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const uint8_t* code = codes + i * code_size;

        // split the code into parts
        const uint8_t* binary_data = code;

        // Cast to appropriate type based on nb_bits
        // For 1-bit: use BaseFactorsData (8 bytes)
        // For multi-bit: use FactorsData (12 bytes, but only first 8 bytes used
        // for decode)
        const BaseFactorsData* fac = (ex_bits == 0)
                ? reinterpret_cast<const BaseFactorsData*>(code + (d + 7) / 8)
                : reinterpret_cast<const FactorsData*>(code + (d + 7) / 8);

        // this is the baseline code
        //
        // compute <q,o> using floats
        for (size_t j = 0; j < d; j++) {
            // extract i-th bit
            const uint8_t masker = (1 << (j % 8));
            const float bit = ((binary_data[j / 8] & masker) == masker) ? 1 : 0;

            // compute the output code
            x[i * d + j] = (bit - 0.5f) * fac->dp_multiplier * 2 * inv_d_sqrt +
                    ((centroid_in == nullptr) ? 0 : centroid_in[j]);
        }
    }
}

// Implementation of RaBitQDistanceComputer (declared in header)

float RaBitQDistanceComputer::lower_bound_distance(const uint8_t* code) {
    FAISS_ASSERT(code != nullptr);

    // Compute estimated distance using 1-bit codes
    float est_distance = distance_to_code_1bit(code);

    // Extract f_error from the code
    size_t size = (d + 7) / 8;
    const FactorsData* base_fac =
            reinterpret_cast<const FactorsData*>(code + size);
    float f_error = base_fac->f_error;

    // Compute proper lower bound using RaBitQ error formula:
    // lower_bound = est_distance - f_error * g_error
    // This guarantees: lower_bound ≤ true_distance
    float lower_bound = est_distance - (f_error * g_error);

    // Distance cannot be negative
    return std::max(0.0f, lower_bound);
}

namespace {

struct RaBitQDistanceComputerNotQ : RaBitQDistanceComputer {
    // the rotated query (qr - c)
    std::vector<float> rotated_q;
    // some additional numbers for the query
    QueryFactorsData query_fac;

    RaBitQDistanceComputerNotQ();

    // Compute distance using only 1-bit codes (fast)
    float distance_to_code_1bit(const uint8_t* code) override;

    // Compute full distance using 1-bit + ex-bits (accurate)
    float distance_to_code_full(const uint8_t* code) override;

    void set_query(const float* x) override;
};

RaBitQDistanceComputerNotQ::RaBitQDistanceComputerNotQ() = default;

float RaBitQDistanceComputerNotQ::distance_to_code_1bit(const uint8_t* code) {
    FAISS_ASSERT(code != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));
    FAISS_ASSERT(rotated_q.size() == d);

    // split the code into parts
    const uint8_t* binary_data = code;

    // Cast to appropriate type based on nb_bits
    // For 1-bit: use BaseFactorsData (8 bytes)
    // For multi-bit: use FactorsData (12 bytes) which includes f_error
    size_t ex_bits = nb_bits - 1;
    const BaseFactorsData* base_fac = (ex_bits == 0)
            ? reinterpret_cast<const BaseFactorsData*>(code + (d + 7) / 8)
            : reinterpret_cast<const FactorsData*>(code + (d + 7) / 8);

    // this is the baseline code
    //
    // compute <q,o> using floats
    float dot_qo = 0;
    // It was a willful decision (after the discussion) to not to pre-cache
    //   the sum of all bits, just in order to reduce the overhead per vector.
    uint64_t sum_q = 0;

    for (size_t i = 0; i < d; i++) {
        // Extract i-th bit
        bool bit = rabitq_utils::extract_bit_standard(binary_data, i);
        // accumulate dp
        dot_qo += bit ? rotated_q[i] : 0;
        // accumulate sum-of-bits
        sum_q += bit ? 1 : 0;
    }

    // Apply query factors
    float final_dot =
            query_fac.c1 * dot_qo + query_fac.c2 * sum_q - query_fac.c34;

    // pre_dist = ||or - c||^2 + ||qr - c||^2 -
    //     2 * ||or - c|| * ||qr - c|| * <q,o> - (IP ? ||or||^2 : 0)
    float pre_dist = base_fac->or_minus_c_l2sqr + query_fac.qr_to_c_L2sqr -
            2 * base_fac->dp_multiplier * final_dot;

    if (metric_type == MetricType::METRIC_L2) {
        // ||or - q||^ 2
        return pre_dist;
    } else {
        // metric == MetricType::METRIC_INNER_PRODUCT
        return -0.5f * (pre_dist - query_fac.qr_norm_L2sqr);
    }
}

float RaBitQDistanceComputerNotQ::distance_to_code_full(const uint8_t* code) {
    FAISS_ASSERT(code != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));
    FAISS_ASSERT(rotated_q.size() == d);

    size_t ex_bits = nb_bits - 1;

    if (ex_bits == 0) {
        // No ex-bits, just return 1-bit distance
        return distance_to_code_1bit(code);
    }

    // Extract pointers to code sections
    const uint8_t* binary_data = code;
    size_t offset = (d + 7) / 8 + sizeof(FactorsData);
    const uint8_t* ex_code = code + offset;
    const ExFactorsData* ex_fac = reinterpret_cast<const ExFactorsData*>(
            ex_code + (d * ex_bits + 7) / 8);

    // Compute inner product with on-the-fly code extraction
    // OPTIMIZATION: Extract codes during dot product loop instead of unpacking
    // all codes first. This eliminates buffer allocation, reduces memory
    // accesses, and improves cache locality (5-10x speedup expected).
    float ex_ip = 0;
    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);

    for (size_t i = 0; i < d; i++) {
        // Get sign bit from 1-bit codes (0 or 1)
        bool sign_bit = rabitq_utils::extract_bit_standard(binary_data, i);

        // Extract magnitude code on-the-fly (no unpacking required)
        int ex_code_val =
                rabitq_utils::extract_code_inline(ex_code, i, ex_bits);

        // Form total code: total_code = sign_bit * 2^ex_bits + ex_code
        int total_code = (sign_bit ? 1 : 0) << ex_bits;
        total_code += ex_code_val;

        // Reconstruction: u_cb[i] = total_code[i] + cb
        float reconstructed = static_cast<float>(total_code) + cb;

        // Compute contribution to inner product
        ex_ip += rotated_q[i] * reconstructed;
    }

    // Compute refined distance using ex-bits factors
    // L2^2 = ||query||^2 + ||db||^2 - 2*dot(query,db)
    //      = qr_to_c_L2sqr + f_add_ex + f_rescale_ex * ex_ip
    float refined_dist = query_fac.qr_to_c_L2sqr + ex_fac->f_add_ex +
            ex_fac->f_rescale_ex * ex_ip;

    if (metric_type == MetricType::METRIC_L2) {
        return refined_dist;
    } else {
        // 2 * (or, q) = (||or - q||^2 - ||q||^2 - ||or||^2)
        return -0.5f * (refined_dist - query_fac.qr_norm_L2sqr);
    }
}

void RaBitQDistanceComputerNotQ::set_query(const float* x) {
    q = x;
    FAISS_ASSERT(x != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));

    // compute the distance from the query to the centroid
    if (centroid != nullptr) {
        query_fac.qr_to_c_L2sqr = fvec_L2sqr(x, centroid, d);
    } else {
        query_fac.qr_to_c_L2sqr = fvec_norm_L2sqr(x, d);
    }

    // subtract c, obtain P^(-1)(qr - c)
    rotated_q.resize(d);
    for (size_t i = 0; i < d; i++) {
        rotated_q[i] = x[i] - ((centroid == nullptr) ? 0 : centroid[i]);
    }

    // Compute g_error (query norm for lower bound computation)
    // g_error = ||qr - c|| (L2 norm of rotated query)
    g_error = std::sqrt(query_fac.qr_to_c_L2sqr);

    // compute some numbers
    const float inv_d = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    // do not quantize the query
    float sum_q = 0;
    for (size_t i = 0; i < d; i++) {
        sum_q += rotated_q[i];
    }

    query_fac.c1 = 2 * inv_d;
    query_fac.c2 = 0;
    query_fac.c34 = sum_q * inv_d;

    if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
        // precompute if needed
        query_fac.qr_norm_L2sqr = fvec_norm_L2sqr(x, d);
    }
}

//
struct RaBitQDistanceComputerQ : RaBitQDistanceComputer {
    // the rotated and quantized query (qr - c)
    std::vector<float> rotated_q;
    // the rotated and quantized query (qr - c) for fast 1-bit computation
    std::vector<uint8_t> rotated_qq;
    // we're using the proposed relayout-ed scheme from 3.3 that allows
    //    using popcounts for computing the distance.
    std::vector<uint8_t> rearranged_rotated_qq;
    // some additional numbers for the query
    QueryFactorsData query_fac;

    // the number of bits for SQ quantization of the query (qb > 0)
    uint8_t qb = 8;
    bool centered = false;
    // the smallest value divisible by 8 that is not smaller than dim
    size_t popcount_aligned_dim = 0;

    RaBitQDistanceComputerQ();

    // Compute distance using only 1-bit codes (fast)
    float distance_to_code_1bit(const uint8_t* code) override;

    // Compute full distance using 1-bit + ex-bits (accurate)
    float distance_to_code_full(const uint8_t* code) override;

    void set_query(const float* x) override;
};

RaBitQDistanceComputerQ::RaBitQDistanceComputerQ() = default;

float RaBitQDistanceComputerQ::distance_to_code_1bit(const uint8_t* code) {
    FAISS_ASSERT(code != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));

    // split the code into parts
    size_t size = (d + 7) / 8;
    const uint8_t* binary_data = code;

    // Cast to appropriate type based on nb_bits
    // For 1-bit: use BaseFactorsData (8 bytes)
    // For multi-bit: use FactorsData (12 bytes) which includes f_error
    size_t ex_bits = nb_bits - 1;
    const BaseFactorsData* base_fac = (ex_bits == 0)
            ? reinterpret_cast<const BaseFactorsData*>(code + size)
            : reinterpret_cast<const FactorsData*>(code + size);

    // this is ||or - c||^2 - (IP ? ||or||^2 : 0)
    float final_dot = 0;
    if (centered) {
        int64_t int_dot = ((1 << qb) - 1) * d;
        // See RaBitDistanceComputerNotQ::distance_to_code() for baseline code.
        int_dot -= 2 *
                rabitq::bitwise_xor_dot_product(
                           rearranged_rotated_qq.data(), binary_data, size, qb);
        final_dot += int_dot * query_fac.int_dot_scale;
    } else {
        auto dot_qo = rabitq::bitwise_and_dot_product(
                rearranged_rotated_qq.data(), binary_data, size, qb);
        // It was a willful decision (after the discussion) to not to pre-cache
        // the sum of all bits, just in order to reduce the overhead per vector.
        // process 64-bit popcounts
        auto sum_q = rabitq::popcount(binary_data, size);
        // dot-product itself
        final_dot += query_fac.c1 * dot_qo;
        // normalizer coefficients
        final_dot += query_fac.c2 * sum_q;
        // normalizer coefficients
        final_dot -= query_fac.c34;
    }

    // pre_dist = ||or - c||^2 + ||qr - c||^2 -
    //     2 * ||or - c|| * ||qr - c|| * <q,o> - (IP ? ||or||^2 : 0)
    const float pre_dist = base_fac->or_minus_c_l2sqr +
            query_fac.qr_to_c_L2sqr - 2 * base_fac->dp_multiplier * final_dot;

    if (metric_type == MetricType::METRIC_L2) {
        // ||or - q||^ 2
        return pre_dist;
    } else {
        // metric == MetricType::METRIC_INNER_PRODUCT
        // 2 * (or, q) = (||or - q||^2 - ||q||^2 - ||or||^2)
        return -0.5f * (pre_dist - query_fac.qr_norm_L2sqr);
    }
}

float RaBitQDistanceComputerQ::distance_to_code_full(const uint8_t* code) {
    FAISS_ASSERT(code != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));
    FAISS_ASSERT(rotated_q.size() == d);

    size_t ex_bits = nb_bits - 1;

    if (ex_bits == 0) {
        // No ex-bits, just return 1-bit distance
        return distance_to_code_1bit(code);
    }

    // Extract pointers to code sections
    const uint8_t* binary_data = code;
    size_t offset = (d + 7) / 8 + sizeof(FactorsData);
    const uint8_t* ex_code = code + offset;
    const ExFactorsData* ex_fac = reinterpret_cast<const ExFactorsData*>(
            ex_code + (d * ex_bits + 7) / 8);

    // Compute inner product with on-the-fly code extraction
    // OPTIMIZATION: Extract codes during dot product loop instead of unpacking
    // all codes first. This eliminates buffer allocation, reduces memory
    // accesses, and improves cache locality (5-10x speedup expected).
    float ex_ip = 0;
    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);

    for (size_t i = 0; i < d; i++) {
        // Get sign bit from 1-bit codes (0 or 1)
        bool sign_bit = rabitq_utils::extract_bit_standard(binary_data, i);

        // Extract magnitude code on-the-fly (no unpacking required)
        int ex_code_val =
                rabitq_utils::extract_code_inline(ex_code, i, ex_bits);

        // Form total code: total_code = sign_bit * 2^ex_bits + ex_code
        int total_code = (sign_bit ? 1 : 0) << ex_bits;
        total_code += ex_code_val;

        // Reconstruction: u_cb[i] = total_code[i] + cb
        float reconstructed = static_cast<float>(total_code) + cb;

        // Compute contribution to inner product using FLOAT query
        // CRITICAL: Must use rotated_q (float) instead of rotated_qq
        // (quantized) because ex_factors were precomputed during encoding using
        // float residuals. Using the same precision at search time maintains
        // encoding/search consistency.
        ex_ip += rotated_q[i] * reconstructed;
    }

    // Compute refined distance using ex-bits factors
    // L2^2 = ||query||^2 + ||db||^2 - 2*dot(query,db)
    //      = qr_to_c_L2sqr + f_add_ex + f_rescale_ex * ex_ip
    float refined_dist = query_fac.qr_to_c_L2sqr + ex_fac->f_add_ex +
            ex_fac->f_rescale_ex * ex_ip;

    if (metric_type == MetricType::METRIC_L2) {
        return refined_dist;
    } else {
        return -0.5f * (refined_dist - query_fac.qr_norm_L2sqr);
    }
}

// Use shared constant from RaBitQUtils
using rabitq_utils::Z_MAX_BY_QB;

void RaBitQDistanceComputerQ::set_query(const float* x) {
    q = x;
    FAISS_ASSERT(x != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));
    FAISS_THROW_IF_NOT(qb <= 8);
    FAISS_THROW_IF_NOT(qb > 0);

    // Use shared utilities for core query factor computation
    // rotated_q is populated directly by compute_query_factors as an output
    // parameter
    query_fac = rabitq_utils::compute_query_factors(
            x, d, centroid, qb, centered, metric_type, rotated_q, rotated_qq);

    // Compute g_error (query norm for lower bound computation)
    // g_error = ||qr - c|| (L2 norm of rotated query)
    g_error = std::sqrt(query_fac.qr_to_c_L2sqr);

    // Rearrange the query vector for SIMD operations (RaBitQuantizer-specific)
    popcount_aligned_dim = ((d + 7) / 8) * 8;
    size_t offset = (d + 7) / 8;

    rearranged_rotated_qq.resize(offset * qb);
    std::fill(rearranged_rotated_qq.begin(), rearranged_rotated_qq.end(), 0);

    for (size_t idim = 0; idim < d; idim++) {
        for (size_t iv = 0; iv < qb; iv++) {
            const bool bit = ((rotated_qq[idim] & (1 << iv)) != 0);
            rearranged_rotated_qq[iv * offset + idim / 8] |=
                    bit ? (1 << (idim % 8)) : 0;
        }
    }
}

} // anonymous namespace

FlatCodesDistanceComputer* RaBitQuantizer::get_distance_computer(
        uint8_t qb,
        const float* centroid_in,
        bool centered) const {
    if (qb == 0) {
        auto dc = std::make_unique<RaBitQDistanceComputerNotQ>();
        dc->metric_type = metric_type;
        dc->d = d;
        dc->centroid = centroid_in;
        dc->nb_bits = nb_bits;

        return dc.release();
    } else {
        auto dc = std::make_unique<RaBitQDistanceComputerQ>();
        dc->metric_type = metric_type;
        dc->d = d;
        dc->centroid = centroid_in;
        dc->qb = qb;
        dc->centered = centered;
        dc->nb_bits = nb_bits;

        return dc.release();
    }
}

} // namespace faiss
