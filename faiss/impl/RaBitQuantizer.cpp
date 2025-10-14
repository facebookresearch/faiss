/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQuantizer.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::FactorsData;
using rabitq_utils::QueryFactorsData;

static size_t get_code_size(const size_t d) {
    return (d + 7) / 8 + sizeof(FactorsData);
}

RaBitQuantizer::RaBitQuantizer(size_t d, MetricType metric)
        : Quantizer(d, get_code_size(d)), metric_type{metric} {}

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

    // compute codes
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        // the code
        uint8_t* code = codes + i * code_size;
        FactorsData* fac = reinterpret_cast<FactorsData*>(code + (d + 7) / 8);

        // cleanup it
        if (code != nullptr) {
            memset(code, 0, code_size);
        }

        const float* x_row = x + i * d;

        // Use shared utilities for computing factors
        *fac = rabitq_utils::compute_vector_factors(
                x_row, d, centroid_in, metric_type);

        // Pack bits into standard RaBitQ format
        for (size_t j = 0; j < d; j++) {
            const float x_val = x_row[j];
            const float centroid_val =
                    (centroid_in == nullptr) ? 0.0f : centroid_in[j];
            const float or_minus_c = x_val - centroid_val;
            const bool xb = (or_minus_c > 0.0f);

            // store the output data
            if (code != nullptr && xb) {
                rabitq_utils::set_bit_standard(code, j);
            }
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

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const uint8_t* code = codes + i * code_size;

        // split the code into parts
        const uint8_t* binary_data = code;
        const FactorsData* fac =
                reinterpret_cast<const FactorsData*>(code + (d + 7) / 8);

        //
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

struct RaBitDistanceComputer : FlatCodesDistanceComputer {
    // dimensionality
    size_t d = 0;
    // a centroid to use
    const float* centroid = nullptr;

    // the metric
    MetricType metric_type = MetricType::METRIC_L2;

    RaBitDistanceComputer();

    float symmetric_dis(idx_t i, idx_t j) override;
};

RaBitDistanceComputer::RaBitDistanceComputer() = default;

float RaBitDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    FAISS_THROW_MSG("Not implemented");
}

struct RaBitDistanceComputerNotQ : RaBitDistanceComputer {
    // the rotated query (qr - c)
    std::vector<float> rotated_q;
    // some additional numbers for the query
    QueryFactorsData query_fac;

    RaBitDistanceComputerNotQ();

    float distance_to_code(const uint8_t* code) override;

    void set_query(const float* x) override;
};

RaBitDistanceComputerNotQ::RaBitDistanceComputerNotQ() = default;

float RaBitDistanceComputerNotQ::distance_to_code(const uint8_t* code) {
    FAISS_ASSERT(code != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));

    // split the code into parts
    const uint8_t* binary_data = code;
    const FactorsData* fac =
            reinterpret_cast<const FactorsData*>(code + (d + 7) / 8);

    // this is the baseline code
    //
    // compute <q,o> using floats
    float dot_qo = 0;
    // It was a willful decision (after the discussion) to not to pre-cache
    //   the sum of all bits, just in order to reduce the overhead per vector.
    uint64_t sum_q = 0;
    for (size_t i = 0; i < d; i++) {
        // extract i-th bit
        const uint8_t masker = (1 << (i % 8));
        const bool b_bit = ((binary_data[i / 8] & masker) == masker);

        // accumulate dp
        dot_qo += (b_bit) ? rotated_q[i] : 0;
        // accumulate sum-of-bits
        sum_q += (b_bit) ? 1 : 0;
    }

    float final_dot = 0;
    // dot-product itself
    final_dot += query_fac.c1 * dot_qo;
    // normalizer coefficients
    final_dot += query_fac.c2 * sum_q;
    // normalizer coefficients
    final_dot -= query_fac.c34;

    // this is ||or - c||^2 - (IP ? ||or||^2 : 0)
    const float or_c_l2sqr = fac->or_minus_c_l2sqr;

    // pre_dist = ||or - c||^2 + ||qr - c||^2 -
    //     2 * ||or - c|| * ||qr - c|| * <q,o> - (IP ? ||or||^2 : 0)
    const float pre_dist = or_c_l2sqr + query_fac.qr_to_c_L2sqr -
            2 * fac->dp_multiplier * final_dot;

    if (metric_type == MetricType::METRIC_L2) {
        // ||or - q||^ 2
        return pre_dist;
    } else {
        // metric == MetricType::METRIC_INNER_PRODUCT

        // this is ||q||^2
        const float query_norm_sqr = query_fac.qr_norm_L2sqr;

        // 2 * (or, q) = (||or - q||^2 - ||q||^2 - ||or||^2)
        return -0.5f * (pre_dist - query_norm_sqr);
    }
}

void RaBitDistanceComputerNotQ::set_query(const float* x) {
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
struct RaBitDistanceComputerQ : RaBitDistanceComputer {
    // the rotated and quantized query (qr - c)
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

    RaBitDistanceComputerQ();

    float distance_to_code(const uint8_t* code) override;

    void set_query(const float* x) override;
};

RaBitDistanceComputerQ::RaBitDistanceComputerQ() = default;

float RaBitDistanceComputerQ::distance_to_code(const uint8_t* code) {
    FAISS_ASSERT(code != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));

    // split the code into parts
    size_t size = (d + 7) / 8;
    const uint8_t* binary_data = code;
    const FactorsData* fac = reinterpret_cast<const FactorsData*>(code + size);

    // this is ||or - c||^2 - (IP ? ||or||^2 : 0)
    float final_dot = 0;
    if (centered) {
        int64_t int_dot = ((1 << qb) - 1) * d;
        int_dot -= 2 *
                rabitq::bitwise_xor_dot_product(
                           rearranged_rotated_qq.data(), binary_data, size, qb);
        final_dot += int_dot * query_fac.int_dot_scale;
    } else {
        // See RaBitDistanceComputerNotQ::distance_to_code() for baseline code.
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

    // this is ||or - c||^2 - (IP ? ||or||^2 : 0)
    const float or_c_l2sqr = fac->or_minus_c_l2sqr;

    // pre_dist = ||or - c||^2 + ||qr - c||^2 -
    //     2 * ||or - c|| * ||qr - c|| * <q,o> - (IP ? ||or||^2 : 0)
    const float pre_dist = or_c_l2sqr + query_fac.qr_to_c_L2sqr -
            2 * fac->dp_multiplier * final_dot;

    if (metric_type == MetricType::METRIC_L2) {
        // ||or - q||^ 2
        return pre_dist;
    } else {
        // metric == MetricType::METRIC_INNER_PRODUCT

        // this is ||q||^2
        const float query_norm_sqr = query_fac.qr_norm_L2sqr;

        // 2 * (or, q) = (||or - q||^2 - ||q||^2 - ||or||^2)
        return -0.5f * (pre_dist - query_norm_sqr);
    }
}

// Use shared constant from RaBitQUtils
using rabitq_utils::Z_MAX_BY_QB;

void RaBitDistanceComputerQ::set_query(const float* x) {
    FAISS_ASSERT(x != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));
    FAISS_THROW_IF_NOT(qb <= 8);
    FAISS_THROW_IF_NOT(qb > 0);

    // Use shared utilities for core query factor computation
    std::vector<float> rotated_q;
    query_fac = rabitq_utils::compute_query_factors(
            x, d, centroid, qb, centered, metric_type, rotated_q, rotated_qq);

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

FlatCodesDistanceComputer* RaBitQuantizer::get_distance_computer(
        uint8_t qb,
        const float* centroid_in,
        bool centered) const {
    if (qb == 0) {
        auto dc = std::make_unique<RaBitDistanceComputerNotQ>();
        dc->metric_type = metric_type;
        dc->d = d;
        dc->centroid = centroid_in;

        return dc.release();
    } else {
        auto dc = std::make_unique<RaBitDistanceComputerQ>();
        dc->metric_type = metric_type;
        dc->d = d;
        dc->centroid = centroid_in;
        dc->qb = qb;
        dc->centered = centered;

        return dc.release();
    }
}

} // namespace faiss
