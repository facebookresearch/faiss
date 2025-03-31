/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQuantizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

namespace faiss {

struct FactorsData {
    // ||or - c||^2 - ((metric==IP) ? ||or||^2 : 0)
    float or_minus_c_l2sqr = 0;
    float dp_multiplier = 0;
};

struct QueryFactorsData {
    float c1 = 0;
    float c2 = 0;
    float c34 = 0;

    float qr_to_c_L2sqr = 0;
    float qr_norm_L2sqr = 0;
};

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

    // compute some helper constants
    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    // compute codes
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        // ||or - c||^2
        float norm_L2sqr = 0;
        // ||or||^2, which is equal to ||P(or)||^2 and ||P^(-1)(or)||^2
        float or_L2sqr = 0;
        // dot product
        float dp_oO = 0;

        // the code
        uint8_t* code = codes + i * code_size;
        FactorsData* fac = reinterpret_cast<FactorsData*>(code + (d + 7) / 8);

        // cleanup it
        if (code != nullptr) {
            memset(code, 0, code_size);
        }

        for (size_t j = 0; j < d; j++) {
            const float or_minus_c = x[i * d + j] -
                    ((centroid_in == nullptr) ? 0 : centroid_in[j]);
            norm_L2sqr += or_minus_c * or_minus_c;
            or_L2sqr += x[i * d + j] * x[i * d + j];

            const bool xb = (or_minus_c > 0);

            dp_oO += xb ? or_minus_c : (-or_minus_c);

            // store the output data
            if (code != nullptr) {
                if (xb) {
                    // enable a particular bit
                    code[j / 8] |= (1 << (j % 8));
                }
            }
        }

        // compute factors

        // compute the inverse norm
        const float inv_norm_L2 =
                (std::abs(norm_L2sqr) < std::numeric_limits<float>::epsilon())
                ? 1.0f
                : (1.0f / std::sqrt(norm_L2sqr));
        dp_oO *= inv_norm_L2;
        dp_oO *= inv_d_sqrt;

        const float inv_dp_oO =
                (std::abs(dp_oO) < std::numeric_limits<float>::epsilon())
                ? 1.0f
                : (1.0f / dp_oO);

        fac->or_minus_c_l2sqr = norm_L2sqr;
        if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
            fac->or_minus_c_l2sqr -= or_L2sqr;
        }

        fac->dp_multiplier = inv_dp_oO * std::sqrt(norm_L2sqr);
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
    const uint8_t* binary_data = code;
    const FactorsData* fac =
            reinterpret_cast<const FactorsData*>(code + (d + 7) / 8);

    // // this is the baseline code
    // //
    // // compute <q,o> using integers
    // size_t dot_qo = 0;
    // for (size_t i = 0; i < d; i++) {
    //     // extract i-th bit
    //     const uint8_t masker = (1 << (i % 8));
    //     const uint8_t bit = ((binary_data[i / 8] & masker) == masker) ? 1 :
    //     0;
    //
    //     // accumulate dp
    //     dot_qo += bit * rotated_qq[i];
    // }

    // this is the scheme for popcount
    const size_t di_8b = (d + 7) / 8;
    const size_t di_64b = (di_8b / 8) * 8;

    uint64_t dot_qo = 0;
    for (size_t j = 0; j < qb; j++) {
        const uint8_t* query_j = rearranged_rotated_qq.data() + j * di_8b;

        // process 64-bit popcounts
        uint64_t count_dot = 0;
        for (size_t i = 0; i < di_64b; i += 8) {
            const auto qv = *(const uint64_t*)(query_j + i);
            const auto yv = *(const uint64_t*)(binary_data + i);
            count_dot += __builtin_popcountll(qv & yv);
        }

        // process leftovers
        for (size_t i = di_64b; i < di_8b; i++) {
            const auto qv = *(query_j + i);
            const auto yv = *(binary_data + i);
            count_dot += __builtin_popcount(qv & yv);
        }

        dot_qo += (count_dot << j);
    }

    // It was a willful decision (after the discussion) to not to pre-cache
    //   the sum of all bits, just in order to reduce the overhead per vector.
    uint64_t sum_q = 0;
    {
        // process 64-bit popcounts
        for (size_t i = 0; i < di_64b; i += 8) {
            const auto yv = *(const uint64_t*)(binary_data + i);
            sum_q += __builtin_popcountll(yv);
        }

        // process leftovers
        for (size_t i = di_64b; i < di_8b; i++) {
            const auto yv = *(binary_data + i);
            sum_q += __builtin_popcount(yv);
        }
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

void RaBitDistanceComputerQ::set_query(const float* x) {
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

    // allocate space
    rotated_qq.resize(d);

    // rotate the query
    std::vector<float> rotated_q(d);
    for (size_t i = 0; i < d; i++) {
        rotated_q[i] = x[i] - ((centroid == nullptr) ? 0 : centroid[i]);
    }

    // compute some numbers
    const float inv_d = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    // quantize the query. compute min and max
    float v_min = std::numeric_limits<float>::max();
    float v_max = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < d; i++) {
        const float v_q = rotated_q[i];
        v_min = std::min(v_min, v_q);
        v_max = std::max(v_max, v_q);
    }

    const float pow_2_qb = 1 << qb;

    const float delta = (v_max - v_min) / (pow_2_qb - 1);
    const float inv_delta = 1.0f / delta;

    size_t sum_qq = 0;
    for (int32_t i = 0; i < d; i++) {
        const float v_q = rotated_q[i];

        // a default non-randomized SQ
        const int v_qq = std::round((v_q - v_min) * inv_delta);

        rotated_qq[i] = std::min(255, std::max(0, v_qq));
        sum_qq += v_qq;
    }

    // rearrange the query vector
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

    query_fac.c1 = 2 * delta * inv_d;
    query_fac.c2 = 2 * v_min * inv_d;
    query_fac.c34 = inv_d * (delta * sum_qq + d * v_min);

    if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
        // precompute if needed
        query_fac.qr_norm_L2sqr = fvec_norm_L2sqr(x, d);
    }
}

FlatCodesDistanceComputer* RaBitQuantizer::get_distance_computer(
        uint8_t qb,
        const float* centroid_in) const {
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

        return dc.release();
    }
}

} // namespace faiss
