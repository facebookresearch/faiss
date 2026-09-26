/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQuantizer.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizerMultiBit.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_integer_adc.h>
#include <faiss/utils/rabitq_simd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace faiss {

RaBitQStats rabitq_stats;

void RaBitQStats::add_atomic(const RaBitQStats& other) {
    detail::atomic_fetch_add_relaxed(n_1bit, other.n_1bit);
    detail::atomic_fetch_add_relaxed(n_refine, other.n_refine);
}

// Import shared utilities from RaBitQUtils
using rabitq_utils::ExtraBitsFactors;
using rabitq_utils::ProgressiveBitsFactors;
using rabitq_utils::QueryFactorsData;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

namespace rabitq_integer_adc {

int64_t dot_product_scalar(
        const int8_t* query,
        const int8_t* levels,
        size_t d) {
    int64_t result = 0;
    for (size_t j = 0; j < d; j++) {
        result += int64_t(query[j]) * int64_t(levels[j]);
    }
    return result;
}

void dot_product_batch_4_scalar(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3) {
    dot0 = dot1 = dot2 = dot3 = 0;
    for (size_t j = 0; j < d; j++) {
        const int64_t q = query[j];
        dot0 += q * levels0[j];
        dot1 += q * levels1[j];
        dot2 += q * levels2[j];
        dot3 += q * levels3[j];
    }
}

void dot_product_batch_8_scalar(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t dots[8]) {
    for (size_t k = 0; k < 8; ++k) {
        dots[k] = 0;
    }
    for (size_t j = 0; j < d; ++j) {
        const int64_t q = query[j];
        for (size_t k = 0; k < 8; ++k) {
            dots[k] += q * levels[k][j];
        }
    }
}

void dot_product_batch_16_scalar(
        const int8_t* query,
        const int8_t* const levels[16],
        size_t d,
        int64_t dots[16]) {
    for (size_t k = 0; k < 16; ++k) {
        dots[k] = 0;
    }
    for (size_t j = 0; j < d; ++j) {
        const int64_t q = query[j];
        for (size_t k = 0; k < 16; ++k) {
            dots[k] += q * levels[k][j];
        }
    }
}

void dot_product_batch_tail_scalar(
        const int8_t* query,
        const int8_t* const* levels,
        int count,
        size_t d,
        int64_t* dots) {
    for (int k = 0; k < count; ++k) {
        dots[k] = 0;
    }
    for (size_t j = 0; j < d; ++j) {
        const int64_t q = query[j];
        for (int k = 0; k < count; ++k) {
            dots[k] += q * levels[k][j];
        }
    }
}

} // namespace rabitq_integer_adc

RaBitQuantizer::RaBitQuantizer(
        size_t d_in,
        MetricType metric,
        size_t nb_bits_in)
        : Quantizer(d_in, 0), // code_size will be set below
          metric_type{metric},
          nb_bits{nb_bits_in} {
    // Validate nb_bits range
    FAISS_THROW_IF_NOT(nb_bits >= 1 && nb_bits <= 9);

    // Set code_size using compute_code_size
    code_size = compute_code_size(d, nb_bits);
}

size_t RaBitQuantizer::compute_code_size(size_t d_in, size_t num_bits) const {
    // Validate inputs
    FAISS_THROW_IF_NOT(num_bits >= 1 && num_bits <= 9);

    size_t ex_bits = num_bits - 1;

    // Base: 1-bit codes + base factors
    // Layout for 1-bit: [binary_code: (d+7)/8 bytes][SignBitFactors: 8 bytes]
    //   base_factors = or_minus_c_l2sqr (4) + dp_multiplier (4)
    // Layout for multi-bit: [binary_code: (d+7)/8
    // bytes][SignBitFactorsWithError: 12 bytes]
    //   factors = or_minus_c_l2sqr (4) + dp_multiplier (4) + f_error (4)
    size_t base_size = (d_in + 7) / 8 +
            (ex_bits == 0 ? sizeof(SignBitFactors)
                          : sizeof(SignBitFactorsWithError));

    // Extra: ex-bit codes + ex factors (only if ex_bits > 0)
    // Layout: [ex_code: (d*ex_bits+7)/8 bytes][ex_factors: 8 bytes]
    size_t ex_size = 0;
    if (ex_bits > 0) {
        ex_size = (d_in * ex_bits + 7) / 8 + sizeof(ExtraBitsFactors);
    }

    return base_size + ex_size;
}

void RaBitQuantizer::train(size_t /*n*/, const float* /*x*/) {
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
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        // Pointer to this vector's code
        uint8_t* code = codes + i * code_size;

        // Clear code memory
        memset(code, 0, code_size);

        const float* x_row = x + i * d;

        // Pointer arithmetic for code layout:
        // For 1-bit: [binary_code: (d+7)/8 bytes][SignBitFactors: 8 bytes]
        // For multi-bit: [binary_code: (d+7)/8 bytes][SignBitFactorsWithError:
        // 12 bytes]
        //                [ex_code: (d*ex_bits+7)/8 bytes][ex_factors: 8 bytes]
        uint8_t* binary_code = code;

        // Step 1: Compute 1-bit quantization and base factors
        // Store residual for potential ex-bits quantization
        std::vector<float> residual(d);

        // Use shared utilities for computing factors
        SignBitFactorsWithError factors_data =
                rabitq_utils::compute_vector_factors(
                        x_row, d, centroid_in, metric_type, ex_bits > 0);

        // Write appropriate factors based on nb_bits
        if (ex_bits == 0) {
            // For 1-bit: write only SignBitFactors (8 bytes)
            SignBitFactors* base_factors =
                    reinterpret_cast<SignBitFactors*>(code + (d + 7) / 8);
            base_factors->or_minus_c_l2sqr = factors_data.or_minus_c_l2sqr;
            base_factors->dp_multiplier = factors_data.dp_multiplier;
        } else {
            // For multi-bit: write full SignBitFactorsWithError (12 bytes)
            SignBitFactorsWithError* full_factors =
                    reinterpret_cast<SignBitFactorsWithError*>(
                            code + (d + 7) / 8);
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
            uint8_t* ex_code =
                    code + (d + 7) / 8 + sizeof(SignBitFactorsWithError);
            // Pointer to ex-factors section
            ExtraBitsFactors* ex_factors = reinterpret_cast<ExtraBitsFactors*>(
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
    FAISS_THROW_IF_MSG(
            codes == nullptr, "RaBitQuantizer::decode_core: null codes buffer");
    FAISS_THROW_IF_MSG(
            x == nullptr, "RaBitQuantizer::decode_core: null output buffer");

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));
    const size_t ex_bits = nb_bits - 1;

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const uint8_t* code = codes + i * code_size;

        // split the code into parts
        const uint8_t* binary_data = code;

        // Cast to appropriate type based on nb_bits
        // For 1-bit: use SignBitFactors (8 bytes)
        // For multi-bit: use SignBitFactorsWithError (12 bytes, but only first
        // 8 bytes used for decode)
        const SignBitFactors* fac = (ex_bits == 0)
                ? reinterpret_cast<const SignBitFactors*>(code + (d + 7) / 8)
                : reinterpret_cast<const SignBitFactorsWithError*>(
                          code + (d + 7) / 8);

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

template <SIMDLevel SL>
float symmetric_dis_1bit(const RaBitQDistanceComputer& dc, idx_t i, idx_t j) {
    FAISS_THROW_IF_NOT_MSG(
            dc.metric_type == MetricType::METRIC_L2,
            "RaBitQ symmetric distance supports only L2");
    FAISS_ASSERT(i >= 0 && j >= 0);
    FAISS_ASSERT(dc.codes != nullptr);

    const size_t sign_bytes = (dc.d + 7) / 8;
    const uint8_t* code_i = dc.codes + static_cast<size_t>(i) * dc.code_size;
    const uint8_t* code_j = dc.codes + static_cast<size_t>(j) * dc.code_size;
    const auto* factors_i =
            reinterpret_cast<const SignBitFactors*>(code_i + sign_bytes);
    const auto* factors_j =
            reinterpret_cast<const SignBitFactors*>(code_j + sign_bytes);

    const uint64_t xor_popcount =
            rabitq::bitwise_xor_dot_product<SL>(code_i, code_j, sign_bytes, 1);
    const float sign_dot =
            static_cast<float>(dc.d) - 2.0f * static_cast<float>(xor_popcount);

    // The L2-optimal reconstruction of residual r is alpha * sign(r), where
    // alpha = ||r||_1 / d. The stored factors give
    // alpha_i * alpha_j = ||r_i||^2 * ||r_j||^2 /
    //     (d * dp_multiplier_i * dp_multiplier_j).
    float cross_term = 0.0f;
    if (factors_i->dp_multiplier != 0.0f && factors_j->dp_multiplier != 0.0f) {
        // Dividing each norm first avoids overflowing the product of two
        // squared norms even when the final distance is representable.
        const float scaled_norm_i =
                factors_i->or_minus_c_l2sqr / factors_i->dp_multiplier;
        const float scaled_norm_j =
                factors_j->or_minus_c_l2sqr / factors_j->dp_multiplier;
        cross_term = (scaled_norm_i * (sign_dot / static_cast<float>(dc.d))) *
                scaled_norm_j;
    }
    const float distance = factors_i->or_minus_c_l2sqr +
            factors_j->or_minus_c_l2sqr - 2.0f * cross_term;
    return std::max(0.0f, distance);
}

float RaBitQDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    return symmetric_dis_1bit<SIMDLevel::NONE>(*this, i, j);
}

namespace {

// Distance computers templatized on SIMDLevel to avoid per-call dynamic
// dispatch. The SIMDLevel is baked in at construction time via
// get_distance_computer, so virtual calls through the base class go
// directly to the SIMD-specialized code.

template <SIMDLevel SL>
struct RaBitQDistanceComputerNotQ final : RaBitQDistanceComputer {
    // the rotated query (qr - c)
    std::vector<float> rotated_q;
    // some additional numbers for the query
    QueryFactorsData query_fac;

    RaBitQDistanceComputerNotQ() = default;

    float symmetric_dis(idx_t i, idx_t j) final {
        return symmetric_dis_1bit<SL>(*this, i, j);
    }

    // Compute distance using only 1-bit codes (fast)
    float distance_to_code_1bit_impl(
            const uint8_t* binary_data,
            const SignBitFactors* base_fac) const {
        // this is the baseline code
        //
        // compute <q,o> using floats
        float dot_qo = 0;
        // It was a willful decision (after the discussion) to not to pre-cache
        //   the sum of all bits, just in order to reduce the overhead per
        //   vector.
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
            return std::max(0.0f, pre_dist);
        } else {
            // metric == MetricType::METRIC_INNER_PRODUCT
            // 2 * (or, q) = (||or - q||^2 - ||q||^2 - ||or||^2)
            return -0.5f * (pre_dist - query_fac.qr_norm_L2sqr);
        }
    }

    float distance_to_code_1bit(const uint8_t* code) final {
        FAISS_ASSERT(code != nullptr);
        FAISS_ASSERT(
                (metric_type == MetricType::METRIC_L2 ||
                 metric_type == MetricType::METRIC_INNER_PRODUCT));
        FAISS_ASSERT(rotated_q.size() == d);

        const size_t code_size_base = (d + 7) / 8;
        const size_t ex_bits = nb_bits - 1;
        const SignBitFactors* base_fac = (ex_bits == 0)
                ? reinterpret_cast<const SignBitFactors*>(code + code_size_base)
                : reinterpret_cast<const SignBitFactorsWithError*>(
                          code + code_size_base);
        return distance_to_code_1bit_impl(code, base_fac);
    }

    // Compute full distance using 1-bit + ex-bits (accurate)
    float distance_to_code_full(const uint8_t* code) final {
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
        size_t offset = (d + 7) / 8 + sizeof(SignBitFactorsWithError);
        const uint8_t* ex_code = code + offset;
        const ExtraBitsFactors* ex_fac =
                reinterpret_cast<const ExtraBitsFactors*>(
                        ex_code + (d * ex_bits + 7) / 8);

        float qr_base = (metric_type == MetricType::METRIC_INNER_PRODUCT)
                ? query_fac.q_dot_c
                : query_fac.qr_to_c_L2sqr;
        return rabitq_utils::compute_full_multibit_distance<SL>(
                binary_data,
                ex_code,
                *ex_fac,
                rotated_q.data(),
                qr_base,
                d,
                ex_bits,
                metric_type);
    }

    void set_query(const float* x) final {
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

        // Compute g_error = ||qr - c|| (L2 norm of rotated query)
        g_error = std::sqrt(query_fac.qr_to_c_L2sqr);

        // compute some numbers — do not quantize the query
        const float inv_d = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

        float sum_q = 0;
        for (size_t i = 0; i < d; i++) {
            sum_q += rotated_q[i];
        }

        query_fac.c1 = 2 * inv_d;
        query_fac.c2 = 0;
        query_fac.c34 = sum_q * inv_d;

        if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
            query_fac.qr_norm_L2sqr = fvec_norm_L2sqr(x, d);
            query_fac.q_dot_c =
                    centroid ? fvec_inner_product(x, centroid, d) : 0.0f;
        }
    }

    size_t scan_codes_multibit(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            size_t code_size,
            idx_t list_no,
            bool store_pairs,
            const IDSelector* sel,
            bool keep_max,
            ResultHandler& handler) final {
        const size_t code_size_base = (d + 7) / 8;
        const size_t ex_bits = nb_bits - 1;
        FAISS_ASSERT(ex_bits > 0);

        // Honor IDSelectorWithContext on the multibit path too, so a RaBitQ
        // index does not silently lose the context hook once nb_bits >= 2 (the
        // 1-bit path already routes through run_scan_codes1).
        const IDSelectorContextDispatch sel_dispatch(sel, store_pairs);

        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            if (sel != nullptr) {
                idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                if (!sel_dispatch.is_member(
                            id, IDScanContext{ids, list_size, j})) {
                    codes += code_size;
                    continue;
                }
            }

            const auto* base_fac =
                    reinterpret_cast<const SignBitFactorsWithError*>(
                            codes + code_size_base);
            const float est_distance =
                    distance_to_code_1bit_impl(codes, base_fac);

            const bool should_refine = rabitq_utils::should_refine_candidate(
                    est_distance,
                    base_fac->f_error,
                    g_error,
                    handler.threshold,
                    keep_max);
            if (should_refine) {
                handler.stats.scan_cnt++;
                const float dis = distance_to_code_full(codes);
                idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];

                if (handler.add_result(dis, id)) {
                    handler.stats.nheap_updates++;
                    nup++;
                }
            }
            codes += code_size;
        }

        return nup;
    }
};

template <SIMDLevel SL>
struct RaBitQDistanceComputerQ final : RaBitQDistanceComputer,
                                       DistanceComputerBatch {
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

    RaBitQDistanceComputerQ() = default;

    float symmetric_dis(idx_t i, idx_t j) final {
        return symmetric_dis_1bit<SL>(*this, i, j);
    }

    // Compute distance using only 1-bit codes (fast)
    float distance_to_code_1bit_impl(
            const uint8_t* binary_data,
            const SignBitFactors* base_fac,
            size_t size) const {
        // this is ||or - c||^2 - (IP ? ||or||^2 : 0)
        float final_dot = 0;
        if (centered) {
            int64_t int_dot = ((1 << qb) - 1) * d;
            // See RaBitDistanceComputerNotQ::distance_to_code() for
            // baseline code.
            int_dot -= 2 *
                    rabitq::bitwise_xor_dot_product<SL>(
                               rearranged_rotated_qq.data(),
                               binary_data,
                               size,
                               qb);
            final_dot += int_dot * query_fac.int_dot_scale;
        } else {
            auto bitwise_result =
                    rabitq::bitwise_and_dot_product_with_popcount<SL>(
                            rearranged_rotated_qq.data(),
                            binary_data,
                            size,
                            qb);
            // dot-product itself
            final_dot += query_fac.c1 * bitwise_result.dot_product;
            // normalizer coefficients
            final_dot += query_fac.c2 * bitwise_result.popcount;
            // normalizer coefficients
            final_dot -= query_fac.c34;
        }

        const float pre_dist = base_fac->or_minus_c_l2sqr +
                query_fac.qr_to_c_L2sqr -
                2 * base_fac->dp_multiplier * final_dot;

        if (metric_type == MetricType::METRIC_L2) {
            // ||or - q||^ 2
            return std::max(0.0f, pre_dist);
        } else {
            // metric == MetricType::METRIC_INNER_PRODUCT
            // 2 * (or, q) = (||or - q||^2 - ||q||^2 - ||or||^2)
            return -0.5f * (pre_dist - query_fac.qr_norm_L2sqr);
        }
    }

    float distance_to_code_1bit(const uint8_t* code) final {
        FAISS_ASSERT(code != nullptr);
        FAISS_ASSERT(
                (metric_type == MetricType::METRIC_L2 ||
                 metric_type == MetricType::METRIC_INNER_PRODUCT));

        const size_t size = (d + 7) / 8;
        const size_t ex_bits = nb_bits - 1;
        const SignBitFactors* base_fac = (ex_bits == 0)
                ? reinterpret_cast<const SignBitFactors*>(code + size)
                : reinterpret_cast<const SignBitFactorsWithError*>(code + size);
        return distance_to_code_1bit_impl(code, base_fac, size);
    }

    int preferred_batch_size() const final {
        return d >= 256 ? 16 : 8;
    }

    int max_tail_batch_size() const final {
        return 7;
    }

    void distances_batch_8(const int32_t* ids, float* distances) final {
        for (int lane = 0; lane < 8; ++lane) {
            const uint8_t* code =
                    codes + static_cast<size_t>(ids[lane]) * code_size;
            distances[lane] = distance_to_code(code);
        }
    }

    void distances_batch_16(const int32_t* ids, float* distances) final {
        for (int lane = 0; lane < 16; ++lane) {
            const uint8_t* code =
                    codes + static_cast<size_t>(ids[lane]) * code_size;
            distances[lane] = distance_to_code(code);
        }
    }

    void distances_batch_tail(const int32_t* ids, int count, float* distances)
            final {
        FAISS_THROW_IF_NOT_MSG(
                count >= 1 && count <= 7,
                "RaBitQ tail batch size must be between 1 and 7");
        for (int lane = 0; lane < count; ++lane) {
            const uint8_t* code =
                    codes + static_cast<size_t>(ids[lane]) * code_size;
            distances[lane] = distance_to_code(code);
        }
    }

    // Compute full distance using 1-bit + ex-bits (accurate)
    float distance_to_code_full(const uint8_t* code) final {
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
        size_t offset = (d + 7) / 8 + sizeof(SignBitFactorsWithError);
        const uint8_t* ex_code = code + offset;
        const ExtraBitsFactors* ex_fac =
                reinterpret_cast<const ExtraBitsFactors*>(
                        ex_code + (d * ex_bits + 7) / 8);

        float qr_base = (metric_type == MetricType::METRIC_INNER_PRODUCT)
                ? query_fac.q_dot_c
                : query_fac.qr_to_c_L2sqr;
        return rabitq_utils::compute_full_multibit_distance<SL>(
                binary_data,
                ex_code,
                *ex_fac,
                rotated_q.data(),
                qr_base,
                d,
                ex_bits,
                metric_type);
    }

    void set_query(const float* x) final {
        q = x;
        FAISS_ASSERT(x != nullptr);
        FAISS_ASSERT(
                (metric_type == MetricType::METRIC_L2 ||
                 metric_type == MetricType::METRIC_INNER_PRODUCT));
        FAISS_THROW_IF_NOT(qb <= 8);
        FAISS_THROW_IF_NOT(qb > 0);

        // Use shared utilities for core query factor computation
        // rotated_q is populated directly by compute_query_factors as an
        // output parameter
        query_fac = rabitq_utils::compute_query_factors(
                x,
                d,
                centroid,
                qb,
                centered,
                metric_type,
                rotated_q,
                rotated_qq);

        // Compute g_error (query norm for lower bound computation)
        // g_error = ||qr - c|| (L2 norm of rotated query)
        g_error = std::sqrt(query_fac.qr_to_c_L2sqr);

        // Rearrange the query vector for SIMD operations
        // (RaBitQuantizer-specific)
        popcount_aligned_dim = ((d + 7) / 8) * 8;
        size_t offset = (d + 7) / 8;

        rearranged_rotated_qq.resize(offset * qb);
        with_selected_simd_levels<
                AVAILABLE_SIMD_LEVELS_NONE | (1 << int(SIMDLevel::AVX2)) |
                (1 << int(SIMDLevel::AVX512))>([&]<SIMDLevel RSL>() {
            rabitq::rearrange_bit_planes<RSL>(
                    rotated_qq.data(), d, qb, rearranged_rotated_qq.data());
        });
    }

    size_t scan_codes_multibit(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            size_t code_size,
            idx_t list_no,
            bool store_pairs,
            const IDSelector* sel,
            bool keep_max,
            ResultHandler& handler) final {
        const size_t code_size_base = (d + 7) / 8;
        const size_t ex_bits = nb_bits - 1;
        FAISS_ASSERT(ex_bits > 0);

        // Honor IDSelectorWithContext on the multibit path too, so a RaBitQ
        // index does not silently lose the context hook once nb_bits >= 2 (the
        // 1-bit path already routes through run_scan_codes1).
        const IDSelectorContextDispatch sel_dispatch(sel, store_pairs);

        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            if (sel != nullptr) {
                idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                if (!sel_dispatch.is_member(
                            id, IDScanContext{ids, list_size, j})) {
                    codes += code_size;
                    continue;
                }
            }

            const auto* base_fac =
                    reinterpret_cast<const SignBitFactorsWithError*>(
                            codes + code_size_base);
            const float est_distance =
                    distance_to_code_1bit_impl(codes, base_fac, code_size_base);

            const bool should_refine = rabitq_utils::should_refine_candidate(
                    est_distance,
                    base_fac->f_error,
                    g_error,
                    handler.threshold,
                    keep_max);
            if (should_refine) {
                handler.stats.scan_cnt++;
                const float dis = distance_to_code_full(codes);
                idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];

                if (handler.add_result(dis, id)) {
                    handler.stats.nheap_updates++;
                    nup++;
                }
            }
            codes += code_size;
        }

        return nup;
    }
};

/** Full-code scorer over byte-expanded existing RaBitQ levels.
 *
 * This deliberately does not implement RaBitQDistanceComputer's staged
 * interface: integer query ADC changes the full score and has no proven bound
 * for the one-bit pruning stage. Callers must use the ordinary full-code path.
 */
struct RaBitQExpandedDistanceComputer final : FlatCodesDistanceComputer,
                                              DistanceComputerBatch,
                                              DistanceComputerAdaptive {
    size_t d = 0;
    const float* centroid = nullptr;
    bool integer_query = false;
    std::vector<float> residual;
    std::vector<int8_t> quantized;
    float query_norm = 0.0f;
    float half_sum = 0.0f;
    float scale = 1.0f;
    float quantized_norm = 0.0f;
    bool use_arm_dotprod = false;
    bool use_arm_packed2 = false;
    bool use_arm_packed4 = false;
    bool use_arm_split4 = false;
    bool use_arm_progressive = false;
    bool use_arm_nested_lut7 = false;
    bool use_arm_nested_lut4 = false;
    bool use_avx512_vnni = false;
    int64_t query_correction = 0;
    bool packed_storage = false;
    bool split4_storage = false;
    bool progressive_storage = false;
    bool progressive_prefix_only = false;
    bool nested_lut7_storage = false;
    bool nested_lut4_storage = false;
    bool nested_lut4_nibble = false;
    bool nested_lut3_navigation = false;
    bool nested_exact7_storage = false;
    size_t ex_bits = 0;
    size_t sign_bytes = 0;
    size_t ex_offset = 0;
    size_t ex_bytes = 0;
    size_t prefix_bytes = 0;
    size_t tail_bits = 0;
    size_t tail_offset = 0;
    size_t tail_bytes = 0;
    size_t factors_offset = 0;
    size_t nested_high1_offset = 0;
    const uint8_t* nested_lut7 = nullptr;
    const uint8_t* nested_lut4 = nullptr;
    const float* adaptive_error_norms = nullptr;
    float adaptive_sigma = 0.0f;
    uint64_t adaptive_prefix_counter = 0;
    uint64_t adaptive_refine_counter = 0;
    bool adaptive_full_fallback = false;
    const uint8_t* split4_tail_codes = nullptr;
    size_t split4_tail_stride = 0;
    std::vector<int8_t> decoded_levels;

    RaBitQExpandedDistanceComputer(
            const uint8_t* codes,
            size_t d_in,
            const float* centroid_in,
            bool integer_query_in,
            size_t stored_code_size_in = 0,
            size_t nb_bits_in = 0,
            bool progressive_storage_in = false,
            bool progressive_prefix_only_in = false,
            const uint8_t* split4_tail_codes_in = nullptr,
            size_t split4_tail_stride_in = 0,
            bool nested_lut7_storage_in = false,
            const uint8_t* nested_lut7_in = nullptr,
            bool nested_lut3_navigation_in = false,
            const float* adaptive_error_norms_in = nullptr,
            float adaptive_sigma_in = 0.0f,
            bool nested_lut4_storage_in = false,
            bool nested_lut4_nibble_in = false,
            const uint8_t* nested_lut4_in = nullptr)
            : FlatCodesDistanceComputer(
                      codes,
                      stored_code_size_in ? stored_code_size_in
                                          : d_in + sizeof(ExtraBitsFactors)),
              d(d_in),
              centroid(centroid_in),
              integer_query(integer_query_in),
              residual(d),
              quantized(d),
              packed_storage(
                      stored_code_size_in != 0 && !progressive_storage_in &&
                      split4_tail_codes_in == nullptr &&
                      !nested_lut7_storage_in && !nested_lut4_storage_in),
              split4_storage(split4_tail_codes_in != nullptr),
              progressive_storage(progressive_storage_in),
              progressive_prefix_only(progressive_prefix_only_in),
              nested_lut7_storage(nested_lut7_storage_in),
              nested_lut4_storage(nested_lut4_storage_in),
              nested_lut4_nibble(nested_lut4_nibble_in),
              nested_lut3_navigation(nested_lut3_navigation_in),
              ex_bits((packed_storage || progressive_storage ||
                       split4_storage || nested_lut4_storage)
                              ? nb_bits_in - 1
                              : 0),
              sign_bytes((packed_storage || split4_storage) ? (d + 7) / 8 : 0),
              ex_offset(
                      packed_storage
                              ? sign_bytes + sizeof(SignBitFactorsWithError)
                              : 0),
              ex_bytes(
                      split4_storage
                              ? ex_bits * ((d + 7) / 8)
                              : (packed_storage ? (d * ex_bits + 7) / 8 : 0)),
              prefix_bytes(progressive_storage ? (2 * d + 7) / 8 : 0),
              tail_bits(
                      progressive_storage
                              ? (nested_lut4_storage ? 2 : nb_bits_in - 1)
                              : 0),
              tail_offset(
                      progressive_storage
                              ? prefix_bytes + sizeof(ProgressiveBitsFactors)
                              : 0),
              tail_bytes(progressive_storage ? (d * tail_bits + 7) / 8 : 0),
              factors_offset(progressive_storage ? prefix_bytes : 0),
              nested_high1_offset(
                      nested_lut7_storage
                              ? prefix_bytes + sizeof(ProgressiveBitsFactors) +
                                      (4 * d + 7) / 8
                              : 0),
              nested_lut7(nested_lut7_in),
              nested_lut4(nested_lut4_in),
              adaptive_error_norms(adaptive_error_norms_in),
              adaptive_sigma(adaptive_sigma_in),
              split4_tail_codes(split4_tail_codes_in),
              split4_tail_stride(split4_tail_stride_in),
              decoded_levels(
                      (packed_storage || progressive_storage ||
                       split4_storage || nested_lut4_storage)
                              ? 16 * d
                              : 0) {
        FAISS_THROW_IF_NOT_MSG(
                !(packed_storage || progressive_storage || split4_storage ||
                  nested_lut4_storage) ||
                        (integer_query && nb_bits_in >= 2 && nb_bits_in <= 8),
                "packed integer ADC requires 2..8 total bits");
        FAISS_THROW_IF_NOT_MSG(
                !progressive_storage || nb_bits_in >= 3,
                "progressive ADC requires at least 3 total bits");
        FAISS_THROW_IF_NOT_MSG(
                !nested_lut7_storage ||
                        (progressive_storage && nb_bits_in == 7 &&
                         nested_lut7 != nullptr),
                "nested LUT ADC requires a 64-entry seven-bit codebook");
        FAISS_THROW_IF_NOT_MSG(
                !nested_lut4_storage ||
                        (nb_bits_in == 4 && nested_lut4 != nullptr &&
                         (nested_lut4_nibble || progressive_storage)),
                "nested LUT4 ADC requires a four-bit codebook and layout");
        FAISS_THROW_IF_NOT_MSG(
                adaptive_error_norms == nullptr ||
                        (nested_lut7_storage && adaptive_sigma > 0.0f),
                "adaptive scoring requires nested LUT storage and sigma > 0");
        if (nested_lut7_storage) {
            nested_exact7_storage = true;
            for (size_t i = 0; i < 64; ++i) {
                nested_exact7_storage &= nested_lut7[i] == i;
            }
        }
#ifdef COMPILE_SIMD_ARM_NEON
        // SVE falls back to its NEON implementation at this call site, while
        // an explicitly selected NONE level must stay on the portable path.
        const bool selected_arm_neon =
                with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_AVX2_NEON>(
                        []<SIMDLevel SL>() {
                            return SL == SIMDLevel::ARM_NEON;
                        });
        use_arm_dotprod = integer_query && selected_arm_neon &&
                rabitq_integer_adc::arm_dotprod_supported();
        use_arm_packed2 = use_arm_dotprod && packed_storage && ex_bits == 1;
        use_arm_packed4 = use_arm_dotprod && packed_storage && ex_bits == 3;
        use_arm_split4 = use_arm_dotprod && split4_storage;
        use_arm_progressive = use_arm_dotprod && progressive_storage &&
                !nested_lut7_storage &&
                (progressive_prefix_only || ex_bits == 5 || ex_bits == 6);
        use_arm_nested_lut7 = use_arm_dotprod && nested_lut7_storage;
        use_arm_nested_lut4 = use_arm_dotprod && nested_lut4_storage;
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
        // Opt this call site into the SPR level through the regular dispatch
        // machinery. In a DD build, forcing AVX512_VPOPCNT, AVX512, AVX2, or
        // NONE therefore falls through to the portable decode path without
        // ever entering a VNNI kernel. In a static SPR build this resolves at
        // compile time.
        const bool selected_spr =
                with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_BASE_WITH_SPR>(
                        []<SIMDLevel SL>() {
                            return SL == SIMDLevel::AVX512_SPR;
                        });
        use_avx512_vnni = integer_query && selected_spr;
#endif
    }

#ifdef COMPILE_SIMD_AVX512_SPR
    bool use_avx512_prefix_direct() const {
        return use_avx512_vnni && progressive_storage &&
                progressive_prefix_only;
    }

    bool use_avx512_nested_direct() const {
        return use_avx512_vnni && !progressive_prefix_only &&
                !nested_lut3_navigation &&
                (nested_lut4_storage || nested_lut7_storage);
    }

    void dot_product_prefix_avx512(
            const uint8_t* const* code_rows,
            int count,
            int64_t* dots) const {
        FAISS_ASSERT(use_avx512_prefix_direct());
        rabitq_integer_adc::dot_product_progressive_prefix_batch_avx512_vnni(
                quantized.data(), code_rows, count, d, query_correction, dots);
    }

    void dot_product_nested_avx512(
            const uint8_t* const* code_rows,
            int count,
            int64_t* dots) const {
        FAISS_ASSERT(use_avx512_nested_direct());
        if (nested_lut4_storage) {
            rabitq_integer_adc::dot_product_nested_lut4_batch_avx512_vnni(
                    quantized.data(),
                    code_rows,
                    count,
                    d,
                    nested_lut4_nibble ? 0 : tail_offset,
                    nested_lut4,
                    nested_lut4_nibble,
                    query_correction,
                    dots);
        } else {
            rabitq_integer_adc::dot_product_nested_lut7_batch_avx512_vnni(
                    quantized.data(),
                    code_rows,
                    count,
                    d,
                    tail_offset,
                    nested_high1_offset,
                    nested_lut7,
                    query_correction,
                    dots);
        }
    }
#endif

    const uint8_t* split4_tail_for_code(const uint8_t* code) const {
        FAISS_ASSERT(split4_storage && code >= codes);
        const size_t row = static_cast<size_t>(code - codes) / code_size;
        return split4_tail_codes + row * split4_tail_stride;
    }

    const int8_t* levels_for_code(const uint8_t* code, size_t slot) {
        if (!packed_storage && !progressive_storage && !split4_storage &&
            !nested_lut4_storage) {
            return reinterpret_cast<const int8_t*>(code);
        }
        FAISS_ASSERT(slot < 16);
        int8_t* output = decoded_levels.data() + slot * d;
        if (nested_lut4_storage && nested_lut4_nibble) {
            for (size_t j = 0; j < d; ++j) {
                const uint8_t symbol = (code[j >> 1] >> (4 * (j & 1))) & 15;
                const int positive = symbol >> 3;
                const int magnitude = nested_lut4[symbol & 7];
                output[j] = static_cast<int8_t>(
                        positive ? magnitude : -1 - magnitude);
            }
            return output;
        }
        if (progressive_storage) {
            const uint8_t* tail = code + tail_offset;
            const int midpoint = 1 << ex_bits;
            for (size_t j = 0; j < d; ++j) {
                const int prefix =
                        rabitq_utils::extract_code_inline(code, j, 2);
                if (progressive_prefix_only) {
                    output[j] = static_cast<int8_t>(prefix - 2);
                } else if (nested_lut4_storage) {
                    const int positive = prefix >> 1;
                    const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
                    const int local =
                            rabitq_utils::extract_code_inline(tail, j, 2);
                    const int magnitude = nested_lut4[coarse * 4 + local];
                    output[j] = static_cast<int8_t>(
                            positive ? magnitude : -1 - magnitude);
                } else if (nested_lut7_storage) {
                    const int positive = prefix >> 1;
                    const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
                    const int local_high = rabitq_utils::extract_bit_standard(
                            code + nested_high1_offset, j);
                    const int local_low = nested_lut3_navigation
                            ? 7
                            : rabitq_utils::extract_code_inline(tail, j, 4);
                    const int local = local_low | (local_high << 4);
                    const int magnitude = nested_lut7[coarse * 32 + local];
                    output[j] = static_cast<int8_t>(
                            positive ? magnitude : -1 - magnitude);
                } else {
                    const int low = rabitq_utils::extract_code_inline(
                            tail, j, tail_bits);
                    const int sign = prefix >> 1;
                    output[j] = static_cast<int8_t>(
                            (sign << tail_bits) + low - midpoint);
                }
            }
            return output;
        }
        const uint8_t* extra =
                split4_storage ? split4_tail_for_code(code) : code + ex_offset;
        const int midpoint = 1 << ex_bits;
        const uint32_t low_mask = uint32_t(midpoint - 1);
        uint32_t bit_buffer = 0;
        unsigned buffered_bits = 0;
        for (size_t j = 0; j < d; ++j) {
            int low;
            if (split4_storage) {
                low = 0;
                for (size_t bit = 0; bit < ex_bits; ++bit) {
                    low |= int(rabitq_utils::extract_bit_standard(
                                   extra + bit * sign_bytes, j))
                            << bit;
                }
            } else {
                while (buffered_bits < ex_bits) {
                    bit_buffer |= uint32_t(*extra++) << buffered_bits;
                    buffered_bits += 8;
                }
                low = int(bit_buffer & low_mask);
                bit_buffer >>= ex_bits;
                buffered_bits -= ex_bits;
            }
            const int sign = (code[j >> 3] >> (j & 7)) & 1;
            output[j] = static_cast<int8_t>((sign << ex_bits) + low - midpoint);
        }
        return output;
    }

    const uint8_t* factors_for_code(const uint8_t* code) const {
        if (nested_lut4_storage && nested_lut4_nibble) {
            return code + (4 * d + 7) / 8;
        }
        if (progressive_storage) {
            return code + factors_offset;
        }
        return packed_storage ? code + ex_offset + ex_bytes : code + d;
    }

    void set_query(const float* x) final {
        q = x;
        FAISS_THROW_IF_NOT_MSG(x != nullptr, "null RaBitQ query");
        query_norm = 0.0f;
        float sum = 0.0f;
        float maximum = 0.0f;
        for (size_t j = 0; j < d; j++) {
            const float value = x[j] - (centroid ? centroid[j] : 0.0f);
            residual[j] = value;
            query_norm += value * value;
            sum += value;
            maximum = std::max(maximum, std::abs(value));
        }
        half_sum = 0.5f * sum;
        scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        if (integer_query) {
            int64_t quantized_norm_sqr = 0;
#ifdef COMPILE_SIMD_AVX512_SPR
            int64_t quantized_sum = 0;
#endif
            for (size_t j = 0; j < d; j++) {
                quantized[j] = static_cast<int8_t>(std::clamp(
                        std::nearbyint(residual[j] / scale), -127.0f, 127.0f));
                quantized_norm_sqr += int64_t(quantized[j]) * quantized[j];
#ifdef COMPILE_SIMD_AVX512_SPR
                quantized_sum += quantized[j];
#endif
            }
#ifdef COMPILE_SIMD_AVX512_SPR
            query_correction = -128 * quantized_sum;
#endif
            quantized_norm = std::sqrt(static_cast<float>(quantized_norm_sqr));
        }
    }

    float score(const uint8_t* code, float dot) const {
        if (split4_storage) {
            ExtraBitsFactors factors;
            memcpy(&factors,
                   split4_tail_for_code(code) + ex_bytes,
                   sizeof(factors));
            return std::max(
                    0.0f,
                    query_norm + factors.f_add_ex +
                            factors.f_rescale_ex * (dot + half_sum));
        }
        if (progressive_storage) {
            ProgressiveBitsFactors factors;
            memcpy(&factors, factors_for_code(code), sizeof(factors));
            const float rescale =
                    (progressive_prefix_only || nested_lut3_navigation)
                    ? factors.f_rescale_prefix
                    : factors.f_rescale_full;
            return std::max(
                    0.0f,
                    query_norm + factors.f_add + rescale * (dot + half_sum));
        }
        ExtraBitsFactors factors;
        memcpy(&factors, factors_for_code(code), sizeof(factors));
        return std::max(
                0.0f,
                query_norm + factors.f_add_ex +
                        factors.f_rescale_ex * (dot + half_sum));
    }

    float distance_to_code(const uint8_t* code) final {
        if (integer_query) {
            int64_t dot;
#ifdef COMPILE_SIMD_ARM_NEON
            if (use_arm_nested_lut4) {
                dot = progressive_prefix_only
                        ? rabitq_integer_adc::
                                  dot_product_progressive_prefix_arm(
                                          quantized.data(), code, d)
                        : rabitq_integer_adc::dot_product_nested_lut4_arm(
                                  quantized.data(),
                                  code,
                                  d,
                                  nested_lut4_nibble ? 0 : tail_offset,
                                  nested_lut4,
                                  nested_lut4_nibble);
                return score(code, scale * static_cast<float>(dot));
            }
            if (use_arm_nested_lut7) {
                if (progressive_prefix_only) {
                    dot = rabitq_integer_adc::
                            dot_product_progressive_prefix_arm(
                                    quantized.data(), code, d);
                } else if (nested_lut3_navigation) {
                    dot = rabitq_integer_adc::dot_product_nested_lut3_arm(
                            quantized.data(),
                            code,
                            d,
                            nested_high1_offset,
                            nested_lut7);
                } else if (nested_exact7_storage) {
                    dot = rabitq_integer_adc::dot_product_nested_exact7_arm(
                            quantized.data(),
                            code,
                            d,
                            tail_offset,
                            nested_high1_offset);
                } else {
                    dot = rabitq_integer_adc::dot_product_nested_lut7_arm(
                            quantized.data(),
                            code,
                            d,
                            tail_offset,
                            nested_high1_offset,
                            nested_lut7);
                }
                return score(code, scale * static_cast<float>(dot));
            }
            if (use_arm_progressive) {
                if (progressive_prefix_only) {
                    dot = rabitq_integer_adc::
                            dot_product_progressive_prefix_arm(
                                    quantized.data(), code, d);
                } else if (tail_bits == 5) {
                    dot = rabitq_integer_adc::dot_product_progressive_full6_arm(
                            quantized.data(), code, d, tail_offset);
                } else {
                    dot = rabitq_integer_adc::dot_product_progressive_full7_arm(
                            quantized.data(), code, d, tail_offset);
                }
                return score(code, scale * static_cast<float>(dot));
            }
            if (use_arm_packed2) {
                dot = rabitq_integer_adc::dot_product_packed_2bit_arm(
                        quantized.data(), code, d, ex_offset);
                return score(code, scale * static_cast<float>(dot));
            }
            if (use_arm_packed4) {
                dot = rabitq_integer_adc::dot_product_packed_4bit_arm(
                        quantized.data(), code, d, ex_offset);
                return score(code, scale * static_cast<float>(dot));
            }
            if (use_arm_split4) {
                const uint8_t* signs[1] = {code};
                const uint8_t* tails[1] = {split4_tail_for_code(code)};
                rabitq_integer_adc::dot_product_split_4bit_batch_arm(
                        quantized.data(), signs, tails, 1, d, &dot);
                return score(code, scale * static_cast<float>(dot));
            }
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
            if (use_avx512_prefix_direct()) {
                const uint8_t* code_rows[1] = {code};
                dot_product_prefix_avx512(code_rows, 1, &dot);
                return score(code, scale * static_cast<float>(dot));
            }
            if (use_avx512_nested_direct()) {
                const uint8_t* code_rows[1] = {code};
                dot_product_nested_avx512(code_rows, 1, &dot);
                return score(code, scale * static_cast<float>(dot));
            }
#endif
            const int8_t* levels = levels_for_code(code, 0);
#ifdef COMPILE_SIMD_AVX512_SPR
            if (use_avx512_vnni) {
                dot = rabitq_integer_adc::dot_product_avx512_vnni(
                        quantized.data(), levels, d, query_correction);
            } else
#endif
#ifdef COMPILE_SIMD_ARM_NEON
                    if (use_arm_dotprod) {
                dot = rabitq_integer_adc::dot_product_arm(
                        quantized.data(), levels, d);
            } else
#endif
            {
                dot = rabitq_integer_adc::dot_product_scalar(
                        quantized.data(), levels, d);
            }
            return score(code, scale * static_cast<float>(dot));
        }

        const int8_t* levels = levels_for_code(code, 0);
        float dot = 0.0f;
        for (size_t j = 0; j < d; j++) {
            dot += residual[j] * static_cast<float>(levels[j]);
        }
        return score(code, dot);
    }

    void distance_to_code_batch_4(
            const uint8_t* code0,
            const uint8_t* code1,
            const uint8_t* code2,
            const uint8_t* code3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final {
        if (!integer_query) {
            dis0 = distance_to_code(code0);
            dis1 = distance_to_code(code1);
            dis2 = distance_to_code(code2);
            dis3 = distance_to_code(code3);
            return;
        }

        int64_t dot0, dot1, dot2, dot3;
#ifdef COMPILE_SIMD_ARM_NEON
        if (use_arm_nested_lut4) {
            const uint8_t* code_rows[4] = {code0, code1, code2, code3};
            int64_t dots[4];
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 4, d, dots);
            } else {
                rabitq_integer_adc::dot_product_nested_lut4_batch_arm(
                        quantized.data(),
                        code_rows,
                        4,
                        d,
                        nested_lut4_nibble ? 0 : tail_offset,
                        nested_lut4,
                        nested_lut4_nibble,
                        dots);
            }
            dis0 = score(code0, scale * static_cast<float>(dots[0]));
            dis1 = score(code1, scale * static_cast<float>(dots[1]));
            dis2 = score(code2, scale * static_cast<float>(dots[2]));
            dis3 = score(code3, scale * static_cast<float>(dots[3]));
            return;
        }
        if (use_arm_nested_lut7) {
            const uint8_t* code_rows[4] = {code0, code1, code2, code3};
            int64_t dots[4];
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 4, d, dots);
            } else if (nested_lut3_navigation) {
                rabitq_integer_adc::dot_product_nested_lut3_batch_arm(
                        quantized.data(),
                        code_rows,
                        4,
                        d,
                        nested_high1_offset,
                        nested_lut7,
                        dots);
            } else if (nested_exact7_storage) {
                rabitq_integer_adc::dot_product_nested_exact7_batch_arm(
                        quantized.data(),
                        code_rows,
                        4,
                        d,
                        tail_offset,
                        nested_high1_offset,
                        dots);
            } else {
                rabitq_integer_adc::dot_product_nested_lut7_batch_arm(
                        quantized.data(),
                        code_rows,
                        4,
                        d,
                        tail_offset,
                        nested_high1_offset,
                        nested_lut7,
                        dots);
            }
            dis0 = score(code0, scale * static_cast<float>(dots[0]));
            dis1 = score(code1, scale * static_cast<float>(dots[1]));
            dis2 = score(code2, scale * static_cast<float>(dots[2]));
            dis3 = score(code3, scale * static_cast<float>(dots[3]));
            return;
        }
        if (use_arm_progressive) {
            const uint8_t* code_rows[4] = {code0, code1, code2, code3};
            int64_t dots[4];
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 4, d, dots);
            } else if (tail_bits == 5) {
                rabitq_integer_adc::dot_product_progressive_full6_batch_arm(
                        quantized.data(), code_rows, 4, d, tail_offset, dots);
            } else {
                rabitq_integer_adc::dot_product_progressive_full7_batch_arm(
                        quantized.data(), code_rows, 4, d, tail_offset, dots);
            }
            dis0 = score(code0, scale * static_cast<float>(dots[0]));
            dis1 = score(code1, scale * static_cast<float>(dots[1]));
            dis2 = score(code2, scale * static_cast<float>(dots[2]));
            dis3 = score(code3, scale * static_cast<float>(dots[3]));
            return;
        }
        if (use_arm_packed2) {
            const uint8_t* code_rows[4] = {code0, code1, code2, code3};
            int64_t dots[4];
            rabitq_integer_adc::dot_product_packed_2bit_batch_4_arm(
                    quantized.data(), code_rows, d, ex_offset, dots);
            dis0 = score(code0, scale * static_cast<float>(dots[0]));
            dis1 = score(code1, scale * static_cast<float>(dots[1]));
            dis2 = score(code2, scale * static_cast<float>(dots[2]));
            dis3 = score(code3, scale * static_cast<float>(dots[3]));
            return;
        }
        if (use_arm_packed4) {
            const uint8_t* code_rows[4] = {code0, code1, code2, code3};
            int64_t dots[4];
            rabitq_integer_adc::dot_product_packed_4bit_batch_arm(
                    quantized.data(), code_rows, 4, d, ex_offset, dots);
            dis0 = score(code0, scale * static_cast<float>(dots[0]));
            dis1 = score(code1, scale * static_cast<float>(dots[1]));
            dis2 = score(code2, scale * static_cast<float>(dots[2]));
            dis3 = score(code3, scale * static_cast<float>(dots[3]));
            return;
        }
        if (use_arm_split4) {
            const uint8_t* code_rows[4] = {code0, code1, code2, code3};
            const uint8_t* tail_rows[4] = {
                    split4_tail_for_code(code0),
                    split4_tail_for_code(code1),
                    split4_tail_for_code(code2),
                    split4_tail_for_code(code3)};
            int64_t dots[4];
            rabitq_integer_adc::dot_product_split_4bit_batch_arm(
                    quantized.data(), code_rows, tail_rows, 4, d, dots);
            dis0 = score(code0, scale * static_cast<float>(dots[0]));
            dis1 = score(code1, scale * static_cast<float>(dots[1]));
            dis2 = score(code2, scale * static_cast<float>(dots[2]));
            dis3 = score(code3, scale * static_cast<float>(dots[3]));
            return;
        }
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_prefix_direct()) {
            const uint8_t* code_rows[4] = {code0, code1, code2, code3};
            int64_t dots[4];
            dot_product_prefix_avx512(code_rows, 4, dots);
            dis0 = score(code0, scale * static_cast<float>(dots[0]));
            dis1 = score(code1, scale * static_cast<float>(dots[1]));
            dis2 = score(code2, scale * static_cast<float>(dots[2]));
            dis3 = score(code3, scale * static_cast<float>(dots[3]));
            return;
        }
        if (use_avx512_nested_direct()) {
            const uint8_t* code_rows[4] = {code0, code1, code2, code3};
            int64_t dots[4];
            dot_product_nested_avx512(code_rows, 4, dots);
            dis0 = score(code0, scale * static_cast<float>(dots[0]));
            dis1 = score(code1, scale * static_cast<float>(dots[1]));
            dis2 = score(code2, scale * static_cast<float>(dots[2]));
            dis3 = score(code3, scale * static_cast<float>(dots[3]));
            return;
        }
#endif
        const auto* levels0 = levels_for_code(code0, 0);
        const auto* levels1 = levels_for_code(code1, 1);
        const auto* levels2 = levels_for_code(code2, 2);
        const auto* levels3 = levels_for_code(code3, 3);
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_vnni) {
            rabitq_integer_adc::dot_product_batch_4_avx512_vnni(
                    quantized.data(),
                    levels0,
                    levels1,
                    levels2,
                    levels3,
                    d,
                    query_correction,
                    dot0,
                    dot1,
                    dot2,
                    dot3);
        } else
#endif
#ifdef COMPILE_SIMD_ARM_NEON
                if (use_arm_dotprod) {
            rabitq_integer_adc::dot_product_batch_4_arm(
                    quantized.data(),
                    levels0,
                    levels1,
                    levels2,
                    levels3,
                    d,
                    dot0,
                    dot1,
                    dot2,
                    dot3);
        } else
#endif
        {
            rabitq_integer_adc::dot_product_batch_4_scalar(
                    quantized.data(),
                    levels0,
                    levels1,
                    levels2,
                    levels3,
                    d,
                    dot0,
                    dot1,
                    dot2,
                    dot3);
        }
        dis0 = score(code0, scale * static_cast<float>(dot0));
        dis1 = score(code1, scale * static_cast<float>(dot1));
        dis2 = score(code2, scale * static_cast<float>(dot2));
        dis3 = score(code3, scale * static_cast<float>(dot3));
    }

    int preferred_batch_size() const final {
        if (!integer_query) {
            return 4;
        }
#ifdef COMPILE_SIMD_ARM_NEON
        if (use_arm_dotprod && nested_exact7_storage) {
            return 16;
        }
        if (use_arm_dotprod && d >= 256) {
            return 16;
        }
        if (use_arm_dotprod) {
            return 8;
        }
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_nested_direct()) {
            return 8;
        }
        if (use_avx512_vnni) {
            return 8;
        }
#endif
        return 4;
    }

    int max_tail_batch_size() const final {
        if (!integer_query) {
            return 0;
        }
#ifdef COMPILE_SIMD_ARM_NEON
        if (use_arm_dotprod) {
            return 7;
        }
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_nested_direct()) {
            return 7;
        }
#endif
        return 0;
    }

    void distances_batch_8(const int32_t* ids, float* distances) final {
        const uint8_t* code_rows[8];
        for (size_t k = 0; k < 8; ++k) {
            code_rows[k] = codes + static_cast<size_t>(ids[k]) * code_size;
        }
        if (!integer_query) {
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = distance_to_code(code_rows[k]);
            }
            return;
        }

        int64_t dots[8];
#ifdef COMPILE_SIMD_ARM_NEON
        if (use_arm_nested_lut4) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 8, d, dots);
            } else {
                rabitq_integer_adc::dot_product_nested_lut4_batch_arm(
                        quantized.data(),
                        code_rows,
                        8,
                        d,
                        nested_lut4_nibble ? 0 : tail_offset,
                        nested_lut4,
                        nested_lut4_nibble,
                        dots);
            }
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_nested_lut7) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 8, d, dots);
            } else if (nested_lut3_navigation) {
                rabitq_integer_adc::dot_product_nested_lut3_batch_arm(
                        quantized.data(),
                        code_rows,
                        8,
                        d,
                        nested_high1_offset,
                        nested_lut7,
                        dots);
            } else if (nested_exact7_storage) {
                rabitq_integer_adc::dot_product_nested_exact7_batch_arm(
                        quantized.data(),
                        code_rows,
                        8,
                        d,
                        tail_offset,
                        nested_high1_offset,
                        dots);
            } else {
                rabitq_integer_adc::dot_product_nested_lut7_batch_arm(
                        quantized.data(),
                        code_rows,
                        8,
                        d,
                        tail_offset,
                        nested_high1_offset,
                        nested_lut7,
                        dots);
            }
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_progressive) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 8, d, dots);
            } else if (tail_bits == 5) {
                rabitq_integer_adc::dot_product_progressive_full6_batch_arm(
                        quantized.data(), code_rows, 8, d, tail_offset, dots);
            } else {
                rabitq_integer_adc::dot_product_progressive_full7_batch_arm(
                        quantized.data(), code_rows, 8, d, tail_offset, dots);
            }
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_packed2) {
            rabitq_integer_adc::dot_product_packed_2bit_batch_8_arm(
                    quantized.data(), code_rows, d, ex_offset, dots);
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_packed4) {
            rabitq_integer_adc::dot_product_packed_4bit_batch_arm(
                    quantized.data(), code_rows, 8, d, ex_offset, dots);
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_split4) {
            const uint8_t* tail_rows[8];
            for (size_t k = 0; k < 8; ++k) {
                tail_rows[k] = split4_tail_for_code(code_rows[k]);
            }
            rabitq_integer_adc::dot_product_split_4bit_batch_arm(
                    quantized.data(), code_rows, tail_rows, 8, d, dots);
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_prefix_direct()) {
            dot_product_prefix_avx512(code_rows, 8, dots);
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_avx512_nested_direct()) {
            dot_product_nested_avx512(code_rows, 8, dots);
            for (size_t k = 0; k < 8; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
#endif
        const int8_t* levels[8];
        for (size_t k = 0; k < 8; ++k) {
            levels[k] = levels_for_code(code_rows[k], k);
        }
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_vnni) {
            rabitq_integer_adc::dot_product_batch_8_avx512_vnni(
                    quantized.data(), levels, d, query_correction, dots);
        } else
#endif
#ifdef COMPILE_SIMD_ARM_NEON
                if (use_arm_dotprod) {
            rabitq_integer_adc::dot_product_batch_8_arm(
                    quantized.data(), levels, d, dots);
        } else
#endif
        {
            rabitq_integer_adc::dot_product_batch_8_scalar(
                    quantized.data(), levels, d, dots);
        }
        for (size_t k = 0; k < 8; ++k) {
            distances[k] =
                    score(code_rows[k], scale * static_cast<float>(dots[k]));
        }
    }

    void distances_batch_16(const int32_t* ids, float* distances) final {
        const uint8_t* code_rows[16];
        for (size_t k = 0; k < 16; ++k) {
            code_rows[k] = codes + static_cast<size_t>(ids[k]) * code_size;
        }
        if (!integer_query) {
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = distance_to_code(code_rows[k]);
            }
            return;
        }

        int64_t dots[16];
#ifdef COMPILE_SIMD_ARM_NEON
        if (use_arm_nested_lut4) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 16, d, dots);
            } else {
                rabitq_integer_adc::dot_product_nested_lut4_batch_arm(
                        quantized.data(),
                        code_rows,
                        16,
                        d,
                        nested_lut4_nibble ? 0 : tail_offset,
                        nested_lut4,
                        nested_lut4_nibble,
                        dots);
            }
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_nested_lut7) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 16, d, dots);
            } else if (nested_lut3_navigation) {
                rabitq_integer_adc::dot_product_nested_lut3_batch_arm(
                        quantized.data(),
                        code_rows,
                        16,
                        d,
                        nested_high1_offset,
                        nested_lut7,
                        dots);
            } else if (nested_exact7_storage) {
                rabitq_integer_adc::dot_product_nested_exact7_batch_arm(
                        quantized.data(),
                        code_rows,
                        16,
                        d,
                        tail_offset,
                        nested_high1_offset,
                        dots);
            } else {
                rabitq_integer_adc::dot_product_nested_lut7_batch_arm(
                        quantized.data(),
                        code_rows,
                        16,
                        d,
                        tail_offset,
                        nested_high1_offset,
                        nested_lut7,
                        dots);
            }
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_progressive) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, 16, d, dots);
            } else if (tail_bits == 5) {
                rabitq_integer_adc::dot_product_progressive_full6_batch_arm(
                        quantized.data(), code_rows, 16, d, tail_offset, dots);
            } else {
                rabitq_integer_adc::dot_product_progressive_full7_batch_arm(
                        quantized.data(), code_rows, 16, d, tail_offset, dots);
            }
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_packed2) {
            rabitq_integer_adc::dot_product_packed_2bit_batch_16_arm(
                    quantized.data(), code_rows, d, ex_offset, dots);
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_packed4) {
            rabitq_integer_adc::dot_product_packed_4bit_batch_arm(
                    quantized.data(), code_rows, 16, d, ex_offset, dots);
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_split4) {
            const uint8_t* tail_rows[16];
            for (size_t k = 0; k < 16; ++k) {
                tail_rows[k] = split4_tail_for_code(code_rows[k]);
            }
            rabitq_integer_adc::dot_product_split_4bit_batch_arm(
                    quantized.data(), code_rows, tail_rows, 16, d, dots);
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_prefix_direct()) {
            dot_product_prefix_avx512(code_rows, 16, dots);
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_avx512_nested_direct()) {
            dot_product_nested_avx512(code_rows, 16, dots);
            for (size_t k = 0; k < 16; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
#endif
        const int8_t* levels[16];
        for (size_t k = 0; k < 16; ++k) {
            levels[k] = levels_for_code(code_rows[k], k);
        }
#ifdef COMPILE_SIMD_ARM_NEON
        if (use_arm_dotprod) {
            rabitq_integer_adc::dot_product_batch_16_arm(
                    quantized.data(), levels, d, dots);
        } else
#endif
        {
            rabitq_integer_adc::dot_product_batch_16_scalar(
                    quantized.data(), levels, d, dots);
        }
        for (size_t k = 0; k < 16; ++k) {
            distances[k] =
                    score(code_rows[k], scale * static_cast<float>(dots[k]));
        }
    }

    void distances_batch_tail(const int32_t* ids, int count, float* distances)
            final {
        FAISS_THROW_IF_NOT_MSG(
                count >= 1 && count <= 7,
                "RaBitQ tail batch size must be between 1 and 7");
        const uint8_t* code_rows[7];
        for (int k = 0; k < count; ++k) {
            code_rows[k] = codes + static_cast<size_t>(ids[k]) * code_size;
        }
        if (!integer_query) {
            for (int k = 0; k < count; ++k) {
                distances[k] = distance_to_code(code_rows[k]);
            }
            return;
        }

        int64_t dots[7];
#ifdef COMPILE_SIMD_ARM_NEON
        if (use_arm_nested_lut4) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, count, d, dots);
            } else {
                rabitq_integer_adc::dot_product_nested_lut4_batch_arm(
                        quantized.data(),
                        code_rows,
                        count,
                        d,
                        nested_lut4_nibble ? 0 : tail_offset,
                        nested_lut4,
                        nested_lut4_nibble,
                        dots);
            }
            for (int k = 0; k < count; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_nested_lut7) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, count, d, dots);
            } else if (nested_lut3_navigation) {
                rabitq_integer_adc::dot_product_nested_lut3_batch_arm(
                        quantized.data(),
                        code_rows,
                        count,
                        d,
                        nested_high1_offset,
                        nested_lut7,
                        dots);
            } else if (nested_exact7_storage) {
                rabitq_integer_adc::dot_product_nested_exact7_batch_arm(
                        quantized.data(),
                        code_rows,
                        count,
                        d,
                        tail_offset,
                        nested_high1_offset,
                        dots);
            } else {
                rabitq_integer_adc::dot_product_nested_lut7_batch_arm(
                        quantized.data(),
                        code_rows,
                        count,
                        d,
                        tail_offset,
                        nested_high1_offset,
                        nested_lut7,
                        dots);
            }
            for (int k = 0; k < count; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_progressive) {
            if (progressive_prefix_only) {
                rabitq_integer_adc::dot_product_progressive_prefix_batch_arm(
                        quantized.data(), code_rows, count, d, dots);
            } else if (tail_bits == 5) {
                rabitq_integer_adc::dot_product_progressive_full6_batch_arm(
                        quantized.data(),
                        code_rows,
                        count,
                        d,
                        tail_offset,
                        dots);
            } else {
                rabitq_integer_adc::dot_product_progressive_full7_batch_arm(
                        quantized.data(),
                        code_rows,
                        count,
                        d,
                        tail_offset,
                        dots);
            }
            for (int k = 0; k < count; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_packed2) {
            rabitq_integer_adc::dot_product_packed_2bit_batch_tail_arm(
                    quantized.data(), code_rows, count, d, ex_offset, dots);
            for (int k = 0; k < count; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_packed4) {
            rabitq_integer_adc::dot_product_packed_4bit_batch_arm(
                    quantized.data(), code_rows, count, d, ex_offset, dots);
            for (int k = 0; k < count; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_arm_split4) {
            const uint8_t* tail_rows[7];
            for (int k = 0; k < count; ++k) {
                tail_rows[k] = split4_tail_for_code(code_rows[k]);
            }
            rabitq_integer_adc::dot_product_split_4bit_batch_arm(
                    quantized.data(), code_rows, tail_rows, count, d, dots);
            for (int k = 0; k < count; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_prefix_direct()) {
            dot_product_prefix_avx512(code_rows, count, dots);
            for (int k = 0; k < count; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
        if (use_avx512_nested_direct()) {
            dot_product_nested_avx512(code_rows, count, dots);
            for (int k = 0; k < count; ++k) {
                distances[k] = score(
                        code_rows[k], scale * static_cast<float>(dots[k]));
            }
            return;
        }
#endif
        const int8_t* levels[7];
        for (int k = 0; k < count; ++k) {
            levels[k] = levels_for_code(code_rows[k], k);
        }
#ifdef COMPILE_SIMD_ARM_NEON
        if (use_arm_dotprod) {
            rabitq_integer_adc::dot_product_batch_tail_arm(
                    quantized.data(), levels, count, d, dots);
        } else
#endif
        {
            rabitq_integer_adc::dot_product_batch_tail_scalar(
                    quantized.data(), levels, count, d, dots);
        }
        for (int k = 0; k < count; ++k) {
            distances[k] =
                    score(code_rows[k], scale * static_cast<float>(dots[k]));
        }
    }

    int adaptive_batch_size() const final {
#ifdef COMPILE_SIMD_AVX512_SPR
        if (use_avx512_vnni) {
            return 8;
        }
#endif
        return 16;
    }

    void evaluate_current_mode(
            const int32_t* ids,
            int count,
            float* distances) {
        int offset = 0;
        if (count >= 16) {
            distances_batch_16(ids, distances);
            offset = 16;
        }
        if (count - offset >= 8) {
            distances_batch_8(ids + offset, distances + offset);
            offset += 8;
        }
        const int tail = count - offset;
        if (tail > 0) {
            distances_batch_tail(ids + offset, tail, distances + offset);
        }
    }

    void distances_prefix_bounds(
            const int32_t* ids,
            int count,
            float* estimates,
            float* lower_bounds) final {
        FAISS_THROW_IF_NOT_MSG(
                adaptive_error_norms != nullptr,
                "adaptive error-norm sidecar is not prepared");
        FAISS_THROW_IF_NOT_MSG(
                count >= 1 && count <= 16,
                "adaptive prefix batch must contain 1..16 candidates");
        const bool saved_prefix_only = progressive_prefix_only;
        const bool saved_mid = nested_lut3_navigation;
        progressive_prefix_only = true;
        nested_lut3_navigation = false;
        evaluate_current_mode(ids, count, estimates);
        progressive_prefix_only = saved_prefix_only;
        nested_lut3_navigation = saved_mid;

        const float projection_multiplier = 2.0f * scale * quantized_norm *
                adaptive_sigma / std::sqrt(static_cast<float>(d));
        const float query_sum_abs = 2.0f * std::abs(half_sum);
        for (int lane = 0; lane < count; ++lane) {
            const uint8_t* code =
                    codes + static_cast<size_t>(ids[lane]) * code_size;
            ProgressiveBitsFactors factors;
            memcpy(&factors, factors_for_code(code), sizeof(factors));
            const float prefix_scale = -0.5f * factors.f_rescale_prefix;
            const float full_scale = -0.5f * factors.f_rescale_full;
            const float bound =
                    projection_multiplier * adaptive_error_norms[ids[lane]] +
                    query_sum_abs * std::abs(full_scale - prefix_scale);
            lower_bounds[lane] = estimates[lane] - bound;
        }
    }

    void distances_full_selected(
            const int32_t* ids,
            int count,
            float* distances) final {
        FAISS_THROW_IF_NOT_MSG(
                count >= 1 && count <= 16,
                "adaptive refinement batch must contain 1..16 candidates");
        const bool saved_prefix_only = progressive_prefix_only;
        const bool saved_mid = nested_lut3_navigation;
        progressive_prefix_only = false;
        nested_lut3_navigation = false;
        evaluate_current_mode(ids, count, distances);
        progressive_prefix_only = saved_prefix_only;
        nested_lut3_navigation = saved_mid;
    }

    void adaptive_reset_stats() final {
        adaptive_prefix_counter = adaptive_refine_counter = 0;
        adaptive_full_fallback = false;
    }

    void adaptive_record(int prefix_count, int refine_count) final {
        adaptive_prefix_counter += static_cast<uint64_t>(prefix_count);
        adaptive_refine_counter += static_cast<uint64_t>(refine_count);
        // The current refinement pass re-runs the seven-bit kernel. Prefix
        // plus refinement therefore costs approximately 2 + 7*rho bits per
        // dimension and loses to direct full scoring at rho >= 5/7. The HNSW
        // heap threshold is unusually loose during its first few expansions,
        // so wait for 32 full SIMD batches before making the per-query
        // decision.
        if (!adaptive_full_fallback && adaptive_prefix_counter >= 512 &&
            7 * adaptive_refine_counter >= 5 * adaptive_prefix_counter) {
            adaptive_full_fallback = true;
        }
    }

    bool adaptive_should_use_full() const final {
        return adaptive_full_fallback;
    }

    uint64_t adaptive_prefix_count() const final {
        return adaptive_prefix_counter;
    }

    uint64_t adaptive_refine_count() const final {
        return adaptive_refine_counter;
    }

    float symmetric_dis(idx_t, idx_t) final {
        FAISS_THROW_MSG(
                "expanded RaBitQ ADC does not support graph construction");
    }
};

// Use shared constant from RaBitQUtils
using rabitq_utils::Z_MAX_BY_QB;

} // anonymous namespace

FlatCodesDistanceComputer* RaBitQuantizer::get_distance_computer(
        uint8_t qb,
        const float* centroid_in,
        bool centered,
        bool full_distance) const {
    // Dispatch on SIMDLevel once here so the distance computer methods
    // call the SIMD-specialized rabitq functions directly (no per-call
    // with_simd_level overhead).
    //
    // VPOPCNT rather than SPR: Ice Lake and Zen 4 have VPOPCNTDQ without the
    // rest of the SPR feature set. Below it, dispatch falls through to
    // rabitq_avx512.cpp.
    return with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_BASE_WITH_VPOPCNT>(
            [&]<SIMDLevel SL>() -> FlatCodesDistanceComputer* {
                if (qb == 0) {
                    auto dc =
                            std::make_unique<RaBitQDistanceComputerNotQ<SL>>();
                    dc->metric_type = metric_type;
                    dc->d = d;
                    dc->centroid = centroid_in;
                    dc->nb_bits = nb_bits;
                    dc->full_distance = full_distance;

                    return dc.release();
                } else {
                    auto dc = std::make_unique<RaBitQDistanceComputerQ<SL>>();
                    dc->metric_type = metric_type;
                    dc->d = d;
                    dc->centroid = centroid_in;
                    dc->qb = qb;
                    dc->centered = centered;
                    dc->nb_bits = nb_bits;
                    dc->full_distance = full_distance;

                    return dc.release();
                }
            });
}

void RaBitQuantizer::expand_codes(
        const uint8_t* packed_codes,
        size_t n,
        uint8_t* expanded_codes) const {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 2 && nb_bits <= 8,
            "expanded RaBitQ ADC requires 2..8 total bits");
    FAISS_THROW_IF_NOT_MSG(
            n == 0 || (packed_codes && expanded_codes),
            "null RaBitQ code buffer");

    const size_t ex_bits = nb_bits - 1;
    const size_t sign_bytes = (d + 7) / 8;
    const size_t ex_offset = sign_bytes + sizeof(SignBitFactorsWithError);
    const size_t ex_bytes = (d * ex_bits + 7) / 8;
    const size_t expanded_size = d + sizeof(ExtraBitsFactors);
    const int midpoint = 1 << ex_bits;

#pragma omp parallel for if (n > 1000)
    for (int64_t row = 0; row < static_cast<int64_t>(n); row++) {
        const uint8_t* packed = packed_codes + size_t(row) * code_size;
        const uint8_t* extra = packed + ex_offset;
        uint8_t* expanded = expanded_codes + size_t(row) * expanded_size;
        for (size_t j = 0; j < d; j++) {
            const int sign = rabitq_utils::extract_bit_standard(packed, j);
            const int low =
                    rabitq_utils::extract_code_inline(extra, j, ex_bits);
            const int level = (sign << ex_bits) + low - midpoint;
            expanded[j] = static_cast<uint8_t>(static_cast<int8_t>(level));
        }
        memcpy(expanded + d, extra + ex_bytes, sizeof(ExtraBitsFactors));
    }
}

FlatCodesDistanceComputer* RaBitQuantizer::get_expanded_distance_computer(
        const uint8_t* expanded_codes,
        const float* centroid_in,
        bool integer_query) const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2,
            "expanded RaBitQ ADC supports only L2");
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 2 && nb_bits <= 8,
            "expanded RaBitQ ADC requires 2..8 total bits");
    return new RaBitQExpandedDistanceComputer(
            expanded_codes, d, centroid_in, integer_query);
}

FlatCodesDistanceComputer* RaBitQuantizer::get_packed_integer_distance_computer(
        const uint8_t* packed_codes,
        const float* centroid_in) const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2,
            "packed integer RaBitQ ADC supports only L2");
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 2 && nb_bits <= 8,
            "packed integer RaBitQ ADC requires 2..8 total bits");
    return new RaBitQExpandedDistanceComputer(
            packed_codes, d, centroid_in, true, code_size, nb_bits);
}

FlatCodesDistanceComputer* RaBitQuantizer::get_split4_integer_distance_computer(
        const uint8_t* sign_codes,
        size_t sign_stride,
        const uint8_t* tail_codes,
        size_t tail_stride,
        const float* centroid_in) const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2 && nb_bits == 4,
            "split integer ADC requires four-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            sign_codes && tail_codes,
            "split integer ADC requires both code planes");
    return new RaBitQExpandedDistanceComputer(
            sign_codes,
            d,
            centroid_in,
            true,
            sign_stride,
            nb_bits,
            false,
            false,
            tail_codes,
            tail_stride);
}

size_t RaBitQuantizer::progressive_code_size() const {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 3 && nb_bits <= 8,
            "progressive RaBitQ requires 3..8 total bits");
    const size_t prefix_bytes = (2 * d + 7) / 8;
    const size_t tail_bytes = (d * (nb_bits - 1) + 7) / 8;
    return prefix_bytes + sizeof(ProgressiveBitsFactors) + tail_bytes;
}

void RaBitQuantizer::pack_progressive_codes(
        const uint8_t* packed_codes,
        size_t n,
        uint8_t* progressive_codes) const {
    FAISS_THROW_MSG(
            "legacy codes cannot be converted to independent-prefix "
            "progressive codes without the original vectors");
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 3 && nb_bits <= 8,
            "progressive RaBitQ requires 3..8 total bits");
    FAISS_THROW_IF_NOT_MSG(
            n == 0 || (packed_codes && progressive_codes),
            "null progressive RaBitQ code buffer");
    const size_t ex_bits = nb_bits - 1;
    const size_t tail_bits = nb_bits - 2;
    const size_t sign_bytes = (d + 7) / 8;
    const size_t legacy_extra_offset =
            sign_bytes + sizeof(SignBitFactorsWithError);
    const size_t legacy_extra_bytes = (d * ex_bits + 7) / 8;
    const size_t prefix_bytes = (2 * d + 7) / 8;
    const size_t output_size = progressive_code_size();
    const size_t output_factors_offset = prefix_bytes;
    const size_t output_tail_offset =
            prefix_bytes + sizeof(ProgressiveBitsFactors);
    const uint32_t tail_mask = (uint32_t(1) << tail_bits) - 1;

#pragma omp parallel for if (n > 1000)
    for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
        const uint8_t* input = packed_codes + size_t(row) * code_size;
        const uint8_t* extra = input + legacy_extra_offset;
        uint8_t* output = progressive_codes + size_t(row) * output_size;
        uint8_t* tail = output + output_tail_offset;
        memset(output, 0, output_size);
        for (size_t j = 0; j < d; ++j) {
            const uint32_t sign = rabitq_utils::extract_bit_standard(input, j);
            const uint32_t low =
                    rabitq_utils::extract_code_inline(extra, j, ex_bits);
            const uint32_t prefix = (sign << 1) | (low >> (ex_bits - 1));
            const uint32_t suffix = low & tail_mask;
            for (size_t bit = 0; bit < 2; ++bit) {
                if ((prefix >> bit) & 1) {
                    rabitq_utils::set_bit_standard(output, 2 * j + bit);
                }
            }
            for (size_t bit = 0; bit < tail_bits; ++bit) {
                if ((suffix >> bit) & 1) {
                    rabitq_utils::set_bit_standard(tail, tail_bits * j + bit);
                }
            }
        }
        ExtraBitsFactors legacy_factors;
        memcpy(&legacy_factors,
               extra + legacy_extra_bytes,
               sizeof(legacy_factors));
        ProgressiveBitsFactors factors;
        factors.f_add = legacy_factors.f_add_ex;
        factors.f_rescale_prefix = legacy_factors.f_rescale_ex *
                static_cast<float>(size_t(1) << tail_bits);
        factors.f_rescale_full = legacy_factors.f_rescale_ex;
        memcpy(output + output_factors_offset, &factors, sizeof(factors));
    }
}

void RaBitQuantizer::compute_progressive_codes_core(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid_in) const {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 3 && nb_bits <= 8,
            "progressive RaBitQ requires 3..8 total bits");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2,
            "progressive RaBitQ currently supports only L2");
    const size_t tail_bits = nb_bits - 1;
    const uint32_t tail_mask = (uint32_t(1) << tail_bits) - 1;
    const size_t prefix_bytes = (2 * d + 7) / 8;
    const size_t factors_offset = prefix_bytes;
    const size_t tail_offset = prefix_bytes + sizeof(ProgressiveBitsFactors);
    const size_t output_size = progressive_code_size();

#pragma omp parallel for if (n > 1000)
    for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
        const float* input = x + size_t(row) * d;
        uint8_t* output = codes + size_t(row) * output_size;
        uint8_t* tail = output + tail_offset;
        memset(output, 0, output_size);

        std::vector<float> normalized_abs(d);
        float norm_sqr = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            const float value =
                    input[j] - (centroid_in ? centroid_in[j] : 0.0f);
            norm_sqr += value * value;
            normalized_abs[j] = value;
        }
        const float norm = std::sqrt(norm_sqr);
        ProgressiveBitsFactors factors;
        factors.f_add = norm_sqr;
        if (norm < 1e-10f) {
            memcpy(output + factors_offset, &factors, sizeof(factors));
            continue;
        }
        for (size_t j = 0; j < d; ++j) {
            normalized_abs[j] = std::abs(normalized_abs[j]) / norm;
        }

        const float prefix_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), d, 2);
        const float full_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), d, nb_bits);
        double prefix_ipnorm = 0.0;
        double full_ipnorm = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const float magnitude = normalized_abs[j];
            const uint32_t prefix_magnitude =
                    std::min(uint32_t(prefix_t * magnitude + 1e-5f), 1u);
            const uint32_t full_magnitude =
                    std::min(uint32_t(full_t * magnitude + 1e-5f), tail_mask);
            prefix_ipnorm += (prefix_magnitude + 0.5) * magnitude;
            full_ipnorm += (full_magnitude + 0.5) * magnitude;

            const bool positive =
                    input[j] - (centroid_in ? centroid_in[j] : 0.0f) > 0.0f;
            const uint32_t prefix_low =
                    positive ? prefix_magnitude : (~prefix_magnitude) & 1u;
            const uint32_t full_low =
                    positive ? full_magnitude : (~full_magnitude) & tail_mask;
            const uint32_t prefix = (uint32_t(positive) << 1) | prefix_low;
            for (size_t bit = 0; bit < 2; ++bit) {
                if ((prefix >> bit) & 1) {
                    rabitq_utils::set_bit_standard(output, 2 * j + bit);
                }
            }
            for (size_t bit = 0; bit < tail_bits; ++bit) {
                if ((full_low >> bit) & 1) {
                    rabitq_utils::set_bit_standard(tail, tail_bits * j + bit);
                }
            }
        }
        factors.f_rescale_prefix =
                static_cast<float>(-2.0 * norm / prefix_ipnorm);
        factors.f_rescale_full = static_cast<float>(-2.0 * norm / full_ipnorm);
        memcpy(output + factors_offset, &factors, sizeof(factors));
    }
}

FlatCodesDistanceComputer* RaBitQuantizer::
        get_progressive_integer_distance_computer(
                const uint8_t* progressive_codes,
                const float* centroid_in,
                bool prefix_only) const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2,
            "progressive integer RaBitQ ADC supports only L2");
    return new RaBitQExpandedDistanceComputer(
            progressive_codes,
            d,
            centroid_in,
            true,
            progressive_code_size(),
            nb_bits,
            true,
            prefix_only);
}

size_t RaBitQuantizer::nested_lut7_code_size() const {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits == 7, "nested LUT layout requires seven-bit RaBitQ");
    const size_t prefix_bytes = (2 * d + 7) / 8;
    const size_t low4_bytes = (4 * d + 7) / 8;
    const size_t high1_bytes = (d + 7) / 8;
    return prefix_bytes + sizeof(ProgressiveBitsFactors) + low4_bytes +
            high1_bytes;
}

void RaBitQuantizer::compute_nested_lut7_codes_core(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid_in,
        const uint8_t* lut,
        bool mid_navigation_factor,
        bool exact_level_encoding) const {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits == 7 && metric_type == MetricType::METRIC_L2,
            "nested LUT encoding requires seven-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(lut != nullptr, "null nested LUT codebook");
    const size_t prefix_bytes = (2 * d + 7) / 8;
    const size_t factors_offset = prefix_bytes;
    const size_t low4_offset = prefix_bytes + sizeof(ProgressiveBitsFactors);
    const size_t low4_bytes = (4 * d + 7) / 8;
    const size_t high1_offset = low4_offset + low4_bytes;
    const size_t output_size = nested_lut7_code_size();

#pragma omp parallel for if (n > 1000)
    for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
        const float* input = x + size_t(row) * d;
        uint8_t* output = codes + size_t(row) * output_size;
        uint8_t* low4 = output + low4_offset;
        uint8_t* high1 = output + high1_offset;
        memset(output, 0, output_size);

        std::vector<float> normalized_abs(d);
        float norm_sqr = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            const float value =
                    input[j] - (centroid_in ? centroid_in[j] : 0.0f);
            norm_sqr += value * value;
            normalized_abs[j] = value;
        }
        const float norm = std::sqrt(norm_sqr);
        ProgressiveBitsFactors factors;
        factors.f_add = norm_sqr;
        if (norm < 1e-10f) {
            memcpy(output + factors_offset, &factors, sizeof(factors));
            continue;
        }
        for (size_t j = 0; j < d; ++j) {
            normalized_abs[j] = std::abs(normalized_abs[j]) / norm;
        }

        const float prefix_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), d, 2);
        const float full_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), d, 7);
        double prefix_ipnorm = 0.0;
        double full_ipnorm = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const float magnitude = normalized_abs[j];
            const uint32_t fine =
                    std::min(uint32_t(full_t * magnitude + 1e-5f), 63u);
            const uint32_t coarse = exact_level_encoding
                    ? fine >> 5
                    : std::min(uint32_t(prefix_t * magnitude + 1e-5f), 1u);
            uint32_t local = 0;
            uint32_t best_error = 64;
            for (uint32_t candidate = 0; candidate < 32; ++candidate) {
                const uint32_t reconstructed = lut[coarse * 32 + candidate];
                const uint32_t error = reconstructed > fine
                        ? reconstructed - fine
                        : fine - reconstructed;
                if (error < best_error) {
                    best_error = error;
                    local = candidate;
                }
            }
            const uint32_t reconstructed = lut[coarse * 32 + local];
            if (mid_navigation_factor) {
                const uint32_t mid_reconstructed =
                        lut[coarse * 32 + ((local >> 4) << 4) + 7];
                prefix_ipnorm += (mid_reconstructed + 0.5) * magnitude;
            } else {
                prefix_ipnorm += (coarse + 0.5) * magnitude;
            }
            full_ipnorm += (reconstructed + 0.5) * magnitude;

            const bool positive =
                    input[j] - (centroid_in ? centroid_in[j] : 0.0f) > 0.0f;
            const uint32_t prefix_low = positive ? coarse : (~coarse) & 1u;
            const uint32_t prefix = (uint32_t(positive) << 1) | prefix_low;
            for (size_t bit = 0; bit < 2; ++bit) {
                if ((prefix >> bit) & 1) {
                    rabitq_utils::set_bit_standard(output, 2 * j + bit);
                }
            }
            for (size_t bit = 0; bit < 4; ++bit) {
                if ((local >> bit) & 1) {
                    rabitq_utils::set_bit_standard(low4, 4 * j + bit);
                }
            }
            if ((local >> 4) & 1) {
                rabitq_utils::set_bit_standard(high1, j);
            }
        }
        factors.f_rescale_prefix =
                static_cast<float>(-2.0 * norm / prefix_ipnorm);
        factors.f_rescale_full = static_cast<float>(-2.0 * norm / full_ipnorm);
        memcpy(output + factors_offset, &factors, sizeof(factors));
    }
}

FlatCodesDistanceComputer* RaBitQuantizer::
        get_nested_lut7_integer_distance_computer(
                const uint8_t* nested_codes,
                const float* centroid_in,
                bool prefix_only,
                bool mid_navigation,
                const uint8_t* lut,
                const float* adaptive_error_norms,
                float adaptive_sigma) const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2 && nb_bits == 7,
            "nested LUT integer ADC requires seven-bit L2 RaBitQ");
    return new RaBitQExpandedDistanceComputer(
            nested_codes,
            d,
            centroid_in,
            true,
            nested_lut7_code_size(),
            nb_bits,
            true,
            prefix_only,
            nullptr,
            0,
            true,
            lut,
            mid_navigation,
            adaptive_error_norms,
            adaptive_sigma);
}

size_t RaBitQuantizer::nested_lut4_code_size(bool nibble_layout) const {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits == 4, "nested LUT4 layout requires four-bit RaBitQ");
    if (nibble_layout) {
        return (4 * d + 7) / 8 + sizeof(ExtraBitsFactors);
    }
    const size_t prefix_bytes = (2 * d + 7) / 8;
    const size_t local_bytes = (2 * d + 7) / 8;
    return prefix_bytes + sizeof(ProgressiveBitsFactors) + local_bytes;
}

void RaBitQuantizer::compute_nested_lut4_codes_core(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid_in,
        const uint8_t* lut,
        bool nibble_layout) const {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits == 4 && metric_type == MetricType::METRIC_L2,
            "nested LUT4 encoding requires four-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(lut != nullptr, "null nested LUT4 codebook");
    const size_t prefix_bytes = (2 * d + 7) / 8;
    const size_t staged_factors_offset = prefix_bytes;
    const size_t local_offset = prefix_bytes + sizeof(ProgressiveBitsFactors);
    const size_t nibble_bytes = (4 * d + 7) / 8;
    const size_t output_size = nested_lut4_code_size(nibble_layout);

#pragma omp parallel for if (n > 1000)
    for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
        const float* input = x + size_t(row) * d;
        uint8_t* output = codes + size_t(row) * output_size;
        memset(output, 0, output_size);

        std::vector<float> normalized_abs(d);
        float norm_sqr = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            const float value =
                    input[j] - (centroid_in ? centroid_in[j] : 0.0f);
            norm_sqr += value * value;
            normalized_abs[j] = value;
        }
        const float norm = std::sqrt(norm_sqr);
        if (norm < 1e-10f) {
            if (nibble_layout) {
                ExtraBitsFactors factors{};
                factors.f_add_ex = norm_sqr;
                memcpy(output + nibble_bytes, &factors, sizeof(factors));
            } else {
                ProgressiveBitsFactors factors{};
                factors.f_add = norm_sqr;
                memcpy(output + staged_factors_offset,
                       &factors,
                       sizeof(factors));
            }
            continue;
        }
        for (size_t j = 0; j < d; ++j) {
            normalized_abs[j] = std::abs(normalized_abs[j]) / norm;
        }

        const float prefix_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), d, 2);
        const float full_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), d, 4);
        double prefix_ipnorm = 0.0;
        double full_ipnorm = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const float magnitude = normalized_abs[j];
            const uint32_t fine =
                    std::min(uint32_t(full_t * magnitude + 1e-5f), 7u);
            const uint32_t coarse =
                    std::min(uint32_t(prefix_t * magnitude + 1e-5f), 1u);
            uint32_t local = 0;
            uint32_t best_error = 8;
            for (uint32_t candidate = 0; candidate < 4; ++candidate) {
                const uint32_t reconstructed = lut[coarse * 4 + candidate];
                const uint32_t error = reconstructed > fine
                        ? reconstructed - fine
                        : fine - reconstructed;
                if (error < best_error) {
                    best_error = error;
                    local = candidate;
                }
            }
            const uint32_t reconstructed = lut[coarse * 4 + local];
            prefix_ipnorm += (coarse + 0.5) * magnitude;
            full_ipnorm += (reconstructed + 0.5) * magnitude;

            const bool positive =
                    input[j] - (centroid_in ? centroid_in[j] : 0.0f) > 0.0f;
            if (nibble_layout) {
                const uint32_t symbol =
                        (uint32_t(positive) << 3) | (coarse << 2) | local;
                output[j >> 1] |= static_cast<uint8_t>(symbol << (4 * (j & 1)));
            } else {
                const uint32_t prefix_low = positive ? coarse : (~coarse) & 1u;
                const uint32_t prefix = (uint32_t(positive) << 1) | prefix_low;
                output[j >> 2] |= static_cast<uint8_t>(prefix << (2 * (j & 3)));
                output[local_offset + (j >> 2)] |=
                        static_cast<uint8_t>(local << (2 * (j & 3)));
            }
        }
        const float prefix_rescale =
                static_cast<float>(-2.0 * norm / prefix_ipnorm);
        const float full_rescale =
                static_cast<float>(-2.0 * norm / full_ipnorm);
        if (nibble_layout) {
            ExtraBitsFactors factors;
            factors.f_add_ex = norm_sqr;
            factors.f_rescale_ex = full_rescale;
            memcpy(output + nibble_bytes, &factors, sizeof(factors));
        } else {
            ProgressiveBitsFactors factors;
            factors.f_add = norm_sqr;
            factors.f_rescale_prefix = prefix_rescale;
            factors.f_rescale_full = full_rescale;
            memcpy(output + staged_factors_offset, &factors, sizeof(factors));
        }
    }
}

FlatCodesDistanceComputer* RaBitQuantizer::
        get_nested_lut4_integer_distance_computer(
                const uint8_t* nested_codes,
                const float* centroid_in,
                bool prefix_only,
                bool nibble_layout,
                const uint8_t* lut) const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2 && nb_bits == 4,
            "nested LUT4 integer ADC requires four-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            !prefix_only || !nibble_layout,
            "contiguous nibble LUT4 has no standalone prefix navigation");
    return new RaBitQExpandedDistanceComputer(
            nested_codes,
            d,
            centroid_in,
            true,
            nested_lut4_code_size(nibble_layout),
            nb_bits,
            !nibble_layout,
            prefix_only,
            nullptr,
            0,
            false,
            nullptr,
            false,
            nullptr,
            0.0f,
            true,
            nibble_layout,
            lut);
}

bool RaBitQuantizer::expanded_integer_uses_native_dotprod() const {
#ifdef COMPILE_SIMD_AVX512_SPR
    if (SIMDConfig::get_dispatched_level() == SIMDLevel::AVX512_SPR) {
        return true;
    }
#endif
#ifdef COMPILE_SIMD_ARM_NEON
    return rabitq_integer_adc::arm_dotprod_supported();
#else
    return false;
#endif
}

} // namespace faiss
