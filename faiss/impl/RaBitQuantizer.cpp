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
#include <faiss/impl/simd_dispatch.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::ExtraBitsFactors;
using rabitq_utils::QueryFactorsData;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

RaBitQuantizer::RaBitQuantizer(
        size_t d_in,
        MetricType metric,
        size_t nb_bits_in,
        bool dense_layout_in)
        : Quantizer(d_in, 0), // code_size will be set below
          metric_type{metric},
          nb_bits{nb_bits_in},
          dense_layout{dense_layout_in} {
    // Validate nb_bits range
    FAISS_THROW_IF_NOT(nb_bits >= 1 && nb_bits <= 9);
    FAISS_THROW_IF_NOT(!dense_layout || nb_bits > 1);

    // Set code_size using compute_code_size
    code_size = compute_code_size(d, nb_bits);
}

size_t RaBitQuantizer::compute_code_size(size_t d_in, size_t num_bits) const {
    // Validate inputs
    FAISS_THROW_IF_NOT(num_bits >= 1 && num_bits <= 9);

    size_t ex_bits = num_bits - 1;

    if (dense_layout) {
        FAISS_THROW_IF_NOT(num_bits > 1);
        return (d_in * num_bits + 7) / 8 + sizeof(SignBitFactorsWithError) +
                sizeof(ExtraBitsFactors);
    }

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
        const size_t base_code_size =
                dense_layout ? (d * nb_bits + 7) / 8 : (d + 7) / 8;

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
                            code + base_code_size);
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
            if (xb && !dense_layout) {
                rabitq_utils::set_bit_standard(binary_code, j);
            }
        }

        // Step 2: Compute ex-bits quantization (if nb_bits > 1)
        if (ex_bits > 0) {
            // Pointer to ex-bit code section
            uint8_t* ex_code =
                    code + base_code_size + sizeof(SignBitFactorsWithError);
            std::vector<uint8_t> packed_ex_code;
            if (dense_layout) {
                packed_ex_code.resize((d * ex_bits + 7) / 8);
                ex_code = packed_ex_code.data();
            }
            // Pointer to ex-factors section
            ExtraBitsFactors byte_ex_factors;
            ExtraBitsFactors* ex_factors = dense_layout
                    ? &byte_ex_factors
                    : reinterpret_cast<ExtraBitsFactors*>(
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

            if (dense_layout) {
                for (size_t j = 0; j < d; j++) {
                    const uint16_t sign = residual[j] > 0.0f
                            ? static_cast<uint16_t>(1u << ex_bits)
                            : 0;
                    const uint16_t value =
                            sign |
                            static_cast<uint16_t>(
                                    rabitq_utils::extract_code_inline(
                                            ex_code, j, ex_bits));
                    const size_t bit_pos = j * nb_bits;
                    const size_t byte_pos = bit_pos / 8;
                    const size_t shift = bit_pos % 8;
                    const uint32_t shifted = static_cast<uint32_t>(value)
                            << shift;
                    const size_t nbytes = (shift + nb_bits + 7) / 8;
                    for (size_t b = 0; b < nbytes; b++) {
                        binary_code[byte_pos + b] |=
                                static_cast<uint8_t>(shifted >> (8 * b));
                    }
                }
                memcpy(code + base_code_size + sizeof(SignBitFactorsWithError),
                       ex_factors,
                       sizeof(ExtraBitsFactors));
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
        const size_t base_code_size =
                dense_layout ? (d * nb_bits + 7) / 8 : (d + 7) / 8;
        const SignBitFactors* fac = (ex_bits == 0)
                ? reinterpret_cast<const SignBitFactors*>(code + (d + 7) / 8)
                : reinterpret_cast<const SignBitFactorsWithError*>(
                          code + base_code_size);

        // this is the baseline code
        //
        // compute <q,o> using floats
        for (size_t j = 0; j < d; j++) {
            // extract i-th bit
            const uint8_t masker = (1 << (j % 8));
            const float bit = dense_layout
                    ? ((rabitq_utils::extract_code_inline(
                                binary_data, j, nb_bits) &
                        (1u << ex_bits)) != 0
                               ? 1.0f
                               : 0.0f)
                    : (((binary_data[j / 8] & masker) == masker) ? 1.0f : 0.0f);

            // compute the output code
            x[i * d + j] = (bit - 0.5f) * fac->dp_multiplier * 2 * inv_d_sqrt +
                    ((centroid_in == nullptr) ? 0 : centroid_in[j]);
        }
    }
}

namespace {

template <SIMDLevel SL>
void distance_to_code_full_batch_4_impl(
        const uint8_t* const codes[4],
        size_t d,
        size_t nb_bits,
        const float* rotated_q,
        float qr_base,
        MetricType metric_type,
        bool dense_layout,
        float out[4]) {
    const size_t ex_bits = nb_bits - 1;
    if (ex_bits == 0) {
        FAISS_THROW_MSG("multi-bit batch helper requires extra bits");
    }

    const size_t code_size_base =
            dense_layout ? (d * nb_bits + 7) / 8 : (d + 7) / 8;
    const size_t ex_offset = code_size_base + sizeof(SignBitFactorsWithError);
    const size_t ex_code_size = (d * ex_bits + 7) / 8;
    const uint8_t* sign_bits[4];
    const uint8_t* ex_codes[4];
    const ExtraBitsFactors* ex_factors[4];
    for (size_t i = 0; i < 4; i++) {
        sign_bits[i] = codes[i];
        ex_codes[i] = dense_layout ? codes[i] : codes[i] + ex_offset;
        ex_factors[i] = reinterpret_cast<const ExtraBitsFactors*>(
                dense_layout ? codes[i] + code_size_base +
                                sizeof(SignBitFactorsWithError)
                             : ex_codes[i] + ex_code_size);
    }

    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);
    float inner_products[4];
    if (dense_layout) {
        rabitq::multibit::compute_inner_product_dense_batch_4<SL>(
                ex_codes, rotated_q, d, nb_bits, cb, inner_products);
    } else {
        rabitq::multibit::compute_inner_product_batch_4<SL>(
                sign_bits, ex_codes, rotated_q, d, ex_bits, cb, inner_products);
    }

    for (size_t i = 0; i < 4; i++) {
        float distance = qr_base + ex_factors[i]->f_add_ex +
                ex_factors[i]->f_rescale_ex * inner_products[i];
        out[i] = metric_type == MetricType::METRIC_L2 ? std::max(0.0f, distance)
                                                      : distance;
    }
}

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
    bool dense_layout = false;

    RaBitQDistanceComputerNotQ() = default;

    // Compute distance using only 1-bit codes (fast)
    float distance_to_code_1bit_impl(
            const uint8_t* binary_data,
            const SignBitFactors* base_fac) const {
        // this is the baseline code
        //
        // compute <q,o> using floats
        const float dot_qo = rabitq::selected_float_sum<SL>(
                binary_data, rotated_q.data(), d);

        // Apply query factors
        float final_dot = query_fac.c1 * dot_qo - query_fac.c34;

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

        const size_t code_size_base =
                dense_layout ? (d * nb_bits + 7) / 8 : (d + 7) / 8;
        const size_t ex_bits = nb_bits - 1;
        const SignBitFactors* base_fac = (ex_bits == 0)
                ? reinterpret_cast<const SignBitFactors*>(code + code_size_base)
                : reinterpret_cast<const SignBitFactorsWithError*>(
                          code + code_size_base);
        if (!dense_layout) {
            return distance_to_code_1bit_impl(code, base_fac);
        }

        float dot_qo = 0.0f;
        for (size_t i = 0; i < d; i++) {
            if ((rabitq_utils::extract_code_inline(code, i, nb_bits) &
                 (1u << (nb_bits - 1))) != 0) {
                dot_qo += rotated_q[i];
            }
        }
        const float final_dot = query_fac.c1 * dot_qo - query_fac.c34;
        const float pre_dist = base_fac->or_minus_c_l2sqr +
                query_fac.qr_to_c_L2sqr -
                2 * base_fac->dp_multiplier * final_dot;
        return metric_type == MetricType::METRIC_L2
                ? std::max(0.0f, pre_dist)
                : -0.5f * (pre_dist - query_fac.qr_norm_L2sqr);
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
        const size_t dense_code_size = (d * nb_bits + 7) / 8;
        size_t offset = (dense_layout ? dense_code_size : (d + 7) / 8) +
                sizeof(SignBitFactorsWithError);
        const uint8_t* ex_code = dense_layout ? code : code + offset;
        const ExtraBitsFactors* ex_fac =
                reinterpret_cast<const ExtraBitsFactors*>(
                        dense_layout ? code + dense_code_size +
                                        sizeof(SignBitFactorsWithError)
                                     : ex_code + (d * ex_bits + 7) / 8);

        float qr_base = (metric_type == MetricType::METRIC_INNER_PRODUCT)
                ? query_fac.q_dot_c
                : query_fac.qr_to_c_L2sqr;
        if (dense_layout) {
            const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);
            const float ex_ip =
                    rabitq::multibit::compute_inner_product_dense<SL>(
                            ex_code, rotated_q.data(), d, nb_bits, cb);
            const float distance =
                    qr_base + ex_fac->f_add_ex + ex_fac->f_rescale_ex * ex_ip;
            return metric_type == MetricType::METRIC_L2
                    ? std::max(0.0f, distance)
                    : distance;
        }
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

    void distance_to_code_batch_4(
            const uint8_t* code0,
            const uint8_t* code1,
            const uint8_t* code2,
            const uint8_t* code3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final {
        if (nb_bits == 1) {
            dis0 = distance_to_code_full(code0);
            dis1 = distance_to_code_full(code1);
            dis2 = distance_to_code_full(code2);
            dis3 = distance_to_code_full(code3);
            return;
        }
        const uint8_t* codes[4] = {code0, code1, code2, code3};
        float distances[4];
        const float qr_base = metric_type == MetricType::METRIC_INNER_PRODUCT
                ? query_fac.q_dot_c
                : query_fac.qr_to_c_L2sqr;
        distance_to_code_full_batch_4_impl<SL>(
                codes,
                d,
                nb_bits,
                rotated_q.data(),
                qr_base,
                metric_type,
                dense_layout,
                distances);
        dis0 = distances[0];
        dis1 = distances[1];
        dis2 = distances[2];
        dis3 = distances[3];
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
        const size_t code_size_base =
                dense_layout ? (d * nb_bits + 7) / 8 : (d + 7) / 8;
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
            const float est_distance = distance_to_code_1bit(codes);

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
struct RaBitQDistanceComputerQ final : RaBitQDistanceComputer {
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

    void distance_to_code_batch_4(
            const uint8_t* code0,
            const uint8_t* code1,
            const uint8_t* code2,
            const uint8_t* code3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final {
        if (nb_bits == 1) {
            dis0 = distance_to_code_full(code0);
            dis1 = distance_to_code_full(code1);
            dis2 = distance_to_code_full(code2);
            dis3 = distance_to_code_full(code3);
            return;
        }
        const uint8_t* codes[4] = {code0, code1, code2, code3};
        float distances[4];
        const float qr_base = metric_type == MetricType::METRIC_INNER_PRODUCT
                ? query_fac.q_dot_c
                : query_fac.qr_to_c_L2sqr;
        distance_to_code_full_batch_4_impl<SL>(
                codes,
                d,
                nb_bits,
                rotated_q.data(),
                qr_base,
                metric_type,
                false,
                distances);
        dis0 = distances[0];
        dis1 = distances[1];
        dis2 = distances[2];
        dis3 = distances[3];
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

// Use shared constant from RaBitQUtils
using rabitq_utils::Z_MAX_BY_QB;

} // anonymous namespace

FlatCodesDistanceComputer* RaBitQuantizer::get_distance_computer(
        uint8_t qb,
        const float* centroid_in,
        bool centered) const {
    // Dispatch on SIMDLevel once here so the distance computer methods
    // call the SIMD-specialized rabitq functions directly (no per-call
    // with_simd_level overhead).
    //
    // Use A0_SPR (which includes AVX512_SPR) so that on Sapphire Rapids
    // and later x86 microarchitectures the VPOPCNTDQ-based RaBitQ
    // specialization in rabitq_avx512_spr.cpp is selected. On AVX-512
    // CPUs without VPOPCNTDQ, dispatch falls through to the AVX512
    // specialization in rabitq_avx512.cpp.
    return with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_A0_SPR>(
            [&]<SIMDLevel SL>() -> FlatCodesDistanceComputer* {
                if (qb == 0) {
                    auto dc =
                            std::make_unique<RaBitQDistanceComputerNotQ<SL>>();
                    dc->metric_type = metric_type;
                    dc->d = d;
                    dc->centroid = centroid_in;
                    dc->nb_bits = nb_bits;
                    dc->dense_layout = dense_layout;

                    return dc.release();
                } else {
                    FAISS_THROW_IF_NOT_MSG(
                            !dense_layout,
                            "dense-layout RaBitQ currently requires qb=0");
                    auto dc = std::make_unique<RaBitQDistanceComputerQ<SL>>();
                    dc->metric_type = metric_type;
                    dc->d = d;
                    dc->centroid = centroid_in;
                    dc->qb = qb;
                    dc->centered = centered;
                    dc->nb_bits = nb_bits;

                    return dc.release();
                }
            });
}

} // namespace faiss
