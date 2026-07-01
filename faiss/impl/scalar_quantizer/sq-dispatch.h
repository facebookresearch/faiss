/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>
#include <limits>

#ifndef THE_LEVEL_TO_DISPATCH
#error "THE_LEVEL_TO_DISPATCH should be set on input to this header"
#endif

namespace faiss {

namespace scalar_quantizer {

// Define SL as alias for THE_LEVEL_TO_DISPATCH for use in this file
constexpr SIMDLevel SL = THE_LEVEL_TO_DISPATCH;

/*******************************************************************
 * TurboQuant SIMD kernel: masked_sum
 * Compute sum of arr[j] where bit j of the bitmask is set.
 * NONE specialization is inline; AVX2/AVX512/NEON specializations
 * live in sq-avx2.cpp / sq-avx512.cpp / sq-neon.cpp.
 *******************************************************************/

template <SIMDLevel SL0>
float turboq_masked_sum(const float* arr, const uint8_t* bits, size_t d);

template <>
inline float turboq_masked_sum<SIMDLevel::NONE>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    float result = 0;
    for (size_t byte_idx = 0; byte_idx < (d + 7) / 8; byte_idx++) {
        uint8_t b = bits[byte_idx];
        size_t base = byte_idx * 8;
        size_t end = std::min(base + 8, d);
        for (size_t j = base; j < end; j++) {
            if (b & (1 << (j - base))) {
                result += arr[j];
            }
        }
    }
    return result;
}

/*******************************************************************
 * Full TurboQuant DC — lives here because it needs both
 * quantizers.h (QuantizerTurboQuantFull, SQTurboQFactors) and
 * similarities.h (Similarity::metric_type). distance_computers.h
 * can't include quantizers.h due to header ordering.
 *******************************************************************/
template <int NBits, class Similarity, SIMDLevel SL2>
struct DCTurboQuantFull : ScalarQuantizer::TurboQuantRefine::DistanceComputer {
    using Sim = Similarity;
    QuantizerTurboQuantFull<NBits, SIMDLevel::NONE> quant;
    std::vector<float> query;
    std::vector<float> query_proj;
    float q_norm_sq = 0;
    float qjl_coeff = 0;
    float total_qproj_sum = 0;

    // Pre-screening state
    const float* threshold_ptr = nullptr;
    bool prescreen_l2 = false;
    float qjl_error_coeff = 0;
    mutable size_t n_total = 0;
    mutable size_t n_skipped = 0;

    // Integer popcount state
    uint8_t qb = 0;
    bool int_qjl = false;
    std::vector<uint8_t> rearranged_q;
    float mse_base = 0;
    float mse_int_scale = 0;
    float mse_popcnt_scale = 0;

    // Integer QJL popcount state
    std::vector<uint8_t> rearranged_qproj;
    float qjl_int_scale = 0;
    float qjl_popcnt_scale = 0;

    // Scaled centroids for 1-bit MSE fast path (NBits==2)
    float scaled_c0 = 0;
    float scaled_c1 = 0;
    float delta_centroid = 0;
    float total_q_sum = 0;

    // Multi-bit MSE decomposed coefficients (NBits==3, kMSEBits==2)
    float mse_multi_base = 0;
    float mse_coeff_s0 = 0;
    float mse_coeff_s1 = 0;
    float mse_coeff_s01 = 0;
    mutable std::vector<uint8_t> scratch_and;

    DCTurboQuantFull(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {
        qjl_coeff = std::sqrt(M_PI / 2.0f) / static_cast<float>(d);
    }

    void configure(uint8_t qb_in, bool int_qjl_in) override {
        qb = qb_in;
        int_qjl = int_qjl_in;
    }

    void set_prescreen_threshold(const float* ptr, bool l2) override {
        threshold_ptr = ptr;
        prescreen_l2 = l2;
    }

    void clear_prescreen_threshold() override {
        threshold_ptr = nullptr;
    }

    void set_query(const float* x) final {
        q = x;
        size_t d = quant.d;
        query.assign(x, x + d);
        q_norm_sq = fvec_norm_L2sqr(x, d);

        // Project query
        query_proj.resize(d);
        quant.project_forward(x, query_proj.data());
        float inv_sqrt_pd =
                1.0f / std::sqrt(static_cast<float>(quant.padded_d));
        for (size_t j = 0; j < d; j++) {
            query_proj[j] *= inv_sqrt_pd;
        }

        total_qproj_sum = 0;
        for (size_t j = 0; j < d; j++) {
            total_qproj_sum += query_proj[j];
        }

        // Pre-screening: worst-case L1 bound on QJL error
        float qproj_l1 = 0;
        for (size_t j = 0; j < d; j++) {
            qproj_l1 += std::abs(query_proj[j]);
        }
        qjl_error_coeff = qjl_coeff * qproj_l1;

        // Pre-compute for 1-bit MSE fast path
        if constexpr (NBits == 2) {
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
            scaled_c0 = quant.centroids[0] * inv_sqrt_d;
            scaled_c1 = quant.centroids[1] * inv_sqrt_d;
            delta_centroid = scaled_c1 - scaled_c0;
            total_q_sum = 0;
            for (size_t j = 0; j < d; j++) {
                total_q_sum += query[j];
            }

            // Integer popcount setup
            if (qb > 0) {
                size_t byte_size = (d + 7) / 8;
                float q_min = *std::min_element(query.begin(), query.end());
                float q_max = *std::max_element(query.begin(), query.end());
                float q_range = q_max - q_min;
                if (q_range < 1e-30f) {
                    q_range = 1e-30f;
                }
                float max_val = static_cast<float>((1 << qb) - 1);
                float scale = max_val / q_range;
                float delta_q = q_range / max_val;

                rearranged_q.assign(byte_size * qb, 0);
                for (size_t j = 0; j < d; j++) {
                    int qval = static_cast<int>(
                            std::round((query[j] - q_min) * scale));
                    qval = std::max(
                            0, std::min(static_cast<int>(max_val), qval));
                    for (int b = 0; b < qb; b++) {
                        if (qval & (1 << b)) {
                            rearranged_q[b * byte_size + j / 8] |=
                                    (1 << (j % 8));
                        }
                    }
                }
                mse_base = scaled_c0 * total_q_sum;
                mse_int_scale = delta_centroid * delta_q;
                mse_popcnt_scale = delta_centroid * q_min;
            }
        }

        // Pre-compute for 2-bit MSE decomposed path (NBits==3)
        if constexpr (NBits == 3) {
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
            const float* c = quant.centroids;
            total_q_sum = 0;
            for (size_t j = 0; j < d; j++) {
                total_q_sum += query[j];
            }
            mse_multi_base = c[0] * inv_sqrt_d * total_q_sum;
            mse_coeff_s0 = (c[1] - c[0]) * inv_sqrt_d;
            mse_coeff_s1 = (c[2] - c[0]) * inv_sqrt_d;
            mse_coeff_s01 = (c[3] - c[2] - c[1] + c[0]) * inv_sqrt_d;
            scratch_and.resize((d + 7) / 8);
        }

        // Integer QJL: quantize projected query into bit-planes
        if (qb > 0 && int_qjl) {
            size_t byte_size = (d + 7) / 8;
            float qp_min =
                    *std::min_element(query_proj.begin(), query_proj.end());
            float qp_max =
                    *std::max_element(query_proj.begin(), query_proj.end());
            float qp_range = qp_max - qp_min;
            if (qp_range < 1e-30f) {
                qp_range = 1e-30f;
            }
            float max_val = static_cast<float>((1 << qb) - 1);
            float qp_scale = max_val / qp_range;
            float delta_qp = qp_range / max_val;

            rearranged_qproj.assign(byte_size * qb, 0);
            for (size_t j = 0; j < d; j++) {
                int qval = static_cast<int>(
                        std::round((query_proj[j] - qp_min) * qp_scale));
                qval = std::max(0, std::min(static_cast<int>(max_val), qval));
                for (int b = 0; b < qb; b++) {
                    if (qval & (1 << b)) {
                        rearranged_qproj[b * byte_size + j / 8] |=
                                (1 << (j % 8));
                    }
                }
            }
            qjl_popcnt_scale = qp_min;
            qjl_int_scale = delta_qp;
        }

        n_total = 0;
        n_skipped = 0;
    }

    float query_to_code(const uint8_t* code) const final {
        size_t d = quant.d;
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const auto* factors = reinterpret_cast<const SQTurboQFactors*>(
                code + quant.mse_total_bytes + quant.qjl_plane_bytes);
        float norm = factors->norm;
        float gamma = factors->gamma;

        // Stage 1: MSE dot product
        float mse_dot = 0;
        if constexpr (NBits == 2) {
            if (qb > 0) {
                // Integer popcount path for 1-bit MSE
                size_t byte_size = (d + 7) / 8;
                uint64_t and_result = rabitq::bitwise_and_dot_product<SL2>(
                        rearranged_q.data(), code, byte_size, qb);
                uint64_t pop = rabitq::popcount<SL2>(code, byte_size);
                mse_dot = mse_base +
                        mse_int_scale * static_cast<float>(and_result) +
                        mse_popcnt_scale * static_cast<float>(pop);
            } else {
                // Float path: masked accumulation
                float pos_sum = turboq_masked_sum<SL2>(query.data(), code, d);
                mse_dot = scaled_c0 * total_q_sum + delta_centroid * pos_sum;
            }
        } else if constexpr (NBits == 3) {
            // 2-bit MSE: decompose into 3 masked sums over bit-planes.
            size_t pb = quant.mse_plane_bytes;
            float s0 = turboq_masked_sum<SL2>(query.data(), code, d);
            float s1 = turboq_masked_sum<SL2>(query.data(), code + pb, d);
            for (size_t i = 0; i < pb; i++) {
                scratch_and[i] = code[i] & code[pb + i];
            }
            float s01 =
                    turboq_masked_sum<SL2>(query.data(), scratch_and.data(), d);
            mse_dot = mse_multi_base + mse_coeff_s0 * s0 + mse_coeff_s1 * s1 +
                    mse_coeff_s01 * s01;
        } else {
            // kMSEBits > 2: per-dimension fallback
            for (size_t j = 0; j < d; j++) {
                uint8_t idx = quant.load_mse_index(code, j);
                mse_dot += query[j] * quant.centroids[idx] * inv_sqrt_d;
            }
        }

        // Pre-screening
        if (threshold_ptr != nullptr) {
            n_total++;
            float bound = qjl_error_coeff * gamma * norm;
            float mse_ip = norm * mse_dot;

            if constexpr (Similarity::metric_type == METRIC_INNER_PRODUCT) {
                if (mse_ip + bound <= *threshold_ptr) {
                    n_skipped++;
                    return -std::numeric_limits<float>::infinity();
                }
            } else {
                float best_possible =
                        q_norm_sq + norm * norm - 2.0f * (mse_ip + bound);
                if (best_possible >= *threshold_ptr) {
                    n_skipped++;
                    return std::numeric_limits<float>::infinity();
                }
            }
        }

        // Stage 2: QJL dot product
        const uint8_t* qjl_code = code + quant.mse_total_bytes;
        float qjl_dot;
        if (qb > 0 && int_qjl) {
            size_t byte_size = (d + 7) / 8;
            uint64_t and_result = rabitq::bitwise_and_dot_product<SL2>(
                    rearranged_qproj.data(), qjl_code, byte_size, qb);
            uint64_t pop = rabitq::popcount<SL2>(qjl_code, byte_size);
            float pos_sum = qjl_popcnt_scale * static_cast<float>(pop) +
                    qjl_int_scale * static_cast<float>(and_result);
            qjl_dot = qjl_coeff * gamma * (2.0f * pos_sum - total_qproj_sum);
        } else {
            float pos_sum =
                    turboq_masked_sum<SL2>(query_proj.data(), qjl_code, d);
            qjl_dot = qjl_coeff * gamma * (2.0f * pos_sum - total_qproj_sum);
        }

        float estimated_ip = norm * (mse_dot + qjl_dot);

        if constexpr (Similarity::metric_type == METRIC_INNER_PRODUCT) {
            return estimated_ip;
        } else {
            return q_norm_sq + norm * norm - 2.0f * estimated_ip;
        }
    }

    float symmetric_dis(idx_t, idx_t) override {
        FAISS_THROW_MSG("Not implemented");
    }
};

// Returns true if dimension d is compatible with the given SIMD level
template <SIMDLevel SL2>
constexpr bool is_dimension_compatible(size_t d) {
    if constexpr (SL2 == SIMDLevel::AVX512 || SL2 == SIMDLevel::AVX512_SPR) {
        return d % 16 == 0;
    } else if constexpr (SL2 == SIMDLevel::AVX2 || SL2 == SIMDLevel::ARM_NEON) {
        return d % 8 == 0;
    } else {
        return true; // SIMDLevel::NONE has no alignment requirements
    }
}

/*******************************************************************
 * sq_select_quantizer: the big switch returning SQuantizer*
 *******************************************************************/

template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<THE_LEVEL_TO_DISPATCH>(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    // Return nullptr for incompatible dimensions in SIMD cases
    if constexpr (SL != SIMDLevel::NONE) {
        if (!is_dimension_compatible<SL>(d)) {
            return nullptr;
        }
    }
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerTemplate<
                    Codec8bit<SL>,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_6bit:
            return new QuantizerTemplate<
                    Codec6bit<SL>,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerTemplate<
                    Codec4bit<SL>,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerTemplate<
                    Codec8bit<SL>,
                    QuantizerTemplateScaling::UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerTemplate<
                    Codec4bit<SL>,
                    QuantizerTemplateScaling::UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_fp16:
            return new QuantizerFP16<SL>(d, trained);
        case ScalarQuantizer::QT_bf16:
            return new QuantizerBF16<SL>(d, trained);
        case ScalarQuantizer::QT_8bit_direct:
            return new Quantizer8bitDirect<SL>(d, trained);
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned<SL>(d, trained);
        case ScalarQuantizer::QT_0bit:
            FAISS_THROW_MSG(
                    "QT_0bit does not support standalone quantization, use IndexIVFScalarQuantizer");
        case ScalarQuantizer::QT_1bit_tqmse:
            return new QuantizerTurboQuantMSE<1, SL>(d, trained);
        case ScalarQuantizer::QT_2bit_tqmse:
            return new QuantizerTurboQuantMSE<2, SL>(d, trained);
        case ScalarQuantizer::QT_3bit_tqmse:
            return new QuantizerTurboQuantMSE<3, SL>(d, trained);
        case ScalarQuantizer::QT_4bit_tqmse:
            return new QuantizerTurboQuantMSE<4, SL>(d, trained);
        case ScalarQuantizer::QT_8bit_tqmse:
            return new QuantizerTurboQuantMSE<8, SL>(d, trained);
        case ScalarQuantizer::QT_2bit_tq:
            return new QuantizerTurboQuantFull<2, SL>(d, trained);
        case ScalarQuantizer::QT_3bit_tq:
            return new QuantizerTurboQuantFull<3, SL>(d, trained);
        case ScalarQuantizer::QT_4bit_tq:
            return new QuantizerTurboQuantFull<4, SL>(d, trained);
        case ScalarQuantizer::QT_5bit_tq:
            return new QuantizerTurboQuantFull<5, SL>(d, trained);
        default:
            FAISS_THROW_MSG("unknown qtype");
    }
}

/*******************************************************************
 * select_distance_computer_body: helper for sq_select_distance_computer
 *******************************************************************/

template <class Sim, SIMDLevel SL2>
SQDistanceComputer* select_distance_computer_body(
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    // Return nullptr for incompatible dimensions in SIMD cases
    if constexpr (SL2 != SIMDLevel::NONE) {
        if (!is_dimension_compatible<SL2>(d)) {
            return nullptr;
        }
    }
    switch (qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec8bit<SL2>,
                            QuantizerTemplateScaling::UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec4bit<SL2>,
                            QuantizerTemplateScaling::UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_8bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec8bit<SL2>,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec6bit<SL2>,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec4bit<SL2>,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_fp16:
            return new DCTemplate<QuantizerFP16<SL2>, Sim, SL2>(d, trained);

        case ScalarQuantizer::QT_bf16:
            return new DCTemplate<QuantizerBF16<SL2>, Sim, SL2>(d, trained);

        case ScalarQuantizer::QT_8bit_direct:
            if constexpr (
                    SL2 == SIMDLevel::AVX512 || SL2 == SIMDLevel::AVX512_SPR) {
                if (d % 32 == 0) {
                    return new DistanceComputerByte<Sim, SL2>(
                            static_cast<int>(d), trained);
                }
            } else if constexpr (SL2 == SIMDLevel::AVX2) {
                if (d % 16 == 0) {
                    return new DistanceComputerByte<Sim, SL2>(
                            static_cast<int>(d), trained);
                }
            }
            return new DCTemplate<Quantizer8bitDirect<SL2>, Sim, SL2>(
                    d, trained);

        case ScalarQuantizer::QT_8bit_direct_signed:
            if constexpr (SL2 == SIMDLevel::AVX512_SPR) {
                if (d % 64 == 0) {
                    return new DistanceComputerByteSigned<Sim, SL2>(
                            static_cast<int>(d), trained);
                }
            }
            return new DCTemplate<Quantizer8bitDirectSigned<SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_0bit:
            FAISS_THROW_MSG(
                    "QT_0bit does not support standalone distance computation, use IndexIVFScalarQuantizer");
        case ScalarQuantizer::QT_1bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<1, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_2bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<2, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_3bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<3, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_4bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<4, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_8bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<8, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_2bit_tq:
            // FRICTION: bypasses DCTemplate entirely — custom DC
            // that doesn't fit the Quantizer+Similarity decomposition
            return new DCTurboQuantFull<2, Sim, SL2>(d, trained);
        case ScalarQuantizer::QT_3bit_tq:
            return new DCTurboQuantFull<3, Sim, SL2>(d, trained);
        case ScalarQuantizer::QT_4bit_tq:
            return new DCTurboQuantFull<4, Sim, SL2>(d, trained);
        case ScalarQuantizer::QT_5bit_tq:
            return new DCTurboQuantFull<5, Sim, SL2>(d, trained);
        default:
            FAISS_THROW_MSG("unknown qtype");
    }
}

/*******************************************************************
 * sq_select_distance_computer: returns SQDistanceComputer*
 *******************************************************************/

template <>
SQDistanceComputer* sq_select_distance_computer<THE_LEVEL_TO_DISPATCH>(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    if (metric == METRIC_L2) {
        return select_distance_computer_body<SimilarityL2<SL>, SL>(
                qtype, d, trained);
    } else {
        return select_distance_computer_body<SimilarityIP<SL>, SL>(
                qtype, d, trained);
    }
}

/*******************************************************************
 * sq_select_InvertedListScanner: returns InvertedListScanner*
 *******************************************************************/

template <>
InvertedListScanner* sq_select_InvertedListScanner<THE_LEVEL_TO_DISPATCH>(
        QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    auto scan = [&]<class DCClass>() -> InvertedListScanner* {
        if constexpr (DCClass::Sim::metric_type == METRIC_L2) {
            return new IVFSQScannerL2<DCClass>(
                    int(d),
                    trained,
                    code_size,
                    quantizer,
                    store_pairs,
                    sel,
                    by_residual);
        } else if constexpr (
                DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
            return new IVFSQScannerIP<DCClass>(
                    int(d), trained, code_size, store_pairs, sel, by_residual);
        } else {
            FAISS_THROW_MSG("unsupported metric type");
        }
    };

    auto select_by_simd_and_metric =
            [&]<SIMDLevel SL2, class Similarity>() -> InvertedListScanner* {
        // Return nullptr for incompatible dimensions in SIMD cases
        if constexpr (SL2 != SIMDLevel::NONE) {
            if (!is_dimension_compatible<SL2>(d)) {
                return nullptr;
            }
        }
        switch (qtype) {
            case ScalarQuantizer::QT_8bit_uniform:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec8bit<SL2>,
                                QuantizerTemplateScaling::UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_4bit_uniform:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec4bit<SL2>,
                                QuantizerTemplateScaling::UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_8bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec8bit<SL2>,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_4bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec4bit<SL2>,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_6bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec6bit<SL2>,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_fp16:
                return scan.template
                operator()<DCTemplate<QuantizerFP16<SL2>, Similarity, SL2>>();
            case ScalarQuantizer::QT_bf16:
                return scan.template
                operator()<DCTemplate<QuantizerBF16<SL2>, Similarity, SL2>>();
            case ScalarQuantizer::QT_8bit_direct:
                if constexpr (
                        SL2 == SIMDLevel::AVX512 ||
                        SL2 == SIMDLevel::AVX512_SPR) {
                    if (d % 32 == 0) {
                        return scan.template
                        operator()<DistanceComputerByte<Similarity, SL2>>();
                    }
                } else if constexpr (SL2 == SIMDLevel::AVX2) {
                    if (d % 16 == 0) {
                        return scan.template
                        operator()<DistanceComputerByte<Similarity, SL2>>();
                    }
                }
                return scan.template operator()<DCTemplate<
                        Quantizer8bitDirect<SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_8bit_direct_signed:
                if constexpr (SL2 == SIMDLevel::AVX512_SPR) {
                    if (d % 64 == 0) {
                        return scan.template operator()<
                                DistanceComputerByteSigned<Similarity, SL2>>();
                    }
                }
                return scan.template operator()<DCTemplate<
                        Quantizer8bitDirectSigned<SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_0bit:
                return new IVFCoarseDistanceScanner(
                        Similarity::metric_type != METRIC_L2, store_pairs, sel);
            case ScalarQuantizer::QT_1bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<1, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_2bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<2, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_3bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<3, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_4bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<4, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_8bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<8, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_2bit_tq:
                return scan.template
                operator()<DCTurboQuantFull<2, Similarity, SL2>>();
            case ScalarQuantizer::QT_3bit_tq:
                return scan.template
                operator()<DCTurboQuantFull<3, Similarity, SL2>>();
            case ScalarQuantizer::QT_4bit_tq:
                return scan.template
                operator()<DCTurboQuantFull<4, Similarity, SL2>>();
            case ScalarQuantizer::QT_5bit_tq:
                return scan.template
                operator()<DCTurboQuantFull<5, Similarity, SL2>>();
            default:
                FAISS_THROW_MSG("unknown qtype");
        }
    };

    if (mt == METRIC_L2) {
        return select_by_simd_and_metric
                .template operator()<SL, SimilarityL2<SL>>();
    } else if (mt == METRIC_INNER_PRODUCT) {
        return select_by_simd_and_metric
                .template operator()<SL, SimilarityIP<SL>>();
    }
    FAISS_THROW_MSG("unsupported metric type");
}

} // namespace scalar_quantizer

} // namespace faiss
