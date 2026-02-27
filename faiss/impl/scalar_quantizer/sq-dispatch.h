/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

#ifndef THE_LEVEL_TO_DISPATCH
#error "THE_LEVEL_TO_DISPATCH should be set on input to this header"
#endif

namespace faiss {

namespace scalar_quantizer {

// Define SL as alias for THE_LEVEL_TO_DISPATCH for use in this file
constexpr SIMDLevel SL = THE_LEVEL_TO_DISPATCH;

// Returns true if dimension d is compatible with the given SIMD level
template <SIMDLevel SL2>
constexpr bool is_dimension_compatible(size_t d) {
    if constexpr (SL2 == SIMDLevel::AVX512) {
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
            if constexpr (SL2 == SIMDLevel::AVX512) {
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
            return new DCTemplate<Quantizer8bitDirectSigned<SL2>, Sim, SL2>(
                    d, trained);
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
                if constexpr (SL2 == SIMDLevel::AVX512) {
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
                return scan.template operator()<DCTemplate<
                        Quantizer8bitDirectSigned<SL2>,
                        Similarity,
                        SL2>>();
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
