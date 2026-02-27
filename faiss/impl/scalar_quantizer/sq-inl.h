/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Private implementation header for per-SIMD scalar quantizer TUs.
// Do not include in public APIs.

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/simdlib.h>

#include <faiss/impl/simd_dispatch.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/expanded_scanners.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>

// Define USE_* macros for the sub-headers. These macros gate struct template
// specializations that contain SIMD intrinsics. They must use compiler-defined
// macros (not COMPILE_SIMD_*) because in DD mode the COMPILE_SIMD_* macros are
// set globally, but the actual SIMD instructions are only available in per-SIMD
// TUs that receive the appropriate compiler flags.
#if defined(__AVX512F__) && defined(__F16C__)
#define USE_AVX512_F16C
#endif

#if defined(__AVX2__) && defined(__F16C__)
#define USE_F16C
#endif

#if defined(__aarch64__)
#if !defined(__GNUC__) || __GNUC__ >= 8
#define USE_NEON
#endif
#endif

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

#include <faiss/impl/scalar_quantizer/sq_impl.h>

namespace faiss {

namespace scalar_quantizer {

using QuantizerType = ScalarQuantizer::QuantizerType;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * select_quantizer_1_body: the big switch returning SQuantizer*
 *******************************************************************/

template <SIMDLevel SL>
ScalarQuantizer::SQuantizer* select_quantizer_1_body(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerTemplate<
                    Codec8bit,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_6bit:
            return new QuantizerTemplate<
                    Codec6bit,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerTemplate<
                    Codec4bit,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerTemplate<
                    Codec8bit,
                    QuantizerTemplateScaling::UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerTemplate<
                    Codec4bit,
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
 * select_distance_computer_body: the big switch returning
 * SQDistanceComputer*
 *******************************************************************/

template <class Sim>
SQDistanceComputer* select_distance_computer_body(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr SIMDLevel SL = Sim::simd_level;
    switch (qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec8bit,
                            QuantizerTemplateScaling::UNIFORM,
                            SL>,
                    Sim,
                    SL>(d, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec4bit,
                            QuantizerTemplateScaling::UNIFORM,
                            SL>,
                    Sim,
                    SL>(d, trained);

        case ScalarQuantizer::QT_8bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec8bit,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SL>,
                    Sim,
                    SL>(d, trained);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec6bit,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SL>,
                    Sim,
                    SL>(d, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec4bit,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SL>,
                    Sim,
                    SL>(d, trained);

        case ScalarQuantizer::QT_fp16:
            return new DCTemplate<QuantizerFP16<SL>, Sim, SL>(d, trained);

        case ScalarQuantizer::QT_bf16:
            return new DCTemplate<QuantizerBF16<SL>, Sim, SL>(d, trained);

        case ScalarQuantizer::QT_8bit_direct:
            if constexpr (SL == SIMDLevel::AVX512) {
                if (d % 32 == 0) {
                    return new DistanceComputerByte<Sim, SL>(
                            static_cast<int>(d), trained);
                }
            } else if constexpr (SL == SIMDLevel::AVX2) {
                if (d % 16 == 0) {
                    return new DistanceComputerByte<Sim, SL>(
                            static_cast<int>(d), trained);
                }
            }
            return new DCTemplate<Quantizer8bitDirect<SL>, Sim, SL>(d, trained);

        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate<Quantizer8bitDirectSigned<SL>, Sim, SL>(
                    d, trained);
        default:
            FAISS_THROW_MSG("unknown qtype");
    }
}

/*******************************************************************
 * IVFSQScannerIP / IVFSQScannerL2 — moved from anonymous namespace
 * in ScalarQuantizer.cpp
 *******************************************************************/

namespace sq_internal {

template <class DCClass>
struct IVFSQScannerIP : InvertedListScanner {
    DCClass dc;
    bool by_residual;

    float accu0; /// added to all distances

    IVFSQScannerIP(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained), by_residual(by_residual), accu0(0) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
        this->keep_max = true;
    }

    void set_query(const float* query) override {
        dc.set_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        accu0 = by_residual ? coarse_dis : 0;
    }

    float distance_to_code(const uint8_t* code) const final {
        return accu0 + dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return run_scan_codes_fix_C<CMin<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

template <class DCClass>
struct IVFSQScannerL2 : InvertedListScanner {
    DCClass dc;

    bool by_residual;
    const Index* quantizer;
    const float* x; /// current query

    std::vector<float> tmp;

    IVFSQScannerL2(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            const Index* quantizer,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained),
              by_residual(by_residual),
              quantizer(quantizer),
              x(nullptr),
              tmp(d) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        x = query;
        if (!quantizer) {
            dc.set_query(query);
        }
    }

    void set_list(idx_t list_no, float /*coarse_dis*/) override {
        this->list_no = list_no;
        if (by_residual) {
            quantizer->compute_residual(x, tmp.data(), list_no);
            dc.set_query(tmp.data());
        } else {
            dc.set_query(x);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return run_scan_codes_fix_C<CMax<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

} // namespace sq_internal

/*******************************************************************
 * select_InvertedListScanner_body: the lambda chain
 *******************************************************************/

template <SIMDLevel SL>
InvertedListScanner* select_InvertedListScanner_body(
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
            return new sq_internal::IVFSQScannerL2<DCClass>(
                    int(d),
                    trained,
                    code_size,
                    quantizer,
                    store_pairs,
                    sel,
                    by_residual);
        } else if constexpr (
                DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
            return new sq_internal::IVFSQScannerIP<DCClass>(
                    int(d), trained, code_size, store_pairs, sel, by_residual);
        } else {
            FAISS_THROW_MSG("unsupported metric type");
        }
    };

    auto select_by_simd_and_metric =
            [&]<SIMDLevel SL2, class Similarity>() -> InvertedListScanner* {
        switch (qtype) {
            case ScalarQuantizer::QT_8bit_uniform:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec8bit,
                                QuantizerTemplateScaling::UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_4bit_uniform:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec4bit,
                                QuantizerTemplateScaling::UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_8bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec8bit,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_4bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec4bit,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_6bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec6bit,
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
