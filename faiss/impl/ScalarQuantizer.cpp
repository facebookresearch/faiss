/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <cstring>
#include <memory>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/simd_levels.h>

#include <faiss/impl/scalar_quantizer/training.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>

#include <faiss/impl/scalar_quantizer/sq_impl.h>

namespace faiss {

using scalar_quantizer::sq_select_distance_computer;
using scalar_quantizer::sq_select_InvertedListScanner;
using scalar_quantizer::sq_select_quantizer;

/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer(size_t d, QuantizerType qtype)
        : Quantizer(d), qtype(qtype) {
    set_derived_sizes();
}

ScalarQuantizer::ScalarQuantizer() {}

void ScalarQuantizer::set_derived_sizes() {
    switch (qtype) {
        case QT_8bit:
        case QT_8bit_uniform:
        case QT_8bit_direct:
        case QT_8bit_direct_signed:
            code_size = d;
            bits = 8;
            break;
        case QT_4bit:
        case QT_4bit_uniform:
            code_size = (d + 1) / 2;
            bits = 4;
            break;
        case QT_6bit:
            code_size = (d * 6 + 7) / 8;
            bits = 6;
            break;
        case QT_fp16:
            code_size = d * 2;
            bits = 16;
            break;
        case QT_bf16:
            code_size = d * 2;
            bits = 16;
            break;
        default:
            break;
    }
}

void ScalarQuantizer::train(size_t n, const float* x) {
    using scalar_quantizer::train_NonUniform;
    using scalar_quantizer::train_Uniform;

    int bit_per_dim = qtype == QT_4bit_uniform ? 4
            : qtype == QT_4bit                 ? 4
            : qtype == QT_6bit                 ? 6
            : qtype == QT_8bit_uniform         ? 8
            : qtype == QT_8bit                 ? 8
                                               : -1;

    switch (qtype) {
        case QT_4bit_uniform:
        case QT_8bit_uniform:
            train_Uniform(
                    rangestat,
                    rangestat_arg,
                    n * d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_4bit:
        case QT_8bit:
        case QT_6bit:
            train_NonUniform(
                    rangestat,
                    rangestat_arg,
                    n,
                    int(d),
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_fp16:
        case QT_8bit_direct:
        case QT_bf16:
        case QT_8bit_direct_signed:
            // no training necessary
            break;
        default:
            break;
    }
}

ScalarQuantizer::SQuantizer* ScalarQuantizer::select_quantizer() const {
    return with_simd_level([&]<SIMDLevel SL>() -> SQuantizer* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* q = sq_select_quantizer<SL>(qtype, d, trained);
            if (q) {
                return q;
            }
        }
        return sq_select_quantizer<SIMDLevel::NONE>(qtype, d, trained);
    });
}

void ScalarQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

    memset(codes, 0, code_size * n);
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        squant->encode_vector(x + i * d, codes + i * code_size);
    }
}

void ScalarQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        squant->decode_vector(codes + i * code_size, x + i * d);
    }
}

ScalarQuantizer::SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
    return with_simd_level([&]<SIMDLevel SL>() -> SQDistanceComputer* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* dc =
                    sq_select_distance_computer<SL>(metric, qtype, d, trained);
            if (dc) {
                return dc;
            }
        }
        return sq_select_distance_computer<SIMDLevel::NONE>(
                metric, qtype, d, trained);
    });
}

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) const {
    return with_simd_level([&]<SIMDLevel SL>() -> InvertedListScanner* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* s = sq_select_InvertedListScanner<SL>(
                    qtype,
                    mt,
                    d,
                    code_size,
                    trained,
                    quantizer,
                    store_pairs,
                    sel,
                    by_residual);
            if (s) {
                return s;
            }
        }
        return sq_select_InvertedListScanner<SIMDLevel::NONE>(
                qtype,
                mt,
                d,
                code_size,
                trained,
                quantizer,
                store_pairs,
                sel,
                by_residual);
    });
}

} // namespace faiss
