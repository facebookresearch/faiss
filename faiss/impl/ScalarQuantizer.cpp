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

#include <faiss/impl/scalar_quantizer/scanners.h>

#define THE_LEVEL_TO_DISPATCH SIMDLevel::NONE
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

namespace faiss {

/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer(size_t d_in, QuantizerType qtype_in)
        : Quantizer(d_in), qtype(qtype_in) {
    set_derived_sizes();
}

ScalarQuantizer::ScalarQuantizer() {}

void ScalarQuantizer::set_derived_sizes() {
    switch (qtype) {
        case QT_1bit_tqmse:
            code_size = (d + 7) / 8;
            bits = 1;
            break;
        case QT_2bit_tqmse:
            code_size = (d * 2 + 7) / 8;
            bits = 2;
            break;
        case QT_3bit_tqmse:
            code_size = (d * 3 + 7) / 8;
            bits = 3;
            break;
        case QT_8bit:
        case QT_8bit_uniform:
        case QT_8bit_direct:
        case QT_8bit_direct_signed:
        case QT_8bit_tqmse:
            code_size = d;
            bits = 8;
            break;
        case QT_4bit:
        case QT_4bit_uniform:
        case QT_4bit_tqmse:
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
        case QT_0bit:
            code_size = 0;
            bits = 0;
            break;
        default:
            break;
    }
}

void ScalarQuantizer::train(size_t n, const float* x) {
    using scalar_quantizer::train_NonUniform;
    using scalar_quantizer::train_Uniform;

    if (qtype == QT_0bit) {
        return; // nothing to train for centroid-only mode
    }

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
        case QT_1bit_tqmse:
            scalar_quantizer::train_TurboQuantMSE(d, 1, trained);
            break;
        case QT_2bit_tqmse:
            scalar_quantizer::train_TurboQuantMSE(d, 2, trained);
            break;
        case QT_3bit_tqmse:
            scalar_quantizer::train_TurboQuantMSE(d, 3, trained);
            break;
        case QT_4bit_tqmse:
            scalar_quantizer::train_TurboQuantMSE(d, 4, trained);
            break;
        case QT_8bit_tqmse:
            scalar_quantizer::train_TurboQuantMSE(d, 8, trained);
            break;
        default:
            break;
    }
}

ScalarQuantizer::SQuantizer* ScalarQuantizer::select_quantizer() const {
    return with_simd_level([&]<SIMDLevel SL>() -> SQuantizer* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* q = scalar_quantizer::sq_select_quantizer<SL>(
                    qtype, d, trained);
            if (q) {
                return q;
            }
        }
        return scalar_quantizer::sq_select_quantizer<SIMDLevel::NONE>(
                qtype, d, trained);
    });
}

void ScalarQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    if (code_size == 0) {
        return; // QT_0bit: nothing to encode
    }
    std::unique_ptr<SQuantizer> squant(select_quantizer());

    memset(codes, 0, code_size * n);
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        squant->encode_vector(x + i * d, codes + i * code_size);
    }
}

void ScalarQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    if (code_size == 0) {
        memset(x, 0, sizeof(float) * d * n);
        return; // QT_0bit: no per-vector data, zero-fill
    }
    std::unique_ptr<SQuantizer> squant(select_quantizer());

#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        squant->decode_vector(codes + i * code_size, x + i * d);
    }
}

ScalarQuantizer::SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
    return with_simd_level([&]<SIMDLevel SL>() -> SQDistanceComputer* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* dc = scalar_quantizer::sq_select_distance_computer<SL>(
                    metric, qtype, d, trained);
            if (dc) {
                return dc;
            }
        }
        return scalar_quantizer::sq_select_distance_computer<SIMDLevel::NONE>(
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
            auto* s = scalar_quantizer::sq_select_InvertedListScanner<SL>(
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
        return scalar_quantizer::sq_select_InvertedListScanner<SIMDLevel::NONE>(
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
