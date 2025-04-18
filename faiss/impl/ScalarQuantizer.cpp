/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ScalarQuantizer.h>

#include <cstdio>

#include <faiss/impl/platform_macros.h>
#include <omp.h>

#include <faiss/utils/simdlib.h>

#include <faiss/impl/scalar_quantizer/training.h>

#include <faiss/IndexIVF.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/utils.h>

/*******************************************************************
 * ScalarQuantizer implementation
 *
 * The main source of complexity is to support combinations of 4
 * variants without incurring runtime tests or virtual function calls:
 *
 * - 4 / 6 / 8 bits per code component
 * - uniform / non-uniform
 * - IP / L2 distance search
 * - scalar / SIMD distance computation
 *
 * The appropriate Quantizer object is returned via select_quantizer
 * that hides the template mess.
 ********************************************************************/

/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

#include <faiss/impl/scalar_quantizer/codecs.h>

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

#include <faiss/impl/scalar_quantizer/quantizers.h>

/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object. The data fields just encapsulate
 * an accumulator.
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

#include <faiss/impl/scalar_quantizer/distance_computers.h>

/*******************************************************************
 * InvertedListScanner: scans series of codes and keeps the best ones
 *******************************************************************/

#include <faiss/impl/scalar_quantizer/scanners.h>

namespace faiss {
using namespace scalar_quantizer;

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
    }
}

void ScalarQuantizer::train(size_t n, const float* x) {
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
                    d,
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
    }
}

ScalarQuantizer::SQuantizer* ScalarQuantizer::select_quantizer() const {
    // here we can't just dispatch because the SIMD code works only on certain
    // vector sizes
#ifdef COMPILE_SIMD_AVX512
    if (d % 16 == 0 && SIMDConfig::level >= SIMDLevel::AVX512F) {
        return select_quantizer_1<SIMDLevel::AVX512F>(qtype, d, trained);
    } else
#endif
#ifdef COMPILE_SIMD_AVX2
            if (d % 8 == 0 && SIMDConfig::level >= SIMDLevel::AVX2) {
        return select_quantizer_1<SIMDLevel::AVX2>(qtype, d, trained);
    } else
#endif
        return select_quantizer_1<SIMDLevel::NONE>(qtype, d, trained);
}

void ScalarQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

    memset(codes, 0, code_size * n);
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++)
        squant->encode_vector(x + i * d, codes + i * code_size);
}

void ScalarQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

#pragma omp parallel for
    for (int64_t i = 0; i < n; i++)
        squant->decode_vector(codes + i * code_size, x + i * d);
}

SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
#ifdef COMPILE_SIMD_AVX512
    if (d % 16 == 0 && SIMDConfig::level >= SIMDLevel::AVX512F) {
        return select_distance_computer_1<SIMDLevel::AVX512F>(
                metric, qtype, d, trained);
    } else
#endif
#ifdef COMPILE_SIMD_AVX2
            if (d % 8 == 0 && SIMDConfig::level >= SIMDLevel::AVX2) {
        return select_distance_computer_1<SIMDLevel::AVX2>(
                metric, qtype, d, trained);
    } else
#endif
        return select_distance_computer_1<SIMDLevel::NONE>(
                metric, qtype, d, trained);
}

/*******************************************************************
 * IndexScalarQuantizer/IndexIVFScalarQuantizer scanner object
 *
 * It is an InvertedListScanner, but is designed to work with
 * IndexScalarQuantizer as well.
 ********************************************************************/

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) const {
#ifdef COMPILE_SIMD_AVX512
    if (d % 16 == 0 && SIMDConfig::level >= SIMDLevel::AVX512F) {
        return sel0_InvertedListScanner<SIMDLevel::AVX512F>(
                mt, this, quantizer, store_pairs, sel, by_residual);
    } else
#endif
#ifdef COMPILE_SIMD_AVX2
            if (d % 8 == 0 && SIMDConfig::level >= SIMDLevel::AVX2) {
        return sel0_InvertedListScanner<SIMDLevel::AVX2>(
                mt, this, quantizer, store_pairs, sel, by_residual);
    } else
#endif
        return sel0_InvertedListScanner<SIMDLevel::NONE>(
                mt, this, quantizer, store_pairs, sel, by_residual);
}

} // namespace faiss
