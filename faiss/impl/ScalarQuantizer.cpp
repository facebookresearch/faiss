/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <cstdio>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/simdlib.h>

#include <faiss/impl/scalar_quantizer/training.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/expanded_scanners.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>
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

#if defined(__AVX512F__) && defined(__F16C__)
#define USE_AVX512_F16C
#elif defined(__AVX2__)
#ifdef __F16C__
#define USE_F16C
#else
#warning \
        "Cannot enable AVX optimizations in scalar quantizer if -mf16c is not set as well"
#endif
#endif

#if defined(__aarch64__)
#if defined(__GNUC__) && __GNUC__ < 8
#warning \
        "Cannot enable NEON optimizations in scalar quantizer if the compiler is GCC<8"
#else
#define USE_NEON
#endif
#endif

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
 */

#include <faiss/impl/scalar_quantizer/similarities.h>

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

#include <faiss/impl/scalar_quantizer/distance_computers.h>

namespace faiss {

namespace scalar_quantizer {

using QuantizerType = ScalarQuantizer::QuantizerType;
using RangeStat = ScalarQuantizer::RangeStat;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * select_distance_computer: runtime selection of template
 * specialization
 *******************************************************************/

template <class Sim>
SQDistanceComputer* select_distance_computer(
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
#if defined(__AVX512F__)
            if (d % 32 == 0) {
                return new DistanceComputerByte<Sim, SL>(int(d), trained);
            } else
#elif defined(__AVX2__)
            if (d % 16 == 0) {
                return new DistanceComputerByte<Sim, SL>(
                        static_cast<int>(d), trained);
            } else
#endif
            {
                return new DCTemplate<Quantizer8bitDirect<SL>, Sim, SL>(
                        d, trained);
            }
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate<Quantizer8bitDirectSigned<SL>, Sim, SL>(
                    d, trained);
        default:
            FAISS_THROW_MSG("unknown qtype");
    }
}

template <SIMDLevel SL>
ScalarQuantizer::SQuantizer* select_quantizer_1(
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

} // namespace scalar_quantizer

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
        default:
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
#if defined(USE_AVX512_F16C)
    if (d % 16 == 0) {
        return select_quantizer_1<SIMDLevel::AVX512>(qtype, d, trained);
    } else
#elif defined(USE_F16C)
    if (d % 8 == 0) {
        return select_quantizer_1<SIMDLevel::AVX2>(qtype, d, trained);
    } else
#elif defined(USE_NEON)
    if (d % 8 == 0) {
        return select_quantizer_1<SIMDLevel::ARM_NEON>(qtype, d, trained);
    } else
#endif
    {
        return select_quantizer_1<SIMDLevel::NONE>(qtype, d, trained);
    }
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

SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
#if defined(USE_AVX512_F16C)
    if (d % 16 == 0) {
        if (metric == METRIC_L2) {
            return select_distance_computer<SimilarityL2<SIMDLevel::AVX512>>(
                    qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<SIMDLevel::AVX512>>(
                    qtype, d, trained);
        }
    } else
#elif defined(USE_F16C)
    if (d % 8 == 0) {
        if (metric == METRIC_L2) {
            return select_distance_computer<SimilarityL2<SIMDLevel::AVX2>>(
                    qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<SIMDLevel::AVX2>>(
                    qtype, d, trained);
        }
    } else
#elif defined(USE_NEON)
    if (d % 8 == 0) {
        if (metric == METRIC_L2) {
            return select_distance_computer<SimilarityL2<SIMDLevel::ARM_NEON>>(
                    qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<SIMDLevel::ARM_NEON>>(
                    qtype, d, trained);
        }
    } else
#endif
    {
        if (metric == METRIC_L2) {
            return select_distance_computer<SimilarityL2<SIMDLevel::NONE>>(
                    qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<SIMDLevel::NONE>>(
                    qtype, d, trained);
        }
    }
}

/*******************************************************************
 * IndexScalarQuantizer/IndexIVFScalarQuantizer scanner object
 *
 * It is an InvertedListScanner, but is designed to work with
 * IndexScalarQuantizer as well.
 ********************************************************************/

namespace {

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

    // redefining the scan_codes allows to inline the distance_to_code
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
            // shift of x_in wrt centroid
            quantizer->compute_residual(x, tmp.data(), list_no);
            dc.set_query(tmp.data());
        } else {
            dc.set_query(x);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc.query_to_code(code);
    }

    // redefining the scan_codes allows to inline the distance_to_code
    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return run_scan_codes_fix_C<CMax<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

} // anonymous namespace

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) const {
    // this maps the runtime selection of the scanner by using a series of
    // templatized lambdas
    auto scan = [=, this]<class DCClass>() -> InvertedListScanner* {
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
            [&,
             this]<SIMDLevel SL, class Similarity>() -> InvertedListScanner* {
        switch (qtype) {
            case QT_8bit_uniform:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec8bit,
                                QuantizerTemplateScaling::UNIFORM,
                                SL>,
                        Similarity,
                        SL>>();
            case QT_4bit_uniform:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec4bit,
                                QuantizerTemplateScaling::UNIFORM,
                                SL>,
                        Similarity,
                        SL>>();
            case QT_8bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec8bit,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL>,
                        Similarity,
                        SL>>();
            case QT_4bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec4bit,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL>,
                        Similarity,
                        SL>>();
            case QT_6bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec6bit,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL>,
                        Similarity,
                        SL>>();
            case QT_fp16:
                return scan.template
                operator()<DCTemplate<QuantizerFP16<SL>, Similarity, SL>>();
            case QT_bf16:
                return scan.template
                operator()<DCTemplate<QuantizerBF16<SL>, Similarity, SL>>();
            case QT_8bit_direct:
#if defined(__AVX512F__)
                if (d % 32 == 0) {
                    return scan.template
                    operator()<DistanceComputerByte<Similarity, SL>>();
                }
#elif defined(__AVX2__)
                if (d % 16 == 0) {
                    return scan.template
                    operator()<DistanceComputerByte<Similarity, SL>>();
                }
#endif
                return scan.template operator()<
                        DCTemplate<Quantizer8bitDirect<SL>, Similarity, SL>>();
            case QT_8bit_direct_signed:
                return scan.template operator()<DCTemplate<
                        Quantizer8bitDirectSigned<SL>,
                        Similarity,
                        SL>>();
            default:
                FAISS_THROW_MSG("unknown qtype");
        }
    };

    auto select_by_simd = [&]<SIMDLevel SL>() {
        if (mt == METRIC_L2) {
            return select_by_simd_and_metric
                    .template operator()<SL, SimilarityL2<SL>>();
        } else if (mt == METRIC_INNER_PRODUCT) {
            return select_by_simd_and_metric
                    .template operator()<SL, SimilarityIP<SL>>();
        }
        FAISS_THROW_MSG("unsupported metric type");
    };

#if defined(USE_AVX512_F16C)
    if (d % 16 == 0) {
        return select_by_simd.template operator()<SIMDLevel::AVX512>();
    }
#elif defined(USE_F16C)
    if (d % 8 == 0) {
        return select_by_simd.template operator()<SIMDLevel::AVX2>();
    }
#elif defined(USE_NEON)
    if (d % 8 == 0) {
        return select_by_simd.template operator()<SIMDLevel::ARM_NEON>();
    }
#endif
    return select_by_simd.template operator()<SIMDLevel::NONE>();
}

} // namespace faiss
