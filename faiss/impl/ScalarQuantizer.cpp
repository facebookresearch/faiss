/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <omp.h>
#include <cstdio>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/simdlib.h>

#include <faiss/impl/scalar_quantizer/training.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include <faiss/IndexIVF.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
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

typedef ScalarQuantizer::QuantizerType QuantizerType;
typedef ScalarQuantizer::RangeStat RangeStat;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

template <int SIMDWIDTH>
ScalarQuantizer::SQuantizer* select_quantizer_1(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerTemplate<
                    Codec8bit,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_6bit:
            return new QuantizerTemplate<
                    Codec6bit,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerTemplate<
                    Codec4bit,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerTemplate<
                    Codec8bit,
                    QuantizerTemplateScaling::UNIFORM,
                    SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerTemplate<
                    Codec4bit,
                    QuantizerTemplateScaling::UNIFORM,
                    SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_fp16:
            return new QuantizerFP16<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_bf16:
            return new QuantizerBF16<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_8bit_direct:
            return new Quantizer8bitDirect<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned<SIMDWIDTH>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
}

/*******************************************************************
 * select_distance_computer: runtime selection of template
 * specialization
 *******************************************************************/

template <class Sim>
SQDistanceComputer* select_distance_computer(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr int SIMDWIDTH = Sim::simdwidth;
    switch (qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec8bit,
                            QuantizerTemplateScaling::UNIFORM,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec4bit,
                            QuantizerTemplateScaling::UNIFORM,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_8bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec8bit,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec6bit,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec4bit,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_fp16:
            return new DCTemplate<QuantizerFP16<SIMDWIDTH>, Sim, SIMDWIDTH>(
                    d, trained);

        case ScalarQuantizer::QT_bf16:
            return new DCTemplate<QuantizerBF16<SIMDWIDTH>, Sim, SIMDWIDTH>(
                    d, trained);

        case ScalarQuantizer::QT_8bit_direct:
#if defined(__AVX512F__)
            if (d % 32 == 0) {
                return new DistanceComputerByte<Sim, SIMDWIDTH>(d, trained);
            } else
#elif defined(__AVX2__)
            if (d % 16 == 0) {
                return new DistanceComputerByte<Sim, SIMDWIDTH>(d, trained);
            } else
#endif
            {
                return new DCTemplate<
                        Quantizer8bitDirect<SIMDWIDTH>,
                        Sim,
                        SIMDWIDTH>(d, trained);
            }
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate<
                    Quantizer8bitDirectSigned<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
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
#if defined(USE_AVX512_F16C)
    if (d % 16 == 0) {
        return select_quantizer_1<16>(qtype, d, trained);
    } else
#elif defined(USE_F16C) || defined(USE_NEON)
    if (d % 8 == 0) {
        return select_quantizer_1<8>(qtype, d, trained);
    } else
#endif
    {
        return select_quantizer_1<1>(qtype, d, trained);
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
            return select_distance_computer<SimilarityL2<16>>(
                    qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<16>>(
                    qtype, d, trained);
        }
    } else
#elif defined(USE_F16C) || defined(USE_NEON)
    if (d % 8 == 0) {
        if (metric == METRIC_L2) {
            return select_distance_computer<SimilarityL2<8>>(qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<8>>(qtype, d, trained);
        }
    } else
#endif
    {
        if (metric == METRIC_L2) {
            return select_distance_computer<SimilarityL2<1>>(qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<1>>(qtype, d, trained);
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

template <class DCClass, int use_sel>
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
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;

        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            float accu = accu0 + dc.query_to_code(codes);

            if (accu > simi[0]) {
                int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                minheap_replace_top(k, simi, idxi, accu, id);
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            float accu = accu0 + dc.query_to_code(codes);
            if (accu > radius) {
                int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                res.add(accu, id);
            }
        }
    }
};

/* use_sel = 0: don't check selector
 * = 1: check on ids[j]
 * = 2: check in j directly (normally ids is nullptr and store_pairs)
 */
template <class DCClass, int use_sel>
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

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            float dis = dc.query_to_code(codes);

            if (dis < simi[0]) {
                int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            float dis = dc.query_to_code(codes);
            if (dis < radius) {
                int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                res.add(dis, id);
            }
        }
    }
};

template <class DCClass, int use_sel>
InvertedListScanner* sel3_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (DCClass::Sim::metric_type == METRIC_L2) {
        return new IVFSQScannerL2<DCClass, use_sel>(
                sq->d,
                sq->trained,
                sq->code_size,
                quantizer,
                store_pairs,
                sel,
                r);
    } else if (DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
        return new IVFSQScannerIP<DCClass, use_sel>(
                sq->d, sq->trained, sq->code_size, store_pairs, sel, r);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

template <class DCClass>
InvertedListScanner* sel2_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (sel) {
        if (store_pairs) {
            return sel3_InvertedListScanner<DCClass, 2>(
                    sq, quantizer, store_pairs, sel, r);
        } else {
            return sel3_InvertedListScanner<DCClass, 1>(
                    sq, quantizer, store_pairs, sel, r);
        }
    } else {
        return sel3_InvertedListScanner<DCClass, 0>(
                sq, quantizer, store_pairs, sel, r);
    }
}

template <class Similarity, class Codec, QuantizerTemplateScaling SCALING>
InvertedListScanner* sel12_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate<Codec, SCALING, SIMDWIDTH>;
    using DCClass = DCTemplate<QuantizerClass, Similarity, SIMDWIDTH>;
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity>
InvertedListScanner* sel1_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    switch (sq->qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return sel12_InvertedListScanner<
                    Similarity,
                    Codec8bit,
                    QuantizerTemplateScaling::UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit_uniform:
            return sel12_InvertedListScanner<
                    Similarity,
                    Codec4bit,
                    QuantizerTemplateScaling::UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit:
            return sel12_InvertedListScanner<
                    Similarity,
                    Codec8bit,
                    QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit:
            return sel12_InvertedListScanner<
                    Similarity,
                    Codec4bit,
                    QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_6bit:
            return sel12_InvertedListScanner<
                    Similarity,
                    Codec6bit,
                    QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_fp16:
            return sel2_InvertedListScanner<DCTemplate<
                    QuantizerFP16<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_bf16:
            return sel2_InvertedListScanner<DCTemplate<
                    QuantizerBF16<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit_direct:
#if defined(__AVX512F__)
            if (sq->d % 32 == 0) {
                return sel2_InvertedListScanner<
                        DistanceComputerByte<Similarity, SIMDWIDTH>>(
                        sq, quantizer, store_pairs, sel, r);
            } else
#elif defined(__AVX2__)
            if (sq->d % 16 == 0) {
                return sel2_InvertedListScanner<
                        DistanceComputerByte<Similarity, SIMDWIDTH>>(
                        sq, quantizer, store_pairs, sel, r);
            } else
#endif
            {
                return sel2_InvertedListScanner<DCTemplate<
                        Quantizer8bitDirect<SIMDWIDTH>,
                        Similarity,
                        SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
            }
        case ScalarQuantizer::QT_8bit_direct_signed:
            return sel2_InvertedListScanner<DCTemplate<
                    Quantizer8bitDirectSigned<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
    }

    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <int SIMDWIDTH>
InvertedListScanner* sel0_InvertedListScanner(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner<SimilarityL2<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner<SimilarityIP<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

} // anonymous namespace

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) const {
#if defined(USE_AVX512_F16C)
    if (d % 16 == 0) {
        return sel0_InvertedListScanner<16>(
                mt, this, quantizer, store_pairs, sel, by_residual);
    } else
#elif defined(USE_F16C) || defined(USE_NEON)
    if (d % 8 == 0) {
        return sel0_InvertedListScanner<8>(
                mt, this, quantizer, store_pairs, sel, by_residual);
    } else
#endif
    {
        return sel0_InvertedListScanner<1>(
                mt, this, quantizer, store_pairs, sel, by_residual);
    }
}

} // namespace faiss
