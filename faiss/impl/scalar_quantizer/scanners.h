/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

namespace scalar_quantizer {

/*******************************************************************
 * IndexScalarQuantizer/IndexIVFScalarQuantizer scanner object
 *
 * It is an InvertedListScanner, but is designed to work with
 * IndexScalarQuantizer as well.
 ********************************************************************/

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

/* Select the right implementation by dispatching to templatized versions */

template <class DCClass, int use_sel>
InvertedListScanner* sel3_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (DCClass::Sim::metric_type == METRIC_L2) {
        return new IVFSQScannerL2<DCClass, use_sel>(
                static_cast<int>(sq->d),
                sq->trained,
                sq->code_size,
                quantizer,
                store_pairs,
                sel,
                r);
    } else if (DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
        return new IVFSQScannerIP<DCClass, use_sel>(
                static_cast<int>(sq->d),
                sq->trained,
                sq->code_size,
                store_pairs,
                sel,
                r);
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

template <class Similarity, class Codec, QScaling SCALING>
InvertedListScanner* sel12_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr SIMDLevel SL = Similarity::SIMD_LEVEL;
    using QuantizerClass = QuantizerT<Codec, SCALING, SL>;
    using DCClass = DCTemplate<QuantizerClass, Similarity, SL>;
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Sim>
InvertedListScanner* sel1_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr SIMDLevel SL = Sim::SIMD_LEVEL;
    constexpr QScaling NU = QScaling::NON_UNIFORM;
    constexpr QScaling U = QScaling::UNIFORM;

    switch (sq->qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return sel12_InvertedListScanner<Sim, Codec8bit<SL>, U>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit_uniform:
            return sel12_InvertedListScanner<Sim, Codec4bit<SL>, U>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit:
            return sel12_InvertedListScanner<Sim, Codec8bit<SL>, NU>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit:
            return sel12_InvertedListScanner<Sim, Codec4bit<SL>, NU>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_6bit:
            return sel12_InvertedListScanner<Sim, Codec6bit<SL>, NU>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_fp16:
            return sel2_InvertedListScanner<
                    DCTemplate<QuantizerFP16<SL>, Sim, SL>>(
                    sq, quantizer, store_pairs, sel, r);

        case ScalarQuantizer::QT_bf16:
            return sel2_InvertedListScanner<
                    DCTemplate<QuantizerBF16<SL>, Sim, SL>>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit_direct:
            return sel2_InvertedListScanner<
                    DCTemplate<Quantizer8bitDirect<SL>, Sim, SL>>(
                    sq, quantizer, store_pairs, sel, r);

        case ScalarQuantizer::QT_8bit_direct_signed:
            return sel2_InvertedListScanner<
                    DCTemplate<Quantizer8bitDirectSigned<SL>, Sim, SL>>(
                    sq, quantizer, store_pairs, sel, r);
        default:
            FAISS_THROW_MSG("unknown qtype");
            return nullptr;
    }
}

template <SIMDLevel SL>
InvertedListScanner* sel0_InvertedListScanner(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner<SimilarityL2<SL>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner<SimilarityIP<SL>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

// prevent implicit instantiation of the template when there are
// SIMD optimized versions...
extern template InvertedListScanner* sel0_InvertedListScanner<SIMDLevel::AVX2>(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual);

extern template InvertedListScanner* sel0_InvertedListScanner<
        SIMDLevel::AVX512>(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual);

} // namespace scalar_quantizer
} // namespace faiss
