/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

namespace scalar_quantizer {

/*******************************************************************
 * Similarities: accumulates the element-wise similarities
 *******************************************************************/

template <SIMDLevel>
struct SimilarityL2 {};

template <SIMDLevel>
struct SimilarityIP {};

template <>
struct SimilarityL2<SIMDLevel::NONE> {
    static constexpr SIMDLevel SIMD_LEVEL = SIMDLevel::NONE;
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}

    /******* scalar accumulator *******/

    float accu;

    FAISS_ALWAYS_INLINE void begin() {
        accu = 0;
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_component(float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    FAISS_ALWAYS_INLINE void add_component_2(float x1, float x2) {
        float tmp = x1 - x2;
        accu += tmp * tmp;
    }

    FAISS_ALWAYS_INLINE float result() {
        return accu;
    }
};

template <>
struct SimilarityIP<SIMDLevel::NONE> {
    static constexpr int simdwidth = 1;
    static constexpr SIMDLevel SIMD_LEVEL = SIMDLevel::NONE;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;
    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    FAISS_ALWAYS_INLINE void begin() {
        accu = 0;
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_component(float x) {
        accu += *yi++ * x;
    }

    FAISS_ALWAYS_INLINE void add_component_2(float x1, float x2) {
        accu += x1 * x2;
    }

    FAISS_ALWAYS_INLINE float result() {
        return accu;
    }
};

/*******************************************************************
 * Distance computers: compute distances between a query and a code
 *******************************************************************/

using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

template <class Quantizer, class Similarity, SIMDLevel level>
struct DCTemplate : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
            sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, SIMDLevel>
struct DistanceComputerByte : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

/*******************************************************************
 * select_distance_computer: runtime selection of template
 * specialization
 *******************************************************************/

template <class Sim>
SQDistanceComputer* select_distance_computer(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr SIMDLevel SL = Sim::SIMD_LEVEL;
    constexpr QScaling NU = QScaling::NON_UNIFORM;
    constexpr QScaling U = QScaling::UNIFORM;
    switch (qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<QuantizerT<Codec8bit<SL>, U, SL>, Sim, SL>(
                    d, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<QuantizerT<Codec4bit<SL>, U, SL>, Sim, SL>(
                    d, trained);

        case ScalarQuantizer::QT_8bit:
            return new DCTemplate<QuantizerT<Codec8bit<SL>, NU, SL>, Sim, SL>(
                    d, trained);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate<QuantizerT<Codec6bit<SL>, NU, SL>, Sim, SL>(
                    d, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<QuantizerT<Codec4bit<SL>, NU, SL>, Sim, SL>(
                    d, trained);

        case ScalarQuantizer::QT_fp16:
            return new DCTemplate<QuantizerFP16<SL>, Sim, SL>(d, trained);

        case ScalarQuantizer::QT_bf16:
            return new DCTemplate<QuantizerBF16<SL>, Sim, SL>(d, trained);

        case ScalarQuantizer::QT_8bit_direct:
            return new DCTemplate<Quantizer8bitDirect<SL>, Sim, SL>(d, trained);
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate<Quantizer8bitDirectSigned<SL>, Sim, SL>(
                    d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <SIMDLevel SL>
SQDistanceComputer* select_distance_computer_1(
        MetricType metric_type,
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    if (metric_type == METRIC_L2) {
        return select_distance_computer<SimilarityL2<SL>>(qtype, d, trained);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return select_distance_computer<SimilarityIP<SL>>(qtype, d, trained);
    } else {
        FAISS_THROW_MSG("unsuppored metric type");
    }
}

// prevent implicit instantiation of the template
extern template SQDistanceComputer* select_distance_computer_1<SIMDLevel::AVX2>(
        MetricType metric_type,
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

extern template SQDistanceComputer* select_distance_computer_1<
        SIMDLevel::AVX512>(
        MetricType metric_type,
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

} // namespace scalar_quantizer
} // namespace faiss
