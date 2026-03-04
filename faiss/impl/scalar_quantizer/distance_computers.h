/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/simdlib.h>
#include <faiss/utils/bf16.h>

namespace faiss {

namespace scalar_quantizer {

using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, SIMDLevel SL>
struct DCTemplate : SQDistanceComputer {};

#if defined(__AVX512BF16__)

// Fast path for QT_bf16 + IP on CPUs with AVX512_BF16.
//
// Key idea: quantize query to BF16 once in set_query(), then compute inner
// products using VDPBF16PS against BF16-coded vectors.
//
// Notes:
// - Only enabled when __AVX512BF16__ is available (e.g., -march=sapphirerapids).
// - Requires d % 32 == 0 to use dpbf16 cleanly (32 bf16 elements per op).
template <SIMDLevel SL>
struct DCBF16IPDpbf16 : SQDistanceComputer {
    using Sim = SimilarityIP<SL>;

    QuantizerBF16<SL> quant;
    std::vector<uint16_t> qbf16;

    DCBF16IPDpbf16(size_t d, const std::vector<float>& trained)
            : quant(d, trained), qbf16(d) {}

    void set_query(const float* x) final {
        q = x;
        // Match QuantizerBF16::encode_vector semantics (encode_bf16()).
        for (size_t i = 0; i < quant.d; i++) {
            qbf16[i] = encode_bf16(x[i]);
        }
    }

    FAISS_ALWAYS_INLINE float compute_code_ip_bf16(
            const uint16_t* a,
            const uint16_t* b) const {
        // d is expected to be multiple of 32 for this fast path.
        __m512 acc = _mm512_setzero_ps();
        for (size_t i = 0; i < quant.d; i += 32) {
            const __m512i va = _mm512_loadu_si512((const void*)(a + i));
            const __m512i vb = _mm512_loadu_si512((const void*)(b + i));
            const __m512bh bha = (__m512bh)va;
            const __m512bh bhb = (__m512bh)vb;
            acc = _mm512_dpbf16_ps(acc, bha, bhb);
        }
        return _mm512_reduce_add_ps(acc);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const auto* code1 = (const uint16_t*)(codes + i * code_size);
        const auto* code2 = (const uint16_t*)(codes + j * code_size);
        return compute_code_ip_bf16(code1, code2);
    }

    float query_to_code(const uint8_t* code) const final {
        const auto* c = (const uint16_t*)code;
        return compute_code_ip_bf16(qbf16.data(), c);
    }
};

#endif

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

template <class Similarity, SIMDLevel SL>
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
 * Selection function
 *******************************************************************/

template <SIMDLevel SL>
SQDistanceComputer* sq_select_distance_computer(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

} // namespace scalar_quantizer
} // namespace faiss
