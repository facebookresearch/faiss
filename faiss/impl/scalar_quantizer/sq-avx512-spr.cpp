/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512_SPR

#include <immintrin.h>

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

#include <faiss/impl/scalar_quantizer/sq-avx512-impl.h>

namespace faiss {
namespace scalar_quantizer {

/**********************************************************
 * Codecs — inherit AVX512 implementations
 **********************************************************/

template <>
struct Codec8bit<SIMDLevel::AVX512_SPR> : Codec8bit<SIMDLevel::AVX512> {};

template <>
struct Codec4bit<SIMDLevel::AVX512_SPR> : Codec4bit<SIMDLevel::AVX512> {};

template <>
struct Codec6bit<SIMDLevel::AVX512_SPR> : Codec6bit<SIMDLevel::AVX512> {};

/**********************************************************
 * Quantizers — inherit AVX512 implementations
 **********************************************************/

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::AVX512_SPR>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::AVX512> {
    using QuantizerTemplate<
            Codec,
            QuantizerTemplateScaling::UNIFORM,
            SIMDLevel::AVX512>::QuantizerTemplate;
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::AVX512_SPR>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::NON_UNIFORM,
                  SIMDLevel::AVX512> {
    using QuantizerTemplate<
            Codec,
            QuantizerTemplateScaling::NON_UNIFORM,
            SIMDLevel::AVX512>::QuantizerTemplate;
};

template <>
struct QuantizerFP16<SIMDLevel::AVX512_SPR> : QuantizerFP16<SIMDLevel::AVX512> {
    using QuantizerFP16<SIMDLevel::AVX512>::QuantizerFP16;
};

template <>
struct QuantizerBF16<SIMDLevel::AVX512_SPR> : QuantizerBF16<SIMDLevel::AVX512> {
    using QuantizerBF16<SIMDLevel::AVX512>::QuantizerBF16;

    void encode_vector(const float* x, uint8_t* code) const override {
        encode_bf16_simd(x, (uint16_t*)code, this->d);
    }

    void decode_vector(const uint8_t* code, float* x) const override {
        decode_bf16_simd((const uint16_t*)code, x, this->d);
    }
};

template <>
struct Quantizer8bitDirect<SIMDLevel::AVX512_SPR>
        : Quantizer8bitDirect<SIMDLevel::AVX512> {
    using Quantizer8bitDirect<SIMDLevel::AVX512>::Quantizer8bitDirect;
};

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::AVX512_SPR>
        : Quantizer8bitDirectSigned<SIMDLevel::AVX512> {
    using Quantizer8bitDirectSigned<
            SIMDLevel::AVX512>::Quantizer8bitDirectSigned;
};

/**********************************************************
 * Similarities — inherit AVX512 implementations
 **********************************************************/

template <>
struct SimilarityL2<SIMDLevel::AVX512_SPR> : SimilarityL2<SIMDLevel::AVX512> {
    using SimilarityL2<SIMDLevel::AVX512>::SimilarityL2;
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX512_SPR;
};

template <>
struct SimilarityIP<SIMDLevel::AVX512_SPR> : SimilarityIP<SIMDLevel::AVX512> {
    using SimilarityIP<SIMDLevel::AVX512>::SimilarityIP;
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX512_SPR;
};

/**********************************************************
 * Generic DCTemplate and DistanceComputerByte
 *
 * Delegate to AVX512 implementations.
 **********************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512_SPR>
        : DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512> {
    using DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512>::DCTemplate;
};

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::AVX512_SPR>
        : DistanceComputerByte<Similarity, SIMDLevel::AVX512> {
    using DistanceComputerByte<Similarity, SIMDLevel::AVX512>::
            DistanceComputerByte;
};

/**********************************************************
 * BF16 native distance helpers using VDPBF16PS
 **********************************************************/

static FAISS_ALWAYS_INLINE float bf16_vdpbf16ps(
        const uint16_t* a,
        const uint16_t* b,
        size_t d) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= d; i += 32) {
        __m512bh va = (__m512bh)_mm512_loadu_epi16(a + i);
        __m512bh vb = (__m512bh)_mm512_loadu_epi16(b + i);
        acc = _mm512_dpbf16_ps(acc, va, vb);
    }
    // Remainder: 16 elements (d % 16 == 0 but may not be % 32)
    if (i < d) {
        __m256i a_lo = _mm256_loadu_epi16(a + i);
        __m256i b_lo = _mm256_loadu_epi16(b + i);
        __m512bh va =
                (__m512bh)_mm512_inserti64x4(_mm512_setzero_si512(), a_lo, 0);
        __m512bh vb =
                (__m512bh)_mm512_inserti64x4(_mm512_setzero_si512(), b_lo, 0);
        acc = _mm512_dpbf16_ps(acc, va, vb);
    }
    return _mm512_reduce_add_ps(acc);
}

static FAISS_ALWAYS_INLINE float bf16_L2_asymmetric(
        const uint16_t* query_bf16,
        const uint16_t* code,
        size_t d) {
    __m512 acc_qc = _mm512_setzero_ps();
    __m512 acc_cc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= d; i += 32) {
        __m512bh vq = (__m512bh)_mm512_loadu_epi16(query_bf16 + i);
        __m512bh vc = (__m512bh)_mm512_loadu_epi16(code + i);
        acc_qc = _mm512_dpbf16_ps(acc_qc, vq, vc);
        acc_cc = _mm512_dpbf16_ps(acc_cc, vc, vc);
    }
    if (i < d) {
        __m256i q_lo = _mm256_loadu_epi16(query_bf16 + i);
        __m256i c_lo = _mm256_loadu_epi16(code + i);
        __m512bh vq =
                (__m512bh)_mm512_inserti64x4(_mm512_setzero_si512(), q_lo, 0);
        __m512bh vc =
                (__m512bh)_mm512_inserti64x4(_mm512_setzero_si512(), c_lo, 0);
        acc_qc = _mm512_dpbf16_ps(acc_qc, vq, vc);
        acc_cc = _mm512_dpbf16_ps(acc_cc, vc, vc);
    }
    float dot_qc = _mm512_reduce_add_ps(acc_qc);
    float norm_c = _mm512_reduce_add_ps(acc_cc);
    return -2.0f * dot_qc + norm_c;
}

static FAISS_ALWAYS_INLINE float bf16_L2_symmetric(
        const uint16_t* a,
        const uint16_t* b,
        size_t d) {
    __m512 acc_ab = _mm512_setzero_ps();
    __m512 acc_aa = _mm512_setzero_ps();
    __m512 acc_bb = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= d; i += 32) {
        __m512bh va = (__m512bh)_mm512_loadu_epi16(a + i);
        __m512bh vb = (__m512bh)_mm512_loadu_epi16(b + i);
        acc_ab = _mm512_dpbf16_ps(acc_ab, va, vb);
        acc_aa = _mm512_dpbf16_ps(acc_aa, va, va);
        acc_bb = _mm512_dpbf16_ps(acc_bb, vb, vb);
    }
    if (i < d) {
        __m256i a_lo = _mm256_loadu_epi16(a + i);
        __m256i b_lo = _mm256_loadu_epi16(b + i);
        __m512bh va =
                (__m512bh)_mm512_inserti64x4(_mm512_setzero_si512(), a_lo, 0);
        __m512bh vb =
                (__m512bh)_mm512_inserti64x4(_mm512_setzero_si512(), b_lo, 0);
        acc_ab = _mm512_dpbf16_ps(acc_ab, va, vb);
        acc_aa = _mm512_dpbf16_ps(acc_aa, va, va);
        acc_bb = _mm512_dpbf16_ps(acc_bb, vb, vb);
    }
    return _mm512_reduce_add_ps(acc_aa) - 2.0f * _mm512_reduce_add_ps(acc_ab) +
            _mm512_reduce_add_ps(acc_bb);
}

/**********************************************************
 * BF16 + Inner Product distance computer (SPR)
 **********************************************************/

struct DCBF16_IP : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::AVX512_SPR>;

    size_t d;
    std::vector<uint16_t> query_bf16;

    DCBF16_IP(size_t d, const std::vector<float>&) : d(d), query_bf16(d) {}

    void set_query(const float* x) final {
        q = x;
        encode_bf16_simd(x, query_bf16.data(), d);
    }

    float query_to_code(const uint8_t* code) const final {
        return bf16_vdpbf16ps(query_bf16.data(), (const uint16_t*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return bf16_vdpbf16ps(
                (const uint16_t*)(codes + i * code_size),
                (const uint16_t*)(codes + j * code_size),
                d);
    }
};

/**********************************************************
 * BF16 + L2 distance computer (SPR)
 **********************************************************/

struct DCBF16_L2 : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::AVX512_SPR>;

    size_t d;
    std::vector<uint16_t> query_bf16;
    float query_norm_sq;

    DCBF16_L2(size_t d, const std::vector<float>&)
            : d(d), query_bf16(d), query_norm_sq(0) {}

    void set_query(const float* x) final {
        q = x;
        encode_bf16_simd(x, query_bf16.data(), d);
        query_norm_sq = bf16_vdpbf16ps(query_bf16.data(), query_bf16.data(), d);
    }

    float query_to_code(const uint8_t* code) const final {
        return query_norm_sq +
                bf16_L2_asymmetric(query_bf16.data(), (const uint16_t*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return bf16_L2_symmetric(
                (const uint16_t*)(codes + i * code_size),
                (const uint16_t*)(codes + j * code_size),
                d);
    }
};

template <>
struct DCTemplate<
        QuantizerBF16<SIMDLevel::AVX512_SPR>,
        SimilarityIP<SIMDLevel::AVX512_SPR>,
        SIMDLevel::AVX512_SPR> : DCBF16_IP {
    using Sim = SimilarityIP<SIMDLevel::AVX512_SPR>;
    using DCBF16_IP::DCBF16_IP;
};

template <>
struct DCTemplate<
        QuantizerBF16<SIMDLevel::AVX512_SPR>,
        SimilarityL2<SIMDLevel::AVX512_SPR>,
        SIMDLevel::AVX512_SPR> : DCBF16_L2 {
    using Sim = SimilarityL2<SIMDLevel::AVX512_SPR>;
    using DCBF16_L2::DCBF16_L2;
};

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512_SPR
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#endif // COMPILE_SIMD_AVX512_SPR
