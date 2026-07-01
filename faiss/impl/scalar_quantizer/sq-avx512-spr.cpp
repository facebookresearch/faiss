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
#include <faiss/impl/simdlib/simdlib_avx512.h>

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
 * TurboQuant MSE — inherit AVX512 implementations
 **********************************************************/

template <int NBits>
struct QuantizerTurboQuantMSE<NBits, SIMDLevel::AVX512_SPR>
        : QuantizerTurboQuantMSE<NBits, SIMDLevel::AVX512> {
    using QuantizerTurboQuantMSE<NBits, SIMDLevel::AVX512>::
            QuantizerTurboQuantMSE;
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
 * Generic DCTemplate — delegate to AVX512 implementations
 **********************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512_SPR>
        : DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512> {
    using DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512>::DCTemplate;
};

/**********************************************************
 * DistanceComputerByte: AVX512-VNNI
 *
 * Uses _mm512_dpbusd_epi32 to compute dot products of uint8 vectors
 * at 64 bytes per instruction (4x throughput vs generic AVX512).
 **********************************************************/

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::AVX512_SPR>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        if constexpr (Sim::metric_type == METRIC_INNER_PRODUCT) {
            __m512i accu = _mm512_setzero_si512();
            int i = 0;
            for (; i + 64 <= d; i += 64) {
                __m512i c1 = _mm512_loadu_si512(code1 + i);
                __m512i c2 = _mm512_loadu_si512(code2 + i);

                __m512i c2_signed = _mm512_sub_epi8(c2, _mm512_set1_epi8(-128));
                accu = _mm512_dpbusd_epi32(accu, c1, c2_signed);
            }
            int32_t sum_c1 = 0;
            for (int j = 0; j < i; j++) {
                sum_c1 += code1[j];
            }
            int32_t result = _mm512_reduce_add_epi32(accu) + 128 * sum_c1;

            for (; i < d; i++) {
                result += int(code1[i]) * code2[i];
            }
            return result;
        } else {
            __m512i accu = _mm512_setzero_si512();
            int i = 0;
            for (; i + 64 <= d; i += 64) {
                __m256i c1_lo = _mm256_loadu_si256((const __m256i*)(code1 + i));
                __m256i c2_lo = _mm256_loadu_si256((const __m256i*)(code2 + i));
                __m256i c1_hi =
                        _mm256_loadu_si256((const __m256i*)(code1 + i + 32));
                __m256i c2_hi =
                        _mm256_loadu_si256((const __m256i*)(code2 + i + 32));

                __m512i c1_16_lo = _mm512_cvtepu8_epi16(c1_lo);
                __m512i c2_16_lo = _mm512_cvtepu8_epi16(c2_lo);
                __m512i diff_lo = _mm512_sub_epi16(c1_16_lo, c2_16_lo);

                __m512i c1_16_hi = _mm512_cvtepu8_epi16(c1_hi);
                __m512i c2_16_hi = _mm512_cvtepu8_epi16(c2_hi);
                __m512i diff_hi = _mm512_sub_epi16(c1_16_hi, c2_16_hi);

                accu = _mm512_add_epi32(
                        accu, _mm512_madd_epi16(diff_lo, diff_lo));
                accu = _mm512_add_epi32(
                        accu, _mm512_madd_epi16(diff_hi, diff_hi));
            }
            for (; i + 32 <= d; i += 32) {
                __m256i c1v = _mm256_loadu_si256((const __m256i*)(code1 + i));
                __m256i c2v = _mm256_loadu_si256((const __m256i*)(code2 + i));
                __m512i c1_16 = _mm512_cvtepu8_epi16(c1v);
                __m512i c2_16 = _mm512_cvtepu8_epi16(c2v);
                __m512i diff = _mm512_sub_epi16(c1_16, c2_16);
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(diff, diff));
            }
            int32_t result = _mm512_reduce_add_epi32(accu);

            for (; i < d; i++) {
                int diff = int(code1[i]) - code2[i];
                result += diff * diff;
            }
            return result;
        }
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

/**********************************************************
 * DistanceComputerByteSigned: AVX512_SPR specialization for
 * QT_8bit_direct_signed.
 *
 * Storage convention (see Quantizer8bitDirectSigned):
 *     stored_byte = value + 128,  i.e.  value = stored_byte - 128
 *
 * L2: (s_a - 128) - (s_b - 128) == s_a - s_b, so the unsigned
 *     widened-madd kernel is bit-exact for the signed variant.
 *
 * IP: (s_a - 128) * (s_b - 128)
 *       = s_a*s_b - 128*(s_a + s_b) + 16384
 *     summed over d components:
 *       sum_ip_signed = sum_ip_unsigned
 *                       - 128 * (sum(s_a) + sum(s_b))
 *                       + 16384 * d
 *     sum(s_a), sum(s_b) are cheap via _mm512_sad_epu8 against zero.
 **********************************************************/

template <class Similarity>
struct DistanceComputerByteSigned<Similarity, SIMDLevel::AVX512_SPR>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByteSigned(int d, const std::vector<float>&)
            : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        if constexpr (Sim::metric_type == METRIC_INNER_PRODUCT) {
            __m512i accu = _mm512_setzero_si512();
            __m512i sum_a = _mm512_setzero_si512();
            __m512i sum_b = _mm512_setzero_si512();
            const __m512i zero = _mm512_setzero_si512();
            const __m512i bias = _mm512_set1_epi8(-128);

            int i = 0;
            for (; i + 64 <= d; i += 64) {
                __m512i c1 = _mm512_loadu_si512(code1 + i);
                __m512i c2 = _mm512_loadu_si512(code2 + i);

                sum_a = _mm512_add_epi64(sum_a, _mm512_sad_epu8(c1, zero));
                sum_b = _mm512_add_epi64(sum_b, _mm512_sad_epu8(c2, zero));

                __m512i c2_signed = _mm512_sub_epi8(c2, bias);
                accu = _mm512_dpbusd_epi32(accu, c1, c2_signed);
            }
            int32_t sum_c1_for_bias = int32_t(_mm512_reduce_add_epi64(sum_a));
            int32_t result =
                    _mm512_reduce_add_epi32(accu) + 128 * sum_c1_for_bias;

            int32_t tail_sum_a = 0, tail_sum_b = 0;
            for (; i < d; ++i) {
                result += int32_t(code1[i]) * int32_t(code2[i]);
                tail_sum_a += code1[i];
                tail_sum_b += code2[i];
            }

            int32_t total_sum_a = sum_c1_for_bias + tail_sum_a;
            int32_t total_sum_b =
                    int32_t(_mm512_reduce_add_epi64(sum_b)) + tail_sum_b;
            result -= 128 * (total_sum_a + total_sum_b);
            result += 16384 * d;
            return result;
        } else {
            __m512i accu = _mm512_setzero_si512();
            int i = 0;
            for (; i + 64 <= d; i += 64) {
                __m256i c1_lo = _mm256_loadu_si256((const __m256i*)(code1 + i));
                __m256i c2_lo = _mm256_loadu_si256((const __m256i*)(code2 + i));
                __m256i c1_hi =
                        _mm256_loadu_si256((const __m256i*)(code1 + i + 32));
                __m256i c2_hi =
                        _mm256_loadu_si256((const __m256i*)(code2 + i + 32));
                __m512i diff_lo = _mm512_sub_epi16(
                        _mm512_cvtepu8_epi16(c1_lo),
                        _mm512_cvtepu8_epi16(c2_lo));
                __m512i diff_hi = _mm512_sub_epi16(
                        _mm512_cvtepu8_epi16(c1_hi),
                        _mm512_cvtepu8_epi16(c2_hi));
                accu = _mm512_add_epi32(
                        accu, _mm512_madd_epi16(diff_lo, diff_lo));
                accu = _mm512_add_epi32(
                        accu, _mm512_madd_epi16(diff_hi, diff_hi));
            }
            for (; i + 32 <= d; i += 32) {
                __m256i c1v = _mm256_loadu_si256((const __m256i*)(code1 + i));
                __m256i c2v = _mm256_loadu_si256((const __m256i*)(code2 + i));
                __m512i diff = _mm512_sub_epi16(
                        _mm512_cvtepu8_epi16(c1v), _mm512_cvtepu8_epi16(c2v));
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(diff, diff));
            }
            int32_t result = _mm512_reduce_add_epi32(accu);
            for (; i < d; ++i) {
                int32_t diff = int32_t(code1[i]) - int32_t(code2[i]);
                result += diff * diff;
            }
            return result;
        }
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; ++i) {
            tmp[i] = uint8_t(int(x[i]) + 128);
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

/**********************************************************
 * turboq_masked_sum — delegate to AVX512 implementation
 **********************************************************/

template <SIMDLevel SL0>
float turboq_masked_sum(const float* arr, const uint8_t* bits, size_t d);

template <>
float turboq_masked_sum<SIMDLevel::AVX512>(
        const float* arr,
        const uint8_t* bits,
        size_t d);

template <>
float turboq_masked_sum<SIMDLevel::AVX512_SPR>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    return turboq_masked_sum<SIMDLevel::AVX512>(arr, bits, d);
}

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512_SPR
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#endif // COMPILE_SIMD_AVX512_SPR
