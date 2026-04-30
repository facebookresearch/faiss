/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512_SPR

// Include AVX512 implementations (codecs, quantizers, similarities,
// DCTemplate, DistanceComputerByte) so they are visible in this TU.
// SQ_AVX512_SKIP_DISPATCH prevents sq-dispatch.h from being included.
#define SQ_AVX512_SKIP_DISPATCH
#include "sq-avx512.cpp"
#undef SQ_AVX512_SKIP_DISPATCH

#include <immintrin.h>

namespace faiss {
namespace scalar_quantizer {

/*******************************************************************
 * AVX512_SPR specializations
 *
 * For most types, it inherit directly from the AVX512 implementations
 * Only override types with dedicated SPR-specific implementations.
 *******************************************************************/

/*******************************************************************
 * Codecs: inherit from AVX512 (same decode logic)
 *******************************************************************/

template <>
struct Codec8bit<SIMDLevel::AVX512_SPR> : Codec8bit<SIMDLevel::AVX512> {};

template <>
struct Codec4bit<SIMDLevel::AVX512_SPR> : Codec4bit<SIMDLevel::AVX512> {};

template <>
struct Codec6bit<SIMDLevel::AVX512_SPR> : Codec6bit<SIMDLevel::AVX512> {};

/*******************************************************************
 * Quantizers: inherit from AVX512
 *******************************************************************/

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::AVX512_SPR>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::AVX512> {
    using Base = QuantizerTemplate<
            Codec,
            QuantizerTemplateScaling::UNIFORM,
            SIMDLevel::AVX512>;
    using Base::Base;
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
    using Base = QuantizerTemplate<
            Codec,
            QuantizerTemplateScaling::NON_UNIFORM,
            SIMDLevel::AVX512>;
    using Base::Base;
};

template <int NBITS>
struct QuantizerTurboQuantMSE<NBITS, SIMDLevel::AVX512_SPR>
        : QuantizerTurboQuantMSE<NBITS, SIMDLevel::AVX512> {
    using Base = QuantizerTurboQuantMSE<NBITS, SIMDLevel::AVX512>;
    using Base::Base;
};

template <>
struct QuantizerFP16<SIMDLevel::AVX512_SPR> : QuantizerFP16<SIMDLevel::AVX512> {
    using Base = QuantizerFP16<SIMDLevel::AVX512>;
    using Base::Base;
};

template <>
struct QuantizerBF16<SIMDLevel::AVX512_SPR> : QuantizerBF16<SIMDLevel::AVX512> {
    using Base = QuantizerBF16<SIMDLevel::AVX512>;
    using Base::Base;
};

template <>
struct Quantizer8bitDirect<SIMDLevel::AVX512_SPR>
        : Quantizer8bitDirect<SIMDLevel::AVX512> {
    using Base = Quantizer8bitDirect<SIMDLevel::AVX512>;
    using Base::Base;
};

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::AVX512_SPR>
        : Quantizer8bitDirectSigned<SIMDLevel::AVX512> {
    using Base = Quantizer8bitDirectSigned<SIMDLevel::AVX512>;
    using Base::Base;
};

/*******************************************************************
 * Similarities: inherit from AVX512
 *******************************************************************/

template <>
struct SimilarityL2<SIMDLevel::AVX512_SPR> : SimilarityL2<SIMDLevel::AVX512> {
    using SimilarityL2<SIMDLevel::AVX512>::SimilarityL2;
};

template <>
struct SimilarityIP<SIMDLevel::AVX512_SPR> : SimilarityIP<SIMDLevel::AVX512> {
    using SimilarityIP<SIMDLevel::AVX512>::SimilarityIP;
};

/*******************************************************************
 * DCTemplate: inherit from AVX512
 *******************************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512_SPR>
        : DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512> {
    using Base = DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512>;
    using Base::Base;
};

/*******************************************************************
 * DistanceComputerByte: AVX512-VNNI
 *
 * Uses _mm512_dpbusd_epi32 to compute dot products of uint8 vectors
 * at 64 bytes per instruction (4x throughput vs generic AVX512).
 *******************************************************************/

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
            // VNNI: unsigned*signed dot product accumulated into int32
            // dpbusd(src, a_u8, b_i8) = src + sum_j(a_u8[j]*b_i8[j])
            // For unsigned*unsigned: result = dpbusd(0, a, b^0x80) + 128*sum(a)
            // We use _mm512_maddubs_epi16 approach for simplicity:
            // u8*i8 -> i16 pairwise, then madd_epi16 to accumulate to i32.

            __m512i accu = _mm512_setzero_si512();
            int i = 0;
            for (; i + 64 <= d; i += 64) {
                __m512i c1 = _mm512_loadu_si512(code1 + i);
                __m512i c2 = _mm512_loadu_si512(code2 + i);

                // Subtract 128 from c2 to make it "signed" for dpbusd
                // Then correct: a*b = a*(b-128) + 128*a
                __m512i c2_signed = _mm512_sub_epi8(c2, _mm512_set1_epi8(-128));
                accu = _mm512_dpbusd_epi32(accu, c1, c2_signed);
            }
            // Correction: add 128 * sum(code1[0..processed])
            int32_t sum_c1 = 0;
            for (int j = 0; j < i; j++) {
                sum_c1 += code1[j];
            }
            int32_t result = _mm512_reduce_add_epi32(accu) + 128 * sum_c1;

            // Handle tail
            for (; i < d; i++) {
                result += int(code1[i]) * code2[i];
            }
            return result;
        } else {
            // Expand to 16-bit, subtract, square (madd with self), accumulate.
            __m512i accu = _mm512_setzero_si512();
            int i = 0;
            for (; i + 64 <= d; i += 64) {
                // Process 64 bytes in two halves of 32
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

                // madd_epi16 does (d[0]*d[0] + d[1]*d[1]) -> 32-bit
                accu = _mm512_add_epi32(
                        accu, _mm512_madd_epi16(diff_lo, diff_lo));
                accu = _mm512_add_epi32(
                        accu, _mm512_madd_epi16(diff_hi, diff_hi));
            }
            // Process remaining 32-byte chunk
            for (; i + 32 <= d; i += 32) {
                __m256i c1v = _mm256_loadu_si256((const __m256i*)(code1 + i));
                __m256i c2v = _mm256_loadu_si256((const __m256i*)(code2 + i));
                __m512i c1_16 = _mm512_cvtepu8_epi16(c1v);
                __m512i c2_16 = _mm512_cvtepu8_epi16(c2v);
                __m512i diff = _mm512_sub_epi16(c1_16, c2_16);
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(diff, diff));
            }
            int32_t result = _mm512_reduce_add_epi32(accu);

            // Scalar tail
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

/*******************************************************************
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
 *******************************************************************/

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
            // Reuse the unsigned VNNI loop, then correct.
            __m512i accu = _mm512_setzero_si512();
            __m512i sum_a = _mm512_setzero_si512();
            __m512i sum_b = _mm512_setzero_si512();
            const __m512i zero = _mm512_setzero_si512();
            const __m512i bias = _mm512_set1_epi8(-128);

            int i = 0;
            for (; i + 64 <= d; i += 64) {
                __m512i c1 = _mm512_loadu_si512(code1 + i);
                __m512i c2 = _mm512_loadu_si512(code2 + i);

                // sum_a += sum(c1[i..i+63]); same for c2 (cheap via PSADBW).
                sum_a = _mm512_add_epi64(sum_a, _mm512_sad_epu8(c1, zero));
                sum_b = _mm512_add_epi64(sum_b, _mm512_sad_epu8(c2, zero));

                // Unsigned VNNI dot (bias trick):
                __m512i c2_signed = _mm512_sub_epi8(c2, bias);
                accu = _mm512_dpbusd_epi32(accu, c1, c2_signed);
            }
            int32_t sum_c1_for_bias = int32_t(_mm512_reduce_add_epi64(sum_a));
            int32_t result =
                    _mm512_reduce_add_epi32(accu) + 128 * sum_c1_for_bias;

            // Scalar tail for the unsigned product.
            int32_t tail_sum_a = 0, tail_sum_b = 0;
            for (; i < d; ++i) {
                result += int32_t(code1[i]) * int32_t(code2[i]);
                tail_sum_a += code1[i];
                tail_sum_b += code2[i];
            }

            // Apply signed correction over the *full* d:
            //   - 128 * (sum_a + sum_b) + 16384 * d
            int32_t total_sum_a = sum_c1_for_bias + tail_sum_a;
            int32_t total_sum_b =
                    int32_t(_mm512_reduce_add_epi64(sum_b)) + tail_sum_b;
            result -= 128 * (total_sum_a + total_sum_b);
            result += 16384 * d;
            return result;
        } else {
            // L2: (s_a - 128) - (s_b - 128) == s_a - s_b, so the bias
            // cancels and the unsigned widened-madd kernel is bit-exact.
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
        // Encode with the +128 offset to match QT_8bit_direct_signed storage.
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

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512_SPR
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#endif // COMPILE_SIMD_AVX512_SPR
