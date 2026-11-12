/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512
#include <faiss/impl/scalar_quantizer/sq-avx512-impl.h>
#include <faiss/utils/simd_impl/masked_sum-avx512.h>

namespace faiss {

namespace scalar_quantizer {

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        // compute 32 lanes of 16-bit products (32-bytes) at once for
        // the supported metrics
        __m512i accu = _mm512_setzero_si512();
        constexpr int kLanes = 32;
        int i = 0;
        for (; i + kLanes <= d; i += kLanes) {
            __m256i c1 = _mm256_loadu_epi8(code1 + i);
            __m256i c2 = _mm256_loadu_epi8(code2 + i);
            __m512i c1i16 = _mm512_cvtepu8_epi16(c1);
            __m512i c2i16 = _mm512_cvtepu8_epi16(c2);
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(c1i16, c2i16));
            } else {
                __m512i diff = _mm512_sub_epi16(c1i16, c2i16);
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(diff, diff));
            }
        }
        // tail handling for dimensions not divisible by 32
        if (i < d) {
            __mmask32 mask = (__mmask32)((1ULL << (d - i)) - 1ULL);
            __m256i c1 = _mm256_maskz_loadu_epi8(mask, code1 + i);
            __m256i c2 = _mm256_maskz_loadu_epi8(mask, code2 + i);
            __m512i c1i16 = _mm512_cvtepu8_epi16(c1);
            __m512i c2i16 = _mm512_cvtepu8_epi16(c2);
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(c1i16, c2i16));
            } else {
                __m512i diff = _mm512_sub_epi16(c1i16, c2i16);
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(diff, diff));
            }
        }
        return _mm512_reduce_add_epi32(accu);
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

template <class Similarity>
struct DistanceComputerByteSigned<Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByteSigned(int d, const std::vector<float>&)
            : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        // codes store value + 128. madd_epi16 is signed, so IP unbiases the
        // bytes before multiplying; for L2 the +128 cancels in the difference.
        // Only dispatched for d % 32 == 0, so the loop needs no tail.
        __m512i accu = _mm512_setzero_si512();
        constexpr int kLanes = 32;
        for (int i = 0; i + kLanes <= d; i += kLanes) {
            __m512i c1 = _mm512_cvtepu8_epi16(_mm256_loadu_epi8(code1 + i));
            __m512i c2 = _mm512_cvtepu8_epi16(_mm256_loadu_epi8(code2 + i));
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                const __m512i bias = _mm512_set1_epi16(128);
                c1 = _mm512_sub_epi16(c1, bias);
                c2 = _mm512_sub_epi16(c2, bias);
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(c1, c2));
            } else {
                __m512i diff = _mm512_sub_epi16(c1, c2);
                accu = _mm512_add_epi32(accu, _mm512_madd_epi16(diff, diff));
            }
        }
        return _mm512_reduce_add_epi32(accu);
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
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

#include <faiss/impl/scalar_quantizer/sq-dispatch.h>
