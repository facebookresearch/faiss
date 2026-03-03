/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

namespace faiss {

namespace scalar_quantizer {

/**********************************************************
 * Codecs
 **********************************************************/

template <>
struct Codec8bit<SIMDLevel::AVX2> : Codec8bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        const uint64_t c8 = *(uint64_t*)(code + i);

        const __m128i i8 = _mm_set1_epi64x(c8);
        const __m256i i32 = _mm256_cvtepu8_epi32(i8);
        const __m256 f8 = _mm256_cvtepi32_ps(i32);
        const __m256 half_one_255 = _mm256_set1_ps(0.5f / 255.f);
        const __m256 one_255 = _mm256_set1_ps(1.f / 255.f);
        return simd8float32(_mm256_fmadd_ps(f8, one_255, half_one_255));
    }
};

template <>
struct Codec4bit<SIMDLevel::AVX2> : Codec4bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        uint32_t c4 = *(uint32_t*)(code + (i >> 1));
        uint32_t mask = 0x0f0f0f0f;
        uint32_t c4ev = c4 & mask;
        uint32_t c4od = (c4 >> 4) & mask;

        // the 8 lower bytes of c8 contain the values
        __m128i c8 =
                _mm_unpacklo_epi8(_mm_set1_epi32(c4ev), _mm_set1_epi32(c4od));
        __m128i c4lo = _mm_cvtepu8_epi32(c8);
        __m128i c4hi = _mm_cvtepu8_epi32(_mm_srli_si128(c8, 4));
        __m256i i8 = _mm256_castsi128_si256(c4lo);
        i8 = _mm256_insertf128_si256(i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps(i8);
        __m256 half = _mm256_set1_ps(0.5f);
        f8 = _mm256_add_ps(f8, half);
        __m256 one_255 = _mm256_set1_ps(1.f / 15.f);
        return simd8float32(_mm256_mul_ps(f8, one_255));
    }
};

template <>
struct Codec6bit<SIMDLevel::AVX2> : Codec6bit<SIMDLevel::NONE> {
    /* Load 6 bytes that represent 8 6-bit values, return them as a
     * 8*32 bit vector register */
    static FAISS_ALWAYS_INLINE __m256i load6(const uint16_t* code16) {
        const __m128i perm = _mm_set_epi8(
                -1, 5, 5, 4, 4, 3, -1, 3, -1, 2, 2, 1, 1, 0, -1, 0);
        const __m256i shifts = _mm256_set_epi32(2, 4, 6, 0, 2, 4, 6, 0);

        // load 6 bytes
        __m128i c1 =
                _mm_set_epi16(0, 0, 0, 0, 0, code16[2], code16[1], code16[0]);

        // put in 8 * 32 bits
        __m128i c2 = _mm_shuffle_epi8(c1, perm);
        __m256i c3 = _mm256_cvtepi16_epi32(c2);

        // shift and mask out useless bits
        __m256i c4 = _mm256_srlv_epi32(c3, shifts);
        __m256i c5 = _mm256_and_si256(_mm256_set1_epi32(63), c4);
        return c5;
    }

    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        // // Faster code for Intel CPUs or AMD Zen3+, just keeping it here
        // // for the reference, maybe, it becomes used one day.
        // const uint16_t* data16 = (const uint16_t*)(code + (i >> 2) * 3);
        // const uint32_t* data32 = (const uint32_t*)data16;
        // const uint64_t val = *data32 + ((uint64_t)data16[2] << 32);
        // const uint64_t vext = _pdep_u64(val, 0x3F3F3F3F3F3F3F3FULL);
        // const __m128i i8 = _mm_set1_epi64x(vext);
        // const __m256i i32 = _mm256_cvtepi8_epi32(i8);
        // const __m256 f8 = _mm256_cvtepi32_ps(i32);
        // const __m256 half_one_255 = _mm256_set1_ps(0.5f / 63.f);
        // const __m256 one_255 = _mm256_set1_ps(1.f / 63.f);
        // return _mm256_fmadd_ps(f8, one_255, half_one_255);

        __m256i i8 = load6((const uint16_t*)(code + (i >> 2) * 3));
        __m256 f8 = _mm256_cvtepi32_ps(i8);
        // this could also be done with bit manipulations but it is
        // not obviously faster
        const __m256 half_one_255 = _mm256_set1_ps(0.5f / 63.f);
        const __m256 one_255 = _mm256_set1_ps(1.f / 63.f);
        return simd8float32(_mm256_fmadd_ps(f8, one_255, half_one_255));
    }
};

/**********************************************************
 * Quantizers (uniform and non-uniform)
 **********************************************************/

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::AVX2>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      QuantizerTemplateScaling::UNIFORM,
                      SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i).f;
        return simd8float32(_mm256_fmadd_ps(
                xi, _mm256_set1_ps(this->vdiff), _mm256_set1_ps(this->vmin)));
    }
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::AVX2>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::NON_UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      QuantizerTemplateScaling::NON_UNIFORM,
                      SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i).f;
        return simd8float32(_mm256_fmadd_ps(
                xi,
                _mm256_loadu_ps(this->vdiff + i),
                _mm256_loadu_ps(this->vmin + i)));
    }
};

/**********************************************************
 * FP16 Quantizer
 **********************************************************/

#if defined(USE_F16C)

template <>
struct QuantizerFP16<SIMDLevel::AVX2> : QuantizerFP16<SIMDLevel::NONE> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i codei = _mm_loadu_si128((const __m128i*)(code + 2 * i));
        return simd8float32(_mm256_cvtph_ps(codei));
    }
};

#endif

/**********************************************************
 * BF16 Quantizer
 **********************************************************/

template <>
struct QuantizerBF16<SIMDLevel::AVX2> : QuantizerBF16<SIMDLevel::NONE> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i code_128i = _mm_loadu_si128((const __m128i*)(code + 2 * i));
        __m256i code_256i = _mm256_cvtepu16_epi32(code_128i);
        code_256i = _mm256_slli_epi32(code_256i, 16);
        return simd8float32(_mm256_castsi256_ps(code_256i));
    }
};

/**********************************************************
 * 8bit Direct Quantizer
 **********************************************************/

template <>
struct Quantizer8bitDirect<SIMDLevel::AVX2>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32(x8);              // 8 * int32
        return simd8float32(_mm256_cvtepi32_ps(y8));        // 8 * float32
    }
};

/**********************************************************
 * 8bit Direct Signed Quantizer
 **********************************************************/

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::AVX2>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32(x8);              // 8 * int32
        __m256i c8 = _mm256_set1_epi32(128);
        __m256i z8 = _mm256_sub_epi32(y8, c8); // subtract 128 from all lanes
        return simd8float32(_mm256_cvtepi32_ps(z8)); // 8 * float32
    }
};

/**********************************************************
 * SimilarityL2 and SimilarityIP
 **********************************************************/

template <>
struct SimilarityL2<SIMDLevel::AVX2> {
    static constexpr int simdwidth = 8;
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX2;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y), yi(nullptr) {}
    simd8float32 accu8 = {};

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(simd8float32 x) {
        __m256 yiv = _mm256_loadu_ps(yi);
        yi += 8;
        __m256 tmp = _mm256_sub_ps(yiv, x.f);
        accu8 = simd8float32(_mm256_fmadd_ps(tmp, tmp, accu8.f));
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            simd8float32 x,
            simd8float32 y_2) {
        __m256 tmp = _mm256_sub_ps(y_2.f, x.f);
        accu8 = simd8float32(_mm256_fmadd_ps(tmp, tmp, accu8.f));
    }

    FAISS_ALWAYS_INLINE float result_8() {
        const __m128 sum = _mm_add_ps(
                _mm256_castps256_ps128(accu8.f),
                _mm256_extractf128_ps(accu8.f, 1));
        const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
        const __m128 v1 = _mm_add_ps(sum, v0);
        __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
        const __m128 v3 = _mm_add_ps(v1, v2);
        return _mm_cvtss_f32(v3);
    }
};

template <>
struct SimilarityIP<SIMDLevel::AVX2> {
    static constexpr int simdwidth = 8;
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX2;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y), yi(nullptr), accu(0) {}

    simd8float32 accu8 = {};

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(simd8float32 x) {
        __m256 yiv = _mm256_loadu_ps(yi);
        yi += 8;
        accu8.f = _mm256_fmadd_ps(yiv, x.f, accu8.f);
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            simd8float32 x1,
            simd8float32 x2) {
        accu8.f = _mm256_fmadd_ps(x1.f, x2.f, accu8.f);
    }

    FAISS_ALWAYS_INLINE float result_8() {
        const __m128 sum = _mm_add_ps(
                _mm256_castps256_ps128(accu8.f),
                _mm256_extractf128_ps(accu8.f, 1));
        const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
        const __m128 v1 = _mm_add_ps(sum, v0);
        __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
        const __m128 v3 = _mm_add_ps(v1, v2);
        return _mm_cvtss_f32(v3);
    }
};

/**********************************************************
 * Distance computers
 **********************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX2> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 xi =
                    quant.reconstruct_8_components(code, static_cast<int>(i));
            sim.add_8_components(xi);
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 x1 =
                    quant.reconstruct_8_components(code1, static_cast<int>(i));
            simd8float32 x2 =
                    quant.reconstruct_8_components(code2, static_cast<int>(i));
            sim.add_8_components_2(x1, x2);
        }
        return sim.result_8();
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

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::AVX2> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        // __m256i accu = _mm256_setzero_ps ();
        __m256i accu = _mm256_setzero_si256();
        for (int i = 0; i < d; i += 16) {
            // load 16 bytes, convert to 16 uint16_t
            __m256i c1 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code1 + i)));
            __m256i c2 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code2 + i)));
            __m256i prod32;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                prod32 = _mm256_madd_epi16(c1, c2);
            } else {
                __m256i diff = _mm256_sub_epi16(c1, c2);
                prod32 = _mm256_madd_epi16(diff, diff);
            }
            accu = _mm256_add_epi32(accu, prod32);
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32(sum, _mm256_extractf128_si256(accu, 1));
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(sum);
    }

    void set_query(const float* x) final {
        /*
        for (int i = 0; i < d; i += 8) {
            __m256 xi = _mm256_loadu_ps (x + i);
            __m256i ci = _mm256_cvtps_epi32(xi);
        */
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

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX2
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#endif // COMPILE_SIMD_AVX2
