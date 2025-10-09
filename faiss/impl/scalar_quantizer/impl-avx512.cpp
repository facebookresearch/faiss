/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_levels.h>

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>

#include <faiss/utils/simdlib.h>

#if defined(__AVX512F__) && defined(__F16C__)
#define USE_AVX512_F16C
#else
#warning "Wrong compiler flags for AVX512_F16C"
#endif

namespace faiss {

namespace scalar_quantizer {

/******************************** Codec specializations */

template <>
struct Codec8bit<SIMDLevel::AVX512> : Codec8bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd16float32
    decode_16_components(const uint8_t* code, int i) {
        const __m128i c16 = _mm_loadu_si128((__m128i*)(code + i));
        const __m512i i32 = _mm512_cvtepu8_epi32(c16);
        const __m512 f16 = _mm512_cvtepi32_ps(i32);
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 255.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 255.f);
        return simd16float32(_mm512_fmadd_ps(f16, one_255, half_one_255));
    }
};

template <>
struct Codec4bit<SIMDLevel::AVX512> : Codec4bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd16float32
    decode_16_components(const uint8_t* code, int i) {
        uint64_t c8 = *(uint64_t*)(code + (i >> 1));
        uint64_t mask = 0x0f0f0f0f0f0f0f0f;
        uint64_t c8ev = c8 & mask;
        uint64_t c8od = (c8 >> 4) & mask;

        __m128i c16 =
                _mm_unpacklo_epi8(_mm_set1_epi64x(c8ev), _mm_set1_epi64x(c8od));
        __m256i c8lo = _mm256_cvtepu8_epi32(c16);
        __m256i c8hi = _mm256_cvtepu8_epi32(_mm_srli_si128(c16, 8));
        __m512i i16 = _mm512_castsi256_si512(c8lo);
        i16 = _mm512_inserti32x8(i16, c8hi, 1);
        __m512 f16 = _mm512_cvtepi32_ps(i16);
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 15.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 15.f);
        return simd16float32(_mm512_fmadd_ps(f16, one_255, half_one_255));
    }
};

template <>
struct Codec6bit<SIMDLevel::AVX512> : Codec6bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd16float32
    decode_16_components(const uint8_t* code, int i) {
        // pure AVX512 implementation (not necessarily the fastest).
        // see:
        // https://github.com/zilliztech/knowhere/blob/main/thirdparty/faiss/faiss/impl/ScalarQuantizerCodec_avx512.h

        // clang-format off

      // 16 components, 16x6 bit=12 bytes
      const __m128i bit_6v =
              _mm_maskz_loadu_epi8(0b0000111111111111, code + (i >> 2) * 3);
      const __m256i bit_6v_256 = _mm256_broadcast_i32x4(bit_6v);

      // 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F
      // 00          01          02          03
      const __m256i shuffle_mask = _mm256_setr_epi16(
              0xFF00, 0x0100, 0x0201, 0xFF02,
              0xFF03, 0x0403, 0x0504, 0xFF05,
              0xFF06, 0x0706, 0x0807, 0xFF08,
              0xFF09, 0x0A09, 0x0B0A, 0xFF0B);
      const __m256i shuffled = _mm256_shuffle_epi8(bit_6v_256, shuffle_mask);

      // 0: xxxxxxxx xx543210
      // 1: xxxx5432 10xxxxxx
      // 2: xxxxxx54 3210xxxx
      // 3: xxxxxxxx 543210xx
      const __m256i shift_right_v = _mm256_setr_epi16(
              0x0U, 0x6U, 0x4U, 0x2U,
              0x0U, 0x6U, 0x4U, 0x2U,
              0x0U, 0x6U, 0x4U, 0x2U,
              0x0U, 0x6U, 0x4U, 0x2U);
      __m256i shuffled_shifted = _mm256_srlv_epi16(shuffled, shift_right_v);

      // remove unneeded bits
      shuffled_shifted =
              _mm256_and_si256(shuffled_shifted, _mm256_set1_epi16(0x003F));

      // scale
      const __m512 f8 =
              _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(shuffled_shifted));
      const __m512 half_one_255 = _mm512_set1_ps(0.5f / 63.f);
      const __m512 one_255 = _mm512_set1_ps(1.f / 63.f);
      return simd16float32(_mm512_fmadd_ps(f8, one_255, half_one_255));

        // clang-format on
    }
};

/******************************** Quantizer specializations */

template <class Codec>
struct QuantizerT<Codec, QScaling::UNIFORM, SIMDLevel::AVX512>
        : QuantizerT<Codec, QScaling::UNIFORM, SIMDLevel::NONE> {
    QuantizerT(size_t d, const std::vector<float>& trained)
            : QuantizerT<Codec, QScaling::UNIFORM, SIMDLevel::NONE>(
                      d,
                      trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i).f;
        return simd16float32(_mm512_fmadd_ps(
                xi, _mm512_set1_ps(this->vdiff), _mm512_set1_ps(this->vmin)));
    }
};

template <class Codec>
struct QuantizerT<Codec, QScaling::NON_UNIFORM, SIMDLevel::AVX512>
        : QuantizerT<Codec, QScaling::NON_UNIFORM, SIMDLevel::NONE> {
    QuantizerT(size_t d, const std::vector<float>& trained)
            : QuantizerT<Codec, QScaling::NON_UNIFORM, SIMDLevel::NONE>(
                      d,
                      trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i).f;
        return simd16float32(_mm512_fmadd_ps(
                xi,
                _mm512_loadu_ps(this->vdiff + i),
                _mm512_loadu_ps(this->vmin + i)));
    }
};

template <>
struct QuantizerFP16<SIMDLevel::AVX512> : QuantizerFP16<SIMDLevel::NONE> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<SIMDLevel::NONE>(d, trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i codei = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        return simd16float32(_mm512_cvtph_ps(codei));
    }
};

template <>
struct QuantizerBF16<SIMDLevel::AVX512> : QuantizerBF16<SIMDLevel::NONE> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<SIMDLevel::NONE>(d, trained) {}
    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i code_256i = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        __m512i code_512i = _mm512_cvtepu16_epi32(code_256i);
        code_512i = _mm512_slli_epi32(code_512i, 16);
        return simd16float32(_mm512_castsi512_ps(code_512i));
    }
};

template <>
struct Quantizer8bitDirect<SIMDLevel::AVX512>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<SIMDLevel::NONE>(d, trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        return simd16float32(_mm512_cvtepi32_ps(y16));       // 16 * float32
    }
};

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::AVX512>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<SIMDLevel::NONE>(d, trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        __m512i c16 = _mm512_set1_epi32(128);
        __m512i z16 = _mm512_sub_epi32(y16, c16); // subtract 128 from all lanes
        return simd16float32(_mm512_cvtepi32_ps(z16)); // 16 * float32
    }
};

/****************************************** Specialization of similarities */

template <>
struct SimilarityL2<SIMDLevel::AVX512> {
    static constexpr SIMDLevel SIMD_LEVEL = SIMDLevel::AVX512;
    static constexpr int simdwidth = 16;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}
    simd16float32 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(simd16float32 x) {
        __m512 yiv = _mm512_loadu_ps(yi);
        yi += 16;
        __m512 tmp = _mm512_sub_ps(yiv, x.f);
        accu16 = simd16float32(_mm512_fmadd_ps(tmp, tmp, accu16.f));
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(
            simd16float32 x,
            simd16float32 y_2) {
        __m512 tmp = _mm512_sub_ps(y_2.f, x.f);
        accu16 = simd16float32(_mm512_fmadd_ps(tmp, tmp, accu16.f));
    }

    FAISS_ALWAYS_INLINE float result_16() {
        // performs better than dividing into _mm256 and adding
        return _mm512_reduce_add_ps(accu16.f);
    }
};

template <>
struct SimilarityIP<SIMDLevel::AVX512> {
    static constexpr SIMDLevel SIMD_LEVEL = SIMDLevel::AVX512;
    static constexpr int simdwidth = 16;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    simd16float32 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(simd16float32 x) {
        __m512 yiv = _mm512_loadu_ps(yi);
        yi += 16;
        accu16.f = _mm512_fmadd_ps(yiv, x.f, accu16.f);
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(
            simd16float32 x1,
            simd16float32 x2) {
        accu16.f = _mm512_fmadd_ps(x1.f, x2.f, accu16.f);
    }

    FAISS_ALWAYS_INLINE float result_16() {
        // performs better than dividing into _mm256 and adding
        return _mm512_reduce_add_ps(accu16.f);
    }
};

/****************************************** Specialization of distance computers
 */

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer { // Update to handle 16 lanes
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 xi = quant.reconstruct_16_components(code, i);
            sim.add_16_components(xi);
        }
        return sim.result_16();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 x1 = quant.reconstruct_16_components(code1, i);
            simd16float32 x2 = quant.reconstruct_16_components(code2, i);
            sim.add_16_components_2(x1, x2);
        }
        return sim.result_16();
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
struct DistanceComputerByte<Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        __m512i accu = _mm512_setzero_si512();
        for (int i = 0; i < d; i += 32) { // Process 32 bytes at a time
            __m512i c1 = _mm512_cvtepu8_epi16(
                    _mm256_loadu_si256((__m256i*)(code1 + i)));
            __m512i c2 = _mm512_cvtepu8_epi16(
                    _mm256_loadu_si256((__m256i*)(code2 + i)));
            __m512i prod32;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                prod32 = _mm512_madd_epi16(c1, c2);
            } else {
                __m512i diff = _mm512_sub_epi16(c1, c2);
                prod32 = _mm512_madd_epi16(diff, diff);
            }
            accu = _mm512_add_epi32(accu, prod32);
        }
        // Horizontally add elements of accu
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

// explicit instantiation

template ScalarQuantizer::SQuantizer* select_quantizer_1<SIMDLevel::AVX512>(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

template SQDistanceComputer* select_distance_computer_1<SIMDLevel::AVX512>(
        MetricType metric_type,
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

template InvertedListScanner* sel0_InvertedListScanner<SIMDLevel::AVX512>(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual);

} // namespace scalar_quantizer

} // namespace faiss

#endif // COMPILE_SIMD_AVX512
