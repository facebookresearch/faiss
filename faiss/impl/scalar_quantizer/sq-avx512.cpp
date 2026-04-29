/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/simdlib/simdlib_avx512.h>

#include <cstring>

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

namespace faiss {

namespace scalar_quantizer {

using simd16float32 = faiss::simd16float32_tpl<SIMDLevel::AVX512>;

namespace {

FAISS_ALWAYS_INLINE uint16_t load_u16(const uint8_t* ptr) {
    uint16_t value;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

FAISS_ALWAYS_INLINE uint32_t load_u32(const uint8_t* ptr) {
    uint32_t value;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

FAISS_ALWAYS_INLINE uint64_t load_u64(const uint8_t* ptr) {
    uint64_t value;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

FAISS_ALWAYS_INLINE uint32_t load_u24(const uint8_t* ptr) {
    return static_cast<uint32_t>(ptr[0]) |
            (static_cast<uint32_t>(ptr[1]) << 8) |
            (static_cast<uint32_t>(ptr[2]) << 16);
}

FAISS_ALWAYS_INLINE __m256i unpack_8x3bit_to_u32(uint32_t packed) {
    const __m256i shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256i indices =
            _mm256_srlv_epi32(_mm256_set1_epi32(packed), shifts);
    return _mm256_and_si256(indices, _mm256_set1_epi32(0x7));
}

FAISS_ALWAYS_INLINE __m512i unpack_16x1bit_to_u32(const uint8_t* code, int i) {
    const uint32_t packed = load_u16(code + (static_cast<size_t>(i) >> 3));
    const __m512i shifts = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i indices =
            _mm512_srlv_epi32(_mm512_set1_epi32(packed), shifts);
    return _mm512_and_si512(indices, _mm512_set1_epi32(0x1));
}

FAISS_ALWAYS_INLINE __m512i unpack_16x2bit_to_u32(const uint8_t* code, int i) {
    const uint32_t packed = load_u32(code + (static_cast<size_t>(i) >> 2));
    const __m512i shifts = _mm512_setr_epi32(
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i indices =
            _mm512_srlv_epi32(_mm512_set1_epi32(packed), shifts);
    return _mm512_and_si512(indices, _mm512_set1_epi32(0x3));
}

FAISS_ALWAYS_INLINE __m512i unpack_16x3bit_to_u32(const uint8_t* code, int i) {
    const size_t byte_offset = (static_cast<size_t>(i) >> 4) * 6;
    const __m256i low = unpack_8x3bit_to_u32(load_u24(code + byte_offset));
    const __m256i high = unpack_8x3bit_to_u32(load_u24(code + byte_offset + 3));
    __m512i indices = _mm512_castsi256_si512(low);
    return _mm512_inserti32x8(indices, high, 1);
}

FAISS_ALWAYS_INLINE __m512i unpack_16x4bit_to_u32(const uint8_t* code, int i) {
    const uint64_t packed = load_u64(code + (static_cast<size_t>(i) >> 1));
    const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i low = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32((uint32_t)packed), shifts),
            _mm256_set1_epi32(0xf));
    const __m256i high = _mm256_and_si256(
            _mm256_srlv_epi32(
                    _mm256_set1_epi32((uint32_t)(packed >> 32)), shifts),
            _mm256_set1_epi32(0xf));
    __m512i indices = _mm512_castsi256_si512(low);
    return _mm512_inserti32x8(indices, high, 1);
}

} // namespace

/**********************************************************
 * Codecs
 **********************************************************/

template <>
struct Codec8bit<SIMDLevel::AVX512> : Codec8bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd16float32
    decode_16_components(const uint8_t* code, size_t i) {
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
    decode_16_components(const uint8_t* code, size_t i) {
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
    decode_16_components(const uint8_t* code, size_t i) {
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

/**********************************************************
 * Quantizers (uniform and non-uniform)
 **********************************************************/

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::AVX512>
        : QuantizerTemplate<
                  Codec,
                  scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
                      SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i).f;
        return simd16float32(_mm512_fmadd_ps(
                xi, _mm512_set1_ps(this->vdiff), _mm512_set1_ps(this->vmin)));
    }
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::AVX512>
        : QuantizerTemplate<
                  Codec,
                  scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
                      SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i).f;
        return simd16float32(_mm512_fmadd_ps(
                xi,
                _mm512_loadu_ps(this->vdiff + i),
                _mm512_loadu_ps(this->vmin + i)));
    }
};

/**********************************************************
 * TurboQuant MSE quantizer
 **********************************************************/

#define DEFINE_TQMSE_AVX512_SPECIALIZATION(NBITS, INDEX_EXPR)               \
    template <>                                                             \
    struct QuantizerTurboQuantMSE<NBITS, SIMDLevel::AVX512>                 \
            : QuantizerTurboQuantMSE<NBITS, SIMDLevel::NONE> {              \
        using Base = QuantizerTurboQuantMSE<NBITS, SIMDLevel::NONE>;        \
                                                                            \
        QuantizerTurboQuantMSE(size_t d, const std::vector<float>& trained) \
                : Base(d, trained) {                                        \
            assert(d % 16 == 0);                                            \
        }                                                                   \
                                                                            \
        FAISS_ALWAYS_INLINE simd16float32                                   \
        reconstruct_16_components(const uint8_t* code, int i) const {       \
            const __m512i indices = (INDEX_EXPR);                           \
            return simd16float32(_mm512_i32gather_ps(                       \
                    indices, this->centroids, sizeof(float)));              \
        }                                                                   \
    }

DEFINE_TQMSE_AVX512_SPECIALIZATION(1, unpack_16x1bit_to_u32(code, i));
DEFINE_TQMSE_AVX512_SPECIALIZATION(2, unpack_16x2bit_to_u32(code, i));
DEFINE_TQMSE_AVX512_SPECIALIZATION(3, unpack_16x3bit_to_u32(code, i));
DEFINE_TQMSE_AVX512_SPECIALIZATION(4, unpack_16x4bit_to_u32(code, i));

#undef DEFINE_TQMSE_AVX512_SPECIALIZATION

template <>
struct QuantizerTurboQuantMSE<8, SIMDLevel::AVX512>
        : QuantizerTurboQuantMSE<8, SIMDLevel::NONE> {
    using Base = QuantizerTurboQuantMSE<8, SIMDLevel::NONE>;

    QuantizerTurboQuantMSE(size_t d, const std::vector<float>& trained)
            : Base(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        const __m128i packed = _mm_loadu_si128(
                (const __m128i*)(code + static_cast<size_t>(i)));
        const __m512i indices = _mm512_cvtepu8_epi32(packed);
        return simd16float32(
                _mm512_i32gather_ps(indices, this->centroids, sizeof(float)));
    }
};

/**********************************************************
 * FP16 Quantizer
 **********************************************************/

template <>
struct QuantizerFP16<SIMDLevel::AVX512> : QuantizerFP16<SIMDLevel::NONE> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i codei = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        return simd16float32(_mm512_cvtph_ps(codei));
    }
};

/**********************************************************
 * BF16 Quantizer
 **********************************************************/

template <>
struct QuantizerBF16<SIMDLevel::AVX512> : QuantizerBF16<SIMDLevel::NONE> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i code_256i = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        __m512i code_512i = _mm512_cvtepu16_epi32(code_256i);
        code_512i = _mm512_slli_epi32(code_512i, 16);
        return simd16float32(_mm512_castsi512_ps(code_512i));
    }
};

/**********************************************************
 * 8bit Direct Quantizer
 **********************************************************/

template <>
struct Quantizer8bitDirect<SIMDLevel::AVX512>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        return simd16float32(_mm512_cvtepi32_ps(y16));       // 16 * float32
    }
};

/**********************************************************
 * 8bit Direct Signed Quantizer
 **********************************************************/

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::AVX512>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        __m512i c16 = _mm512_set1_epi32(128);
        __m512i z16 = _mm512_sub_epi32(y16, c16); // subtract 128 from all lanes
        return simd16float32(_mm512_cvtepi32_ps(z16)); // 16 * float32
    }
};

/**********************************************************
 * Similarities (L2 and IP)
 **********************************************************/

template <>
struct SimilarityL2<SIMDLevel::AVX512> {
    static constexpr int simdwidth = 16;
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX512;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y), yi(nullptr) {}

    simd16float32 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(simd16float32 x) {
        simd16float32 yiv(yi);
        yi += 16;
        simd16float32 tmp = yiv - x;
        accu16 = accu16 + tmp * tmp;
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(
            simd16float32 x,
            simd16float32 y_2) {
        simd16float32 tmp = y_2 - x;
        accu16 = accu16 + tmp * tmp;
    }

    FAISS_ALWAYS_INLINE float result_16() {
        return horizontal_add(accu16);
    }
};

template <>
struct SimilarityIP<SIMDLevel::AVX512> {
    static constexpr int simdwidth = 16;
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX512;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    explicit SimilarityIP(const float* y) : y(y), yi(nullptr) {}

    simd16float32 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(simd16float32 x) {
        simd16float32 yiv(yi);
        yi += 16;
        accu16 = accu16 + yiv * x;
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(
            simd16float32 x1,
            simd16float32 x2) {
        accu16 = accu16 + x1 * x2;
    }

    FAISS_ALWAYS_INLINE float result_16() {
        return horizontal_add(accu16);
    }
};

/**********************************************************
 * Distance Computers
 **********************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer {
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

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        Similarity sim0(q);
        Similarity sim1(q);
        Similarity sim2(q);
        Similarity sim3(q);

        sim0.begin_16();
        sim1.begin_16();
        sim2.begin_16();
        sim3.begin_16();

        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 xi0 = quant.reconstruct_16_components(code_0, i);
            simd16float32 xi1 = quant.reconstruct_16_components(code_1, i);
            simd16float32 xi2 = quant.reconstruct_16_components(code_2, i);
            simd16float32 xi3 = quant.reconstruct_16_components(code_3, i);
            sim0.add_16_components(xi0);
            sim1.add_16_components(xi1);
            sim2.add_16_components(xi2);
            sim3.add_16_components(xi3);
        }

        dis0 = sim0.result_16();
        dis1 = sim1.result_16();
        dis2 = sim2.result_16();
        dis3 = sim3.result_16();
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
        // compute 16 lanes of 32-bit products (16-bytes) at once for
        // the supported metrics
        __m512i accu = _mm512_setzero_si512();
        constexpr int kLanes = 16;
        for (int i = 0; i < d; i += kLanes) {
            __m128i c1 = _mm_loadu_si128((__m128i*)(code1 + i));
            __m128i c2 = _mm_loadu_si128((__m128i*)(code2 + i));
            __m512i c1i = _mm512_cvtepu8_epi32(c1);
            __m512i c2i = _mm512_cvtepu8_epi32(c2);

            __m512i v;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                v = _mm512_mullo_epi32(c1i, c2i);
            } else {
                __m512i diff = _mm512_sub_epi32(c1i, c2i);
                v = _mm512_mullo_epi32(diff, diff);
            }
            accu = _mm512_add_epi32(accu, v);
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

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#endif // COMPILE_SIMD_AVX512
