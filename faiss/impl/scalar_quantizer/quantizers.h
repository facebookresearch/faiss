/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

namespace scalar_quantizer {

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

enum class QuantizerTemplateScaling { UNIFORM = 0, NON_UNIFORM = 1 };

template <class Codec, QuantizerTemplateScaling SCALING, int SIMD>
struct QuantizerTemplate {};

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1>
        : ScalarQuantizer::SQuantizer {
    const size_t d;
    const float vmin, vdiff;

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : d(d), vmin(trained[0]), vdiff(trained[1]) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = 0;
            if (vdiff != 0) {
                xi = (x[i] - vmin) / vdiff;
                if (xi < 0) {
                    xi = 0;
                }
                if (xi > 1.0) {
                    xi = 1.0;
                }
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin + xi * vdiff;
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        float xi = Codec::decode_component(code, i);
        return vmin + xi * vdiff;
    }
};

#if defined(__AVX512F__)

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 16>
        : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1>(
                      d,
                      trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i);
        return simd16float32(_mm512_fmadd_ps(
                xi, _mm512_set1_ps(this->vdiff), _mm512_set1_ps(this->vmin)));
    }
};

#elif defined(__AVX2__)

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 8>
        : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1>(
                      d,
                      trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i).f;
        return simd8float32(_mm256_fmadd_ps(
                xi, _mm256_set1_ps(this->vdiff), _mm256_set1_ps(this->vmin)));
    }
};

#endif

#ifdef USE_NEON

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 8>
        : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1>(
                      d,
                      trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        float32x4x2_t xi = Codec::decode_8_components(code, i);
        return simd8float32(float32x4x2_t(
                {vfmaq_f32(
                         vdupq_n_f32(this->vmin),
                         xi.val[0],
                         vdupq_n_f32(this->vdiff)),
                 vfmaq_f32(
                         vdupq_n_f32(this->vmin),
                         xi.val[1],
                         vdupq_n_f32(this->vdiff))}));
    }
};

#endif

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1>
        : ScalarQuantizer::SQuantizer {
    const size_t d;
    const float *vmin, *vdiff;

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : d(d), vmin(trained.data()), vdiff(trained.data() + d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = 0;
            if (vdiff[i] != 0) {
                xi = (x[i] - vmin[i]) / vdiff[i];
                if (xi < 0) {
                    xi = 0;
                }
                if (xi > 1.0) {
                    xi = 1.0;
                }
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin[i] + xi * vdiff[i];
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        float xi = Codec::decode_component(code, i);
        return vmin[i] + xi * vdiff[i];
    }
};

#if defined(__AVX512F__)

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 16>
        : QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      QuantizerTemplateScaling::NON_UNIFORM,
                      1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i).f;
        return simd16float32(_mm512_fmadd_ps(
                xi,
                _mm512_loadu_ps(this->vdiff + i),
                _mm512_loadu_ps(this->vmin + i)));
    }
};

#elif defined(__AVX2__)

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 8>
        : QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      QuantizerTemplateScaling::NON_UNIFORM,
                      1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i).f;
        return simd8float32(_mm256_fmadd_ps(
                xi,
                _mm256_loadu_ps(this->vdiff + i),
                _mm256_loadu_ps(this->vmin + i)));
    }
};

#endif

#ifdef USE_NEON

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 8>
        : QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      QuantizerTemplateScaling::NON_UNIFORM,
                      1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        float32x4x2_t xi = Codec::decode_8_components(code, i).data;

        float32x4x2_t vmin_8 = vld1q_f32_x2(this->vmin + i);
        float32x4x2_t vdiff_8 = vld1q_f32_x2(this->vdiff + i);

        return simd8float32(
                {vfmaq_f32(vmin_8.val[0], xi.val[0], vdiff_8.val[0]),
                 vfmaq_f32(vmin_8.val[1], xi.val[1], vdiff_8.val[1])});
    }
};

#endif

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerFP16 {};

template <>
struct QuantizerFP16<1> : ScalarQuantizer::SQuantizer {
    const size_t d;

    QuantizerFP16(size_t d, const std::vector<float>& /* unused */) : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            ((uint16_t*)code)[i] = encode_fp16(x[i]);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = decode_fp16(((uint16_t*)code)[i]);
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return decode_fp16(((uint16_t*)code)[i]);
    }
};

#if defined(USE_AVX512_F16C)

template <>
struct QuantizerFP16<16> : QuantizerFP16<1> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i codei = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        return simd16float32(_mm512_cvtph_ps(codei));
    }
};

#endif

#if defined(USE_F16C)

template <>
struct QuantizerFP16<8> : QuantizerFP16<1> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i codei = _mm_loadu_si128((const __m128i*)(code + 2 * i));
        return simd8float32(_mm256_cvtph_ps(codei));
    }
};

#endif

#ifdef USE_NEON

template <>
struct QuantizerFP16<8> : QuantizerFP16<1> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint16x4x2_t codei = vld1_u16_x2((const uint16_t*)(code + 2 * i));
        return simd8float32(
                {vcvt_f32_f16(vreinterpret_f16_u16(codei.val[0])),
                 vcvt_f32_f16(vreinterpret_f16_u16(codei.val[1]))});
    }
};
#endif

/*******************************************************************
 * BF16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerBF16 {};

template <>
struct QuantizerBF16<1> : ScalarQuantizer::SQuantizer {
    const size_t d;

    QuantizerBF16(size_t d, const std::vector<float>& /* unused */) : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            ((uint16_t*)code)[i] = encode_bf16(x[i]);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = decode_bf16(((uint16_t*)code)[i]);
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return decode_bf16(((uint16_t*)code)[i]);
    }
};

#if defined(__AVX512F__)

template <>
struct QuantizerBF16<16> : QuantizerBF16<1> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<1>(d, trained) {}
    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i code_256i = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        __m512i code_512i = _mm512_cvtepu16_epi32(code_256i);
        code_512i = _mm512_slli_epi32(code_512i, 16);
        return simd16float32(_mm512_castsi512_ps(code_512i));
    }
};

#elif defined(__AVX2__)

template <>
struct QuantizerBF16<8> : QuantizerBF16<1> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i code_128i = _mm_loadu_si128((const __m128i*)(code + 2 * i));
        __m256i code_256i = _mm256_cvtepu16_epi32(code_128i);
        code_256i = _mm256_slli_epi32(code_256i, 16);
        return simd8float32(_mm256_castsi256_ps(code_256i));
    }
};

#endif

#ifdef USE_NEON

template <>
struct QuantizerBF16<8> : QuantizerBF16<1> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint16x4x2_t codei = vld1_u16_x2((const uint16_t*)(code + 2 * i));
        return simd8float32(
                {vreinterpretq_f32_u32(
                         vshlq_n_u32(vmovl_u16(codei.val[0]), 16)),
                 vreinterpretq_f32_u32(
                         vshlq_n_u32(vmovl_u16(codei.val[1]), 16))});
    }
};
#endif

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirect {};

template <>
struct Quantizer8bitDirect<1> : ScalarQuantizer::SQuantizer {
    const size_t d;

    Quantizer8bitDirect(size_t d, const std::vector<float>& /* unused */)
            : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            code[i] = (uint8_t)x[i];
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = code[i];
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return code[i];
    }
};

#if defined(__AVX512F__)

template <>
struct Quantizer8bitDirect<16> : Quantizer8bitDirect<1> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        return simd16float32(_mm512_cvtepi32_ps(y16));       // 16 * float32
    }
};

#elif defined(__AVX2__)

template <>
struct Quantizer8bitDirect<8> : Quantizer8bitDirect<1> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32(x8);              // 8 * int32
        return simd8float32(_mm256_cvtepi32_ps(y8));        // 8 * float32
    }
};

#endif

#ifdef USE_NEON

template <>
struct Quantizer8bitDirect<8> : Quantizer8bitDirect<1> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint8x8_t x8 = vld1_u8((const uint8_t*)(code + i));
        uint16x8_t y8 = vmovl_u8(x8);
        uint16x4_t y8_0 = vget_low_u16(y8);
        uint16x4_t y8_1 = vget_high_u16(y8);

        // convert uint16 -> uint32 -> fp32
        return simd8float32(
                {vcvtq_f32_u32(vmovl_u16(y8_0)),
                 vcvtq_f32_u32(vmovl_u16(y8_1))});
    }
};

#endif

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirectSigned {};

template <>
struct Quantizer8bitDirectSigned<1> : ScalarQuantizer::SQuantizer {
    const size_t d;

    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& /* unused */)
            : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            code[i] = (uint8_t)(x[i] + 128);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = code[i] - 128;
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return code[i] - 128;
    }
};

#if defined(__AVX512F__)

template <>
struct Quantizer8bitDirectSigned<16> : Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        __m512i c16 = _mm512_set1_epi32(128);
        __m512i z16 = _mm512_sub_epi32(y16, c16); // subtract 128 from all lanes
        return simd16float32(_mm512_cvtepi32_ps(z16)); // 16 * float32
    }
};

#elif defined(__AVX2__)

template <>
struct Quantizer8bitDirectSigned<8> : Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32(x8);              // 8 * int32
        __m256i c8 = _mm256_set1_epi32(128);
        __m256i z8 = _mm256_sub_epi32(y8, c8); // subtract 128 from all lanes
        return simd8float32(_mm256_cvtepi32_ps(z8)); // 8 * float32
    }
};

#endif

#ifdef USE_NEON

template <>
struct Quantizer8bitDirectSigned<8> : Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<1>(d, trained) {}

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint8x8_t x8 = vld1_u8((const uint8_t*)(code + i));
        uint16x8_t y8 = vmovl_u8(x8); // convert uint8 -> uint16
        uint16x4_t y8_0 = vget_low_u16(y8);
        uint16x4_t y8_1 = vget_high_u16(y8);

        float32x4_t z8_0 = vcvtq_f32_u32(
                vmovl_u16(y8_0)); // convert uint16 -> uint32 -> fp32
        float32x4_t z8_1 = vcvtq_f32_u32(vmovl_u16(y8_1));

        // subtract 128 to convert into signed numbers
        return simd8float32(
                {vsubq_f32(z8_0, vmovq_n_f32(128.0)),
                 vsubq_f32(z8_1, vmovq_n_f32(128.0))});
    }
};

#endif

} // namespace scalar_quantizer

} // namespace faiss
