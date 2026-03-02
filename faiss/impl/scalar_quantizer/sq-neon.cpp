/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_NEON

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
struct Codec8bit<SIMDLevel::NEON> : Codec8bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] =
                    Codec8bit<SIMDLevel::NONE>::decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return simd8float32(float32x4x2_t{res1, res2});
    }
};

template <>
struct Codec4bit<SIMDLevel::NEON> : Codec4bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] =
                    Codec4bit<SIMDLevel::NONE>::decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return simd8float32(float32x4x2_t{res1, res2});
    }
};

template <>
struct Codec6bit<SIMDLevel::NEON> : Codec6bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] =
                    Codec6bit<SIMDLevel::NONE>::decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return simd8float32(float32x4x2_t{res1, res2});
    }
};

/**********************************************************
 * Quantizers (uniform and non-uniform)
 **********************************************************/

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::NEON>
        : QuantizerTemplate<
                  Codec,
                  scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
                      SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        simd8float32 xi = Codec::decode_8_components(code, i);
        return simd8float32(
                float32x4x2_t{
                        vfmaq_n_f32(
                                vdupq_n_f32(this->vmin),
                                xi.data.val[0],
                                this->vdiff),
                        vfmaq_n_f32(
                                vdupq_n_f32(this->vmin),
                                xi.data.val[1],
                                this->vdiff)});
    }
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::NEON>
        : QuantizerTemplate<
                  Codec,
                  scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
                      SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        simd8float32 xi = Codec::decode_8_components(code, i);
        return simd8float32(
                float32x4x2_t{
                        vfmaq_f32(
                                vld1q_f32(this->vmin + i),
                                xi.data.val[0],
                                vld1q_f32(this->vdiff + i)),
                        vfmaq_f32(
                                vld1q_f32(this->vmin + i + 4),
                                xi.data.val[1],
                                vld1q_f32(this->vdiff + i + 4))});
    }
};

/**********************************************************
 * FP16 Quantizer
 **********************************************************/

template <>
struct QuantizerFP16<SIMDLevel::NEON> : QuantizerFP16<SIMDLevel::NONE> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint16x4x2_t codei = vld1_u16_x2((const uint16_t*)(code + 2 * i));
        return simd8float32(
                float32x4x2_t{
                        vcvt_f32_f16(vreinterpret_f16_u16(codei.val[0])),
                        vcvt_f32_f16(vreinterpret_f16_u16(codei.val[1]))});
    }
};

/**********************************************************
 * BF16 Quantizer
 **********************************************************/

template <>
struct QuantizerBF16<SIMDLevel::NEON> : QuantizerBF16<SIMDLevel::NONE> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint16x4x2_t codei = vld1_u16_x2((const uint16_t*)(code + 2 * i));
        return simd8float32(
                float32x4x2_t{
                        vreinterpretq_f32_u32(
                                vshlq_n_u32(vmovl_u16(codei.val[0]), 16)),
                        vreinterpretq_f32_u32(
                                vshlq_n_u32(vmovl_u16(codei.val[1]), 16))});
    }
};

/**********************************************************
 * 8bit Direct Quantizer
 **********************************************************/

template <>
struct Quantizer8bitDirect<SIMDLevel::NEON>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint8x8_t x8 = vld1_u8((const uint8_t*)(code + i));
        uint16x8_t y8 = vmovl_u8(x8);
        uint16x4_t y8_0 = vget_low_u16(y8);
        uint16x4_t y8_1 = vget_high_u16(y8);
        return simd8float32(
                float32x4x2_t{
                        vcvtq_f32_u32(vmovl_u16(y8_0)),
                        vcvtq_f32_u32(vmovl_u16(y8_1))});
    }
};

/**********************************************************
 * 8bit Direct Signed Quantizer
 **********************************************************/

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::NEON>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<SIMDLevel::NONE>(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint8x8_t x8 = vld1_u8((const uint8_t*)(code + i));
        uint16x8_t y8 = vmovl_u8(x8);
        int16x8_t z8 = vreinterpretq_s16_u16(
                vsubq_u16(y8, vdupq_n_u16(128))); // subtract 128 from all lanes
        int16x4_t z8_0 = vget_low_s16(z8);
        int16x4_t z8_1 = vget_high_s16(z8);
        return simd8float32(
                float32x4x2_t{
                        vcvtq_f32_s32(vmovl_s16(z8_0)),
                        vcvtq_f32_s32(vmovl_s16(z8_1))});
    }
};

/**********************************************************
 * Similarities (L2 and IP)
 **********************************************************/

template <>
struct SimilarityL2<SIMDLevel::NEON> {
    static constexpr int simdwidth = 8;
    static constexpr SIMDLevel simd_level = SIMDLevel::NEON;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y), yi(nullptr) {}

    simd8float32 accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(simd8float32 x) {
        simd8float32 yiv(yi);
        yi += 8;
        simd8float32 tmp = yiv - x;
        accu8 = accu8 + tmp * tmp;
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            simd8float32 x,
            simd8float32 y_2) {
        simd8float32 tmp = y_2 - x;
        accu8 = accu8 + tmp * tmp;
    }

    FAISS_ALWAYS_INLINE float result_8() {
        return horizontal_add(accu8);
    }
};

template <>
struct SimilarityIP<SIMDLevel::NEON> {
    static constexpr int simdwidth = 8;
    static constexpr SIMDLevel simd_level = SIMDLevel::NEON;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    explicit SimilarityIP(const float* y) : y(y), yi(nullptr) {}

    simd8float32 accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(simd8float32 x) {
        simd8float32 yiv(yi);
        yi += 8;
        accu8 = accu8 + yiv * x;
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            simd8float32 x1,
            simd8float32 x2) {
        accu8 = accu8 + x1 * x2;
    }

    FAISS_ALWAYS_INLINE float result_8() {
        return horizontal_add(accu8);
    }
};

/**********************************************************
 * Distance Computers
 **********************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::NEON> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 xi = quant.reconstruct_8_components(code, i);
            sim.add_8_components(xi);
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 x1 = quant.reconstruct_8_components(code1, i);
            simd8float32 x2 = quant.reconstruct_8_components(code2, i);
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
struct DistanceComputerByte<Similarity, SIMDLevel::NEON> : SQDistanceComputer {
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

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::NEON
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#endif // COMPILE_SIMD_ARM_NEON
