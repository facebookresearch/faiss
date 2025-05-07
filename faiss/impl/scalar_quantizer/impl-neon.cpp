/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>

#if defined(__aarch64__)
#if defined(__GNUC__) && __GNUC__ < 8
#warning \
        "Cannot enable NEON optimizations in scalar quantizer if the compiler is GCC<8"
#else
#define USE_NEON
#endif
#endif

namespace faiss {

namespace scalar_quantizer {
/******************************** Codec specializations */

template <>
struct Codec8bit<SIMDLevel::ARM_NEON> {
    static FAISS_ALWAYS_INLINE decode_8_components(const uint8_t* code, int i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] = decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return simd8float32(float32x4x2_t{res1, res2});
    }
};

template <>
struct Codec4bit<SIMDLevel::ARM_NEON> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, int i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] = decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return simd8float32({res1, res2});
    }
};

template <>
struct Codec6bit<SIMDLevel::ARM_NEON> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, int i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] = decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return simd8float32(float32x4x2_t({res1, res2}));
    }
};
/******************************** Quantizatoin specializations */

template <class Codec>
struct QuantizerT<Codec, QScaling::UNIFORM, SIMDLevel::ARM_NEON>
        : QuantizerT<Codec, QScaling::UNIFORM, SIMDLevel::NONE> {
    QuantizerT(size_t d, const std::vector<float>& trained)
            : QuantizerT<Codec, QScaling::UNIFORM, 1>(d, trained) {}

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

template <class Codec>
struct QuantizerT<Codec, QScaling::NON_UNIFORM, SIMDLevel::ARM_NEON>
        : QuantizerT<Codec, QScaling::NON_UNIFORM, SIMDLevel::NONE> {
    QuantizerT(size_t d, const std::vector<float>& trained)
            : QuantizerT<Codec, QScaling::NON_UNIFORM, 1>(d, trained) {}

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

template <>
struct QuantizerFP16<SIMDLevel::ARM_NEON> : QuantizerFP16<SIMDLevel::NONE> {
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

template <>
struct QuantizerBF16<SIMDLevel::ARM_NEON> : QuantizerBF16<SIMDLevel::NONE> {
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

template <>
struct Quantizer8bitDirect<SIMDLevel::ARM_NEON>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
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

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::ARM_NEON>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
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

/****************************************** Specialization of similarities */

template <>
struct SimilarityL2<SIMDLevel::ARM_NEON> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;
    explicit SimilarityL2(const float* y) : y(y) {}
    simd8float32 accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8 = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(simd8float32 x) {
        float32x4x2_t yiv = vld1q_f32_x2(yi);
        yi += 8;

        float32x4_t sub0 = vsubq_f32(yiv.val[0], x.val[0]);
        float32x4_t sub1 = vsubq_f32(yiv.val[1], x.val[1]);

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], sub0, sub0);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], sub1, sub1);

        accu8 = simd8float32({accu8_0, accu8_1});
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            simd8float32 x,
            simd8float32 y) {
        float32x4_t sub0 = vsubq_f32(y.val[0], x.val[0]);
        float32x4_t sub1 = vsubq_f32(y.val[1], x.val[1]);

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], sub0, sub0);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], sub1, sub1);

        accu8 = simd8float32({accu8_0, accu8_1});
    }

    FAISS_ALWAYS_INLINE float result_8() {
        float32x4_t sum_0 = vpaddq_f32(accu8.data.val[0], accu8.data.val[0]);
        float32x4_t sum_1 = vpaddq_f32(accu8.data.val[1], accu8.data.val[1]);

        float32x4_t sum2_0 = vpaddq_f32(sum_0, sum_0);
        float32x4_t sum2_1 = vpaddq_f32(sum_1, sum_1);
        return vgetq_lane_f32(sum2_0, 0) + vgetq_lane_f32(sum2_1, 0);
    }
};

template <>
struct SimilarityIP<SIMDLevel::ARM_NEON> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    explicit SimilarityIP(const float* y) : y(y) {}
    float32x4x2_t accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8 = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(float32x4x2_t x) {
        float32x4x2_t yiv = vld1q_f32_x2(yi);
        yi += 8;

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], yiv.val[0], x.val[0]);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], yiv.val[1], x.val[1]);
        accu8 = {accu8_0, accu8_1};
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            float32x4x2_t x1,
            float32x4x2_t x2) {
        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], x1.val[0], x2.val[0]);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], x1.val[1], x2.val[1]);
        accu8 = {accu8_0, accu8_1};
    }

    FAISS_ALWAYS_INLINE float result_8() {
        float32x4x2_t sum = {
                vpaddq_f32(accu8.val[0], accu8.val[0]),
                vpaddq_f32(accu8.val[1], accu8.val[1])};

        float32x4x2_t sum2 = {
                vpaddq_f32(sum.val[0], sum.val[0]),
                vpaddq_f32(sum.val[1], sum.val[1])};
        return vgetq_lane_f32(sum2.val[0], 0) + vgetq_lane_f32(sum2.val[1], 0);
    }
};

/****************************************** Specialization of distance computers
 */

// this is the same code as the AVX2 version... Possible to mutualize?
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
struct DistanceComputerByte<Similarity, SIMDLevel::ARM_NEON>
        : SQDistanceComputer {
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
