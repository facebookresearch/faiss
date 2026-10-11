/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_NEON

#include <faiss/impl/simdlib/simdlib_neon.h>

#include <algorithm>
#include <array>
#include <cstring>

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

namespace faiss {

namespace scalar_quantizer {

using simd8float32 = faiss::simd8float32_tpl<SIMDLevel::ARM_NEON>;

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

FAISS_ALWAYS_INLINE uint32_t load_u24(const uint8_t* ptr) {
    return static_cast<uint32_t>(ptr[0]) |
            (static_cast<uint32_t>(ptr[1]) << 8) |
            (static_cast<uint32_t>(ptr[2]) << 16);
}

FAISS_ALWAYS_INLINE void unpack_8x1bit_to_u8(
        const uint8_t* code,
        int i,
        uint8_t out[8]) {
    const uint8_t packed = code[static_cast<size_t>(i) >> 3];
    for (size_t j = 0; j < 8; ++j) {
        out[j] = (packed >> j) & 0x1;
    }
}

FAISS_ALWAYS_INLINE void unpack_8x2bit_to_u8(
        const uint8_t* code,
        int i,
        uint8_t out[8]) {
    const uint16_t packed = load_u16(code + (static_cast<size_t>(i) >> 2));
    for (size_t j = 0; j < 8; ++j) {
        out[j] = (packed >> (2 * j)) & 0x3;
    }
}

FAISS_ALWAYS_INLINE void unpack_8x3bit_to_u8(
        const uint8_t* code,
        int i,
        uint8_t out[8]) {
    const uint32_t packed =
            load_u24(code + ((static_cast<size_t>(i) >> 3) * 3));
    for (size_t j = 0; j < 8; ++j) {
        out[j] = (packed >> (3 * j)) & 0x7;
    }
}

FAISS_ALWAYS_INLINE void unpack_8x4bit_to_u8(
        const uint8_t* code,
        int i,
        uint8_t out[8]) {
    const uint32_t packed = load_u32(code + (static_cast<size_t>(i) >> 1));
    for (size_t j = 0; j < 8; ++j) {
        out[j] = (packed >> (4 * j)) & 0xf;
    }
}

FAISS_ALWAYS_INLINE simd8float32
gather_8_components(const float* codebook, const uint8_t indices[8]) {
    float result[8];
    for (size_t j = 0; j < 8; ++j) {
        result[j] = codebook[indices[j]];
    }
    return simd8float32(
            float32x4x2_t{vld1q_f32(result), vld1q_f32(result + 4)});
}

} // namespace

/**********************************************************
 * Codecs
 **********************************************************/

template <>
struct Codec8bit<SIMDLevel::ARM_NEON> : Codec8bit<SIMDLevel::NONE> {
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
struct Codec4bit<SIMDLevel::ARM_NEON> : Codec4bit<SIMDLevel::NONE> {
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
struct Codec6bit<SIMDLevel::ARM_NEON> : Codec6bit<SIMDLevel::NONE> {
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
        SIMDLevel::ARM_NEON>
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

    /// Raw codec decode without denormalization (for pre-decode opt)
    FAISS_ALWAYS_INLINE simd8float32
    decode_8_raw(const uint8_t* code, int i) const {
        return Codec::decode_8_components(code, i);
    }
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::ARM_NEON>
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
 * Lloyd-Max scalar quantizer
 **********************************************************/

// NEON Lloyd-Max: decode via gather, encode stays scalar.
// NEON doesn't have movemask so 1-bit encode is also scalar.
#define DEFINE_LLOYD_MAX_NEON_SPECIALIZATION(NBITS, UNPACK_FN)               \
    template <>                                                              \
    struct QuantizerLloydMax<NBITS, SIMDLevel::ARM_NEON>                     \
            : QuantizerLloydMax<NBITS, SIMDLevel::NONE> {                    \
        using Base = QuantizerLloydMax<NBITS, SIMDLevel::NONE>;              \
                                                                             \
        QuantizerLloydMax(size_t d, const std::vector<float>& trained)       \
                : Base(d, trained) {                                         \
            assert(d % 8 == 0);                                              \
        }                                                                    \
                                                                             \
        FAISS_ALWAYS_INLINE simd8float32                                     \
        reconstruct_8_components(const uint8_t* code, int i) const {         \
            uint8_t indices[8];                                              \
            UNPACK_FN(code, i, indices);                                     \
            return gather_8_components(this->centroids, indices);            \
        }                                                                    \
                                                                             \
        void decode_vector(const uint8_t* code, float* x) const final {      \
            for (size_t i = 0; i < this->d; i += 8) {                        \
                simd8float32 xi =                                            \
                        reconstruct_8_components(code, static_cast<int>(i)); \
                vst1q_f32(x + i, xi.data.val[0]);                            \
                vst1q_f32(x + i + 4, xi.data.val[1]);                        \
            }                                                                \
        }                                                                    \
    }

DEFINE_LLOYD_MAX_NEON_SPECIALIZATION(1, unpack_8x1bit_to_u8);
DEFINE_LLOYD_MAX_NEON_SPECIALIZATION(2, unpack_8x2bit_to_u8);
DEFINE_LLOYD_MAX_NEON_SPECIALIZATION(3, unpack_8x3bit_to_u8);
DEFINE_LLOYD_MAX_NEON_SPECIALIZATION(4, unpack_8x4bit_to_u8);

#undef DEFINE_LLOYD_MAX_NEON_SPECIALIZATION

template <>
struct QuantizerLloydMax<8, SIMDLevel::ARM_NEON>
        : QuantizerLloydMax<8, SIMDLevel::NONE> {
    using Base = QuantizerLloydMax<8, SIMDLevel::NONE>;

    QuantizerLloydMax(size_t d, const std::vector<float>& trained)
            : Base(d, trained) {
        assert(d % 8 == 0);
    }

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint8_t indices[8];
        std::memcpy(indices, code + static_cast<size_t>(i), sizeof(indices));
        return gather_8_components(this->centroids, indices);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < this->d; i += 8) {
            simd8float32 xi =
                    reconstruct_8_components(code, static_cast<int>(i));
            vst1q_f32(x + i, xi.data.val[0]);
            vst1q_f32(x + i + 4, xi.data.val[1]);
        }
    }
};

/**********************************************************
 * FP16 Quantizer
 **********************************************************/

template <>
struct QuantizerFP16<SIMDLevel::ARM_NEON> : QuantizerFP16<SIMDLevel::NONE> {
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
struct QuantizerBF16<SIMDLevel::ARM_NEON> : QuantizerBF16<SIMDLevel::NONE> {
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
struct Quantizer8bitDirect<SIMDLevel::ARM_NEON>
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
struct Quantizer8bitDirectSigned<SIMDLevel::ARM_NEON>
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
struct SimilarityL2<SIMDLevel::ARM_NEON> {
    static constexpr int simdwidth = 8;
    static constexpr SIMDLevel simd_level = SIMDLevel::ARM_NEON;
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

    static void adjust_query_for_raw_decode(
            const float* x,
            float* q_adj,
            size_t d,
            float vmin,
            float vdiff,
            float& scale_factor,
            float& bias) {
        float inv_vdiff = (vdiff != 0) ? 1.0f / vdiff : 0.0f;
        for (size_t i = 0; i < d; i++) {
            q_adj[i] = (x[i] - vmin) * inv_vdiff;
        }
        scale_factor = vdiff * vdiff;
        bias = 0;
    }
};

template <>
struct SimilarityIP<SIMDLevel::ARM_NEON> {
    static constexpr int simdwidth = 8;
    static constexpr SIMDLevel simd_level = SIMDLevel::ARM_NEON;
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

    static void adjust_query_for_raw_decode(
            const float* x,
            float* q_adj,
            size_t d,
            float vmin,
            float vdiff,
            float& scale_factor,
            float& bias) {
        float sum_q = 0;
        for (size_t i = 0; i < d; i++) {
            q_adj[i] = x[i];
            sum_q += x[i];
        }
        scale_factor = vdiff;
        bias = vmin * sum_q;
    }
};

/**********************************************************
 * Distance Computers
 **********************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::ARM_NEON>
        : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    // Pre-adjusted query buffer for uniform quantizers
    std::vector<float> q_adj;
    float scale_factor = 0;
    float bias = 0;

    static constexpr bool has_decode_raw() {
        return requires(const Quantizer& q, const uint8_t* c, int i) {
            { q.decode_8_raw(c, i) };
        };
    }

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {
        if constexpr (has_decode_raw()) {
            q_adj.resize(d);
        }
    }

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
        if constexpr (has_decode_raw()) {
            Sim::adjust_query_for_raw_decode(
                    x,
                    q_adj.data(),
                    quant.d,
                    quant.vmin,
                    quant.vdiff,
                    scale_factor,
                    bias);
        }
    }

    float query_to_code_predecoded(const uint8_t* code) const {
        Similarity sim(q_adj.data());
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 xi = quant.decode_8_raw(code, i);
            sim.add_8_components(xi);
        }
        return bias + scale_factor * sim.result_8();
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        if constexpr (has_decode_raw()) {
            return query_to_code_predecoded(code);
        } else {
            return compute_distance(q, code);
        }
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

        sim0.begin_8();
        sim1.begin_8();
        sim2.begin_8();
        sim3.begin_8();

        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 xi0 = quant.reconstruct_8_components(code_0, i);
            simd8float32 xi1 = quant.reconstruct_8_components(code_1, i);
            simd8float32 xi2 = quant.reconstruct_8_components(code_2, i);
            simd8float32 xi3 = quant.reconstruct_8_components(code_3, i);
            sim0.add_8_components(xi0);
            sim1.add_8_components(xi1);
            sim2.add_8_components(xi2);
            sim3.add_8_components(xi3);
        }

        dis0 = sim0.result_8();
        dis1 = sim1.result_8();
        dis2 = sim2.result_8();
        dis3 = sim3.result_8();
    }
};

// Byte-domain kernels for QT_8bit_direct{,_signed}. The dispatch only
// selects them when d % 16 == 0, so no loop needs a tail.

namespace {

template <MetricType metric, bool biased, size_t N, bool split = N == 1>
FAISS_ALWAYS_INLINE std::array<uint32x4_t, N> neon_byte_accumulators(
        const uint8_t* query,
        const std::array<const uint8_t*, N>& codes,
        int start,
        int end) {
    static_assert(!split || N == 1);
    const uint8x16_t bias = vdupq_n_u8(0x80);
    std::array<uint32x4_t, N> accumulators;
    // Longer single-candidate loops use independent low/high chains.
    std::array<uint32x4_t, N> high_accumulators;
    for (size_t j = 0; j < N; ++j) {
        accumulators[j] = vdupq_n_u32(0);
        if constexpr (split) {
            high_accumulators[j] = vdupq_n_u32(0);
        }
    }
    auto accumulate = [&](int i) {
        uint8x16_t q = vld1q_u8(query + i);
        if constexpr (metric == METRIC_INNER_PRODUCT && biased) {
            q = veorq_u8(q, bias);
        }
        for (size_t j = 0; j < N; ++j) {
            auto& high = split ? high_accumulators[j] : accumulators[j];
            uint8x16_t c = vld1q_u8(codes[j] + i);
            if constexpr (metric == METRIC_L2) {
                const uint8x16_t diff = vabdq_u8(q, c);
                accumulators[j] = vpadalq_u16(
                        accumulators[j],
                        vmull_u8(vget_low_u8(diff), vget_low_u8(diff)));
                high = vpadalq_u16(
                        high, vmull_u8(vget_high_u8(diff), vget_high_u8(diff)));
            } else if constexpr (biased) {
                // Stored signed bytes are value+128; xor removes the bias.
                const int8x16_t qs = vreinterpretq_s8_u8(q);
                const int8x16_t cs = vreinterpretq_s8_u8(veorq_u8(c, bias));
                accumulators[j] = vreinterpretq_u32_s32(vpadalq_s16(
                        vreinterpretq_s32_u32(accumulators[j]),
                        vmull_s8(vget_low_s8(qs), vget_low_s8(cs))));
                high = vreinterpretq_u32_s32(vpadalq_s16(
                        vreinterpretq_s32_u32(high),
                        vmull_s8(vget_high_s8(qs), vget_high_s8(cs))));
            } else {
                accumulators[j] = vpadalq_u16(
                        accumulators[j],
                        vmull_u8(vget_low_u8(q), vget_low_u8(c)));
                high = vpadalq_u16(
                        high, vmull_u8(vget_high_u8(q), vget_high_u8(c)));
            }
        }
    };
    if constexpr (split) {
        int i = start;
        for (; end - i >= 32; i += 32) {
            accumulate(i);
            accumulate(i + 16);
        }
        if (i < end) {
            accumulate(i);
        }
        accumulators[0] = vaddq_u32(accumulators[0], high_accumulators[0]);
    } else {
        for (int i = start; i < end; i += 16) {
            accumulate(i);
        }
    }
    return accumulators;
}

template <MetricType metric, bool biased, size_t N>
FAISS_ALWAYS_INLINE void neon_byte_distances(
        const uint8_t* query,
        const std::array<const uint8_t*, N>& codes,
        int d,
        float* distances) {
    if constexpr (N == 1) {
        // Avoid loop setup and the wide-vector guard for one or two chunks.
        if (d <= 32) {
            if (d == 16) {
                const auto accumulators =
                        neon_byte_accumulators<metric, biased, N, false>(
                                query, codes, 0, 16);
                distances[0] = static_cast<float>(
                        vaddvq_s32(vreinterpretq_s32_u32(accumulators[0])));
                return;
            }
            if (d == 32) {
                const auto accumulators =
                        neon_byte_accumulators<metric, biased, N, false>(
                                query, codes, 0, 32);
                distances[0] = static_cast<float>(
                        vaddvq_s32(vreinterpretq_s32_u32(accumulators[0])));
                return;
            }
        }
    }
    // Each block's entire sum fits int32 (32768 * 255^2 < INT32_MAX).
    constexpr int block_size = 32768;
    if (d <= block_size) {
        const auto accumulators =
                neon_byte_accumulators<metric, biased, N>(query, codes, 0, d);
        for (size_t j = 0; j < N; ++j) {
            distances[j] = static_cast<float>(
                    vaddvq_s32(vreinterpretq_s32_u32(accumulators[j])));
        }
        return;
    }

    // Widen before adding blocks, not after a full-vector reduction.
    std::array<int64_t, N> totals{};
    for (int start = 0; start < d;) {
        const int end = start + std::min(d - start, block_size);
        const auto accumulators = neon_byte_accumulators<metric, biased, N>(
                query, codes, start, end);
        for (size_t j = 0; j < N; ++j) {
            totals[j] += vaddvq_s32(vreinterpretq_s32_u32(accumulators[j]));
        }
        start = end;
    }
    for (size_t j = 0; j < N; ++j) {
        distances[j] = static_cast<float>(totals[j]);
    }
}

} // namespace

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::ARM_NEON>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {
        FAISS_THROW_IF_NOT(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE float compute_code_distance(
            const uint8_t* code1,
            const uint8_t* code2) const {
        float distance;
        neon_byte_distances<Sim::metric_type, false, 1>(
                code1, {code2}, d, &distance);
        return distance;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    float compute_distance(const float* x, const uint8_t* code) {
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

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const override {
        float distances[4];
        neon_byte_distances<Sim::metric_type, false, 4>(
                tmp.data(), {code_0, code_1, code_2, code_3}, d, distances);
        dis0 = distances[0];
        dis1 = distances[1];
        dis2 = distances[2];
        dis3 = distances[3];
    }
};

template <class Similarity>
struct DistanceComputerByteSigned<Similarity, SIMDLevel::ARM_NEON>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByteSigned(int d, const std::vector<float>&)
            : d(d), tmp(d) {
        FAISS_THROW_IF_NOT(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE float compute_code_distance(
            const uint8_t* code1,
            const uint8_t* code2) const {
        float distance;
        neon_byte_distances<Sim::metric_type, true, 1>(
                code1, {code2}, d, &distance);
        return distance;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = uint8_t(int(x[i]) + 128);
        }
    }

    float compute_distance(const float* x, const uint8_t* code) {
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

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const override {
        float distances[4];
        neon_byte_distances<Sim::metric_type, true, 4>(
                tmp.data(), {code_0, code_1, code_2, code_3}, d, distances);
        dis0 = distances[0];
        dis1 = distances[1];
        dis2 = distances[2];
        dis3 = distances[3];
    }
};

/**********************************************************
 * TurboQuant masked_sum NEON specialization (scalar fallback)
 **********************************************************/

template <SIMDLevel SL0>
float turboq_masked_sum(const float* arr, const uint8_t* bits, size_t d);

template <>
float turboq_masked_sum<SIMDLevel::ARM_NEON>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    float result = 0;
    for (size_t byte_idx = 0; byte_idx < (d + 7) / 8; byte_idx++) {
        uint8_t b = bits[byte_idx];
        size_t base = byte_idx * 8;
        size_t end = std::min(base + 8, d);
        for (size_t j = base; j < end; j++) {
            if (b & (1 << (j - base))) {
                result += arr[j];
            }
        }
    }
    return result;
}

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::ARM_NEON
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#ifdef COMPILE_SIMD_ARM_SVE

// ARM_SVE: SVE is a superset of NEON. Forward to the NEON implementation
// until a dedicated SVE specialization is written.

namespace faiss {
namespace scalar_quantizer {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<SIMDLevel::ARM_SVE>(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return sq_select_quantizer<SIMDLevel::ARM_NEON>(qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
SQDistanceComputer* sq_select_distance_computer<SIMDLevel::ARM_SVE>(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return sq_select_distance_computer<SIMDLevel::ARM_NEON>(
            metric, qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
InvertedListScanner* sq_select_InvertedListScanner<SIMDLevel::ARM_SVE>(
        QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    return sq_select_InvertedListScanner<SIMDLevel::ARM_NEON>(
            qtype,
            mt,
            d,
            code_size,
            trained,
            quantizer,
            store_pairs,
            sel,
            by_residual);
}

} // namespace scalar_quantizer
} // namespace faiss

#endif // COMPILE_SIMD_ARM_SVE

#endif // COMPILE_SIMD_ARM_NEON
