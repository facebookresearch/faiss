/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define THE_SIMD_LEVEL SIMDLevel::LOONGARCH_LASX

#include <faiss/impl/simdlib/simdlib_lasx.h>

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <bit>
#include <cstring>

namespace faiss::scalar_quantizer::detail {
inline uint32_t load_sq_word(const uint8_t* code, size_t n) {
    uint32_t word = 0;
    std::memcpy(&word, code, n);
    return word;
}

template <SIMDLevel SL>
struct LoongArchSQOps;
template <>
struct LoongArchSQOps<SIMDLevel::LOONGARCH_LASX> {
    using Vec = simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>;
    template <int Bits, bool normalized = true>
    static Vec decode_codes(const uint8_t* code) {
        Vec out;
        constexpr int bytes = Bits / 2;
        const uint32_t lo = load_sq_word(code, bytes),
                       hi = load_sq_word(code + bytes, bytes);
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};

        const __m256i words = __lasx_xvpermi_q(
                __lasx_xvreplgr2vr_w(hi), __lasx_xvreplgr2vr_w(lo), 0x20);

        const auto indices = __lasx_xvand_v(
                __lasx_xvsrl_w(words, __lasx_xvld(shifts, 0)),
                __lasx_xvreplgr2vr_w((1 << Bits) - 1));
        auto value = __lasx_xvffint_s_w(indices);
        if constexpr (normalized)
            value = __lasx_xvfdiv_s(
                    __lasx_xvfadd_s(
                            value, (__m256)__lasx_xvreplgr2vr_w(0x3f000000)),
                    (__m256)__lasx_xvreplgr2vr_w(
                            std::bit_cast<int32_t>(float((1 << Bits) - 1))));
        out.data = (__m256i)value;

        return out;
    }
    template <bool bf16>
    static Vec decode_half(const uint8_t* code) {
        Vec out;

        const auto raw = __lasx_xvpermi_q(
                __lasx_xvldrepl_d(code, 8), __lasx_xvldrepl_d(code, 0), 0x20);
        if constexpr (bf16)
            out.data =
                    __lasx_xvslli_w(__lasx_xvilvl_h(__lasx_xvldi(0), raw), 16);
        else
            out.data = (__m256i)__lasx_xvfcvtl_s_h(raw);
        return out;
    }
};
} // namespace faiss::scalar_quantizer::detail

namespace faiss::scalar_quantizer {

using simd8float32 = simd8float32_tpl<THE_SIMD_LEVEL>;

template <>
struct Codec8bit<THE_SIMD_LEVEL> : Codec8bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        return detail::LoongArchSQOps<THE_SIMD_LEVEL>::decode_codes<8>(
                code + i);
    }
};

template <>
struct Codec4bit<THE_SIMD_LEVEL> : Codec4bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        float values[8];
        for (size_t j = 0; j < 8; ++j) {
            values[j] =
                    Codec4bit<SIMDLevel::NONE>::decode_component(code, i + j);
        }
        return simd8float32(values);
    }
};

template <>
struct Codec6bit<THE_SIMD_LEVEL> : Codec6bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd8float32
    decode_8_components(const uint8_t* code, size_t i) {
        float values[8];
        for (size_t j = 0; j < 8; ++j) {
            values[j] =
                    Codec6bit<SIMDLevel::NONE>::decode_component(code, i + j);
        }
        return simd8float32(values);
    }
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::UNIFORM,
        THE_SIMD_LEVEL>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::NONE> {
    using Base = QuantizerTemplate<
            Codec,
            QuantizerTemplateScaling::UNIFORM,
            SIMDLevel::NONE>;
    using Base::Base;

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        return fmadd(
                Codec::decode_8_components(code, i),
                simd8float32(this->vdiff),
                simd8float32(this->vmin));
    }
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::NON_UNIFORM,
        THE_SIMD_LEVEL>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::NON_UNIFORM,
                  SIMDLevel::NONE> {
    using Base = QuantizerTemplate<
            Codec,
            QuantizerTemplateScaling::NON_UNIFORM,
            SIMDLevel::NONE>;
    using Base::Base;

    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        return fmadd(
                Codec::decode_8_components(code, i),
                simd8float32(this->vdiff + i),
                simd8float32(this->vmin + i));
    }
};

// Lloyd-Max encoding stays scalar, as in the NEON backend.
#define DEFINE_LLOYD_MAX_LOONGARCH_SPECIALIZATION(NBITS)                      \
    template <>                                                               \
    struct QuantizerLloydMax<NBITS, THE_SIMD_LEVEL>                           \
            : QuantizerLloydMax<NBITS, SIMDLevel::NONE> {                     \
        using Base = QuantizerLloydMax<NBITS, SIMDLevel::NONE>;               \
        using Base::Base;                                                     \
        FAISS_ALWAYS_INLINE simd8float32                                      \
        reconstruct_8_components(const uint8_t* code, int i) const {          \
            float values[8];                                                  \
            for (size_t j = 0; j < 8; ++j) {                                  \
                values[j] = this->centroids[this->decode_index(code, i + j)]; \
            }                                                                 \
            return simd8float32(values);                                      \
        }                                                                     \
        void decode_vector(const uint8_t* code, float* x) const final {       \
            for (size_t i = 0; i < this->d; i += 8) {                         \
                reconstruct_8_components(code, static_cast<int>(i))           \
                        .storeu(x + i);                                       \
            }                                                                 \
        }                                                                     \
    }

DEFINE_LLOYD_MAX_LOONGARCH_SPECIALIZATION(1);
DEFINE_LLOYD_MAX_LOONGARCH_SPECIALIZATION(2);
DEFINE_LLOYD_MAX_LOONGARCH_SPECIALIZATION(3);
DEFINE_LLOYD_MAX_LOONGARCH_SPECIALIZATION(4);
DEFINE_LLOYD_MAX_LOONGARCH_SPECIALIZATION(8);
#undef DEFINE_LLOYD_MAX_LOONGARCH_SPECIALIZATION

template <>
struct QuantizerFP16<THE_SIMD_LEVEL> : QuantizerFP16<SIMDLevel::NONE> {
    using QuantizerFP16<SIMDLevel::NONE>::QuantizerFP16;
    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        return detail::LoongArchSQOps<THE_SIMD_LEVEL>::decode_half<false>(
                code + 2 * i);
    }
};

template <>
struct QuantizerBF16<THE_SIMD_LEVEL> : QuantizerBF16<SIMDLevel::NONE> {
    using QuantizerBF16<SIMDLevel::NONE>::QuantizerBF16;
    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        return detail::LoongArchSQOps<THE_SIMD_LEVEL>::decode_half<true>(
                code + 2 * i);
    }
};

template <>
struct Quantizer8bitDirect<THE_SIMD_LEVEL>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
    using Quantizer8bitDirect<SIMDLevel::NONE>::Quantizer8bitDirect;
    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        return detail::LoongArchSQOps<THE_SIMD_LEVEL>::decode_codes<8, false>(
                code + i);
    }
};

template <>
struct Quantizer8bitDirectSigned<THE_SIMD_LEVEL>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    using Quantizer8bitDirectSigned<SIMDLevel::NONE>::Quantizer8bitDirectSigned;
    FAISS_ALWAYS_INLINE simd8float32
    reconstruct_8_components(const uint8_t* code, int i) const {
        return detail::LoongArchSQOps<THE_SIMD_LEVEL>::decode_codes<8, false>(
                       code + i) -
                simd8float32(128.f);
    }
};

template <>
struct SimilarityL2<THE_SIMD_LEVEL> {
    static constexpr int simdwidth = 8;
    static constexpr SIMDLevel simd_level = THE_SIMD_LEVEL;
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
struct SimilarityIP<THE_SIMD_LEVEL> {
    static constexpr int simdwidth = 8;
    static constexpr SIMDLevel simd_level = THE_SIMD_LEVEL;
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

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, THE_SIMD_LEVEL> : SQDistanceComputer {
    using Sim = Similarity;
    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            sim.add_8_components(quant.reconstruct_8_components(code, i));
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* a, const uint8_t* b) const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            sim.add_8_components_2(
                    quant.reconstruct_8_components(a, i),
                    quant.reconstruct_8_components(b, i));
        }
        return sim.result_8();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }

    float symmetric_dis(idx_t i, idx_t j) final {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }
};

} // namespace faiss::scalar_quantizer

#define THE_LEVEL_TO_DISPATCH THE_SIMD_LEVEL
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>
#undef THE_LEVEL_TO_DISPATCH
