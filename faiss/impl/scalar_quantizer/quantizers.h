/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/utils/simd_levels.h>

#include <faiss/impl/FaissAssert.h>

#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>

namespace faiss {

namespace scalar_quantizer {

using QuantizerType = ScalarQuantizer::QuantizerType;

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

enum class QScaling { UNIFORM = 0, NON_UNIFORM = 1 };

template <class Codec, QScaling SCALING, SIMDLevel LEVEL>
struct QuantizerT {};

template <class Codec>
struct QuantizerT<Codec, QScaling::UNIFORM, SIMDLevel::NONE>
        : ScalarQuantizer::SQuantizer {
    const size_t d;
    const float vmin, vdiff;

    QuantizerT(size_t d, const std::vector<float>& trained)
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

template <class Codec>
struct QuantizerT<Codec, QScaling::NON_UNIFORM, SIMDLevel::NONE>
        : ScalarQuantizer::SQuantizer {
    const size_t d;
    const float *vmin, *vdiff;

    QuantizerT(size_t d, const std::vector<float>& trained)
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

/*******************************************************************
 * Quantizers that are not based on codecs
 *******************************************************************/

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <SIMDLevel level>
struct QuantizerFP16 {};

template <>
struct QuantizerFP16<SIMDLevel::NONE> : ScalarQuantizer::SQuantizer {
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

/*******************************************************************
 * BF16 quantizer
 *******************************************************************/

template <SIMDLevel level>
struct QuantizerBF16 {};

template <>
struct QuantizerBF16<SIMDLevel::NONE> : ScalarQuantizer::SQuantizer {
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

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <SIMDLevel level>
struct Quantizer8bitDirect {};

template <>
struct Quantizer8bitDirect<SIMDLevel::NONE> : ScalarQuantizer::SQuantizer {
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

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/

template <SIMDLevel level>
struct Quantizer8bitDirectSigned {};

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::NONE>
        : ScalarQuantizer::SQuantizer {
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

template <SIMDLevel SL>
ScalarQuantizer::SQuantizer* select_quantizer_1(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    // constexpr SIMDLevel SL = INSTANCIATE_SIMD_LEVEL;
    constexpr QScaling NU = QScaling::NON_UNIFORM;
    constexpr QScaling U = QScaling::UNIFORM;
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerT<Codec8bit<SL>, NU, SL>(d, trained);

        case ScalarQuantizer::QT_6bit:
            return new QuantizerT<Codec6bit<SL>, NU, SL>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerT<Codec4bit<SL>, NU, SL>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerT<Codec8bit<SL>, U, SL>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerT<Codec4bit<SL>, U, SL>(d, trained);
        case ScalarQuantizer::QT_fp16:
            return new QuantizerFP16<SL>(d, trained);
        case ScalarQuantizer::QT_bf16:
            return new QuantizerBF16<SL>(d, trained);
        case ScalarQuantizer::QT_8bit_direct:
            return new Quantizer8bitDirect<SL>(d, trained);
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned<SL>(d, trained);
        default:
            FAISS_THROW_MSG("unknown qtype");
            return nullptr;
    }
}

// prevent implicit instanciation
extern template ScalarQuantizer::SQuantizer* select_quantizer_1<
        SIMDLevel::AVX2>(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

extern template ScalarQuantizer::SQuantizer* select_quantizer_1<
        SIMDLevel::AVX512>(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

} // namespace scalar_quantizer

} // namespace faiss
