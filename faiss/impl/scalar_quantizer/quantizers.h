/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/simdlib.h>

namespace faiss {

namespace scalar_quantizer {

using QuantizerType = ScalarQuantizer::QuantizerType;

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

enum class QuantizerTemplateScaling { UNIFORM = 0, NON_UNIFORM = 1 };

template <class Codec, QuantizerTemplateScaling SCALING, SIMDLevel SL>
struct QuantizerTemplate {};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::NONE> : ScalarQuantizer::SQuantizer {
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

    FAISS_ALWAYS_INLINE float reconstruct_component(
            const uint8_t* code,
            size_t i) const {
        float xi = Codec::decode_component(code, i);
        return vmin + xi * vdiff;
    }
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::NONE> : ScalarQuantizer::SQuantizer {
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

    FAISS_ALWAYS_INLINE float reconstruct_component(
            const uint8_t* code,
            size_t i) const {
        float xi = Codec::decode_component(code, i);
        return vmin[i] + xi * vdiff[i];
    }
};

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <SIMDLevel SL>
struct QuantizerFP16;

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

    FAISS_ALWAYS_INLINE float reconstruct_component(
            const uint8_t* code,
            size_t i) const {
        return decode_fp16(((uint16_t*)code)[i]);
    }
};

template <SIMDLevel SL>
struct QuantizerFP16 : QuantizerFP16<SIMDLevel::NONE> {
    using QuantizerFP16<SIMDLevel::NONE>::QuantizerFP16;
};

/*******************************************************************
 * BF16 quantizer
 *******************************************************************/

template <SIMDLevel SL>
struct QuantizerBF16;

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

    FAISS_ALWAYS_INLINE float reconstruct_component(
            const uint8_t* code,
            size_t i) const {
        return decode_bf16(((uint16_t*)code)[i]);
    }
};

template <SIMDLevel SL>
struct QuantizerBF16 : QuantizerBF16<SIMDLevel::NONE> {
    using QuantizerBF16<SIMDLevel::NONE>::QuantizerBF16;
};

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <SIMDLevel SL>
struct Quantizer8bitDirect;

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

    FAISS_ALWAYS_INLINE float reconstruct_component(
            const uint8_t* code,
            size_t i) const {
        return code[i];
    }
};

template <SIMDLevel SL>
struct Quantizer8bitDirect : Quantizer8bitDirect<SIMDLevel::NONE> {
    using Quantizer8bitDirect<SIMDLevel::NONE>::Quantizer8bitDirect;
};

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/

template <SIMDLevel SL>
struct Quantizer8bitDirectSigned;

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

    FAISS_ALWAYS_INLINE float reconstruct_component(
            const uint8_t* code,
            size_t i) const {
        return code[i] - 128;
    }
};

template <SIMDLevel SL>
struct Quantizer8bitDirectSigned : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    using Quantizer8bitDirectSigned<SIMDLevel::NONE>::Quantizer8bitDirectSigned;
};

/*******************************************************************
 * Selection function
 *******************************************************************/

// declare for all levels
template <SIMDLevel SL>
ScalarQuantizer::SQuantizer* sq_select_quantizer(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

} // namespace scalar_quantizer

} // namespace faiss
