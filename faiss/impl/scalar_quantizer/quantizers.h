/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/simd_levels.h>

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

    QuantizerTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in), vmin(trained[0]), vdiff(trained[1]) {}

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

    QuantizerTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in), vmin(trained.data()), vdiff(trained.data() + d_in) {}

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
 * TurboQuant MSE quantizer
 *******************************************************************/
template <int NBits, SIMDLevel SL>
struct QuantizerTurboQuantMSE;

template <int NBits>
struct QuantizerTurboQuantMSE<NBits, SIMDLevel::NONE>
        : ScalarQuantizer::SQuantizer {
    static_assert(NBits >= 1 && NBits <= 8);

    static constexpr size_t kCentroidsCount = size_t(1) << NBits;
    static constexpr uint16_t kIndexMask =
            static_cast<uint16_t>((1u << NBits) - 1);

    const size_t d;
    const float* centroids;
    const float* boundaries;

    QuantizerTurboQuantMSE(size_t d_in, const std::vector<float>& trained)
            : d(d_in), centroids(nullptr), boundaries(nullptr) {
        FAISS_THROW_IF_NOT(trained.size() == 2 * kCentroidsCount - 1);
        centroids = trained.data();
        boundaries = trained.data() + kCentroidsCount;
    }

    FAISS_ALWAYS_INLINE uint8_t select_index(float x) const {
        return static_cast<uint8_t>(
                std::upper_bound(
                        boundaries, boundaries + (kCentroidsCount - 1), x) -
                boundaries);
    }

    FAISS_ALWAYS_INLINE void encode_index(uint8_t idx, uint8_t* code, size_t i)
            const {
        const size_t bit_offset = i * NBits;
        const size_t byte_offset = bit_offset >> 3;
        const size_t bit_shift = bit_offset & 7;
        const uint16_t packed = static_cast<uint16_t>(idx & kIndexMask)
                << bit_shift;
        code[byte_offset] |= packed & 0xff;
        if (bit_shift + NBits > 8) {
            code[byte_offset + 1] |= packed >> 8;
        }
    }

    FAISS_ALWAYS_INLINE uint8_t
    decode_index(const uint8_t* code, size_t i) const {
        const size_t bit_offset = i * NBits;
        const size_t byte_offset = bit_offset >> 3;
        const size_t bit_shift = bit_offset & 7;

        uint16_t packed = code[byte_offset];
        if (bit_shift + NBits > 8) {
            packed |= static_cast<uint16_t>(code[byte_offset + 1]) << 8;
        }
        return static_cast<uint8_t>((packed >> bit_shift) & kIndexMask);
    }

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            encode_index(select_index(x[i]), code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = centroids[decode_index(code, i)];
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(
            const uint8_t* code,
            size_t i) const {
        return centroids[decode_index(code, i)];
    }
};

template <int NBits, SIMDLevel SL>
struct QuantizerTurboQuantMSE : QuantizerTurboQuantMSE<NBits, SIMDLevel::NONE> {
    using QuantizerTurboQuantMSE<NBits, SIMDLevel::NONE>::
            QuantizerTurboQuantMSE;
};

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <SIMDLevel SL>
struct QuantizerFP16;

template <>
struct QuantizerFP16<SIMDLevel::NONE> : ScalarQuantizer::SQuantizer {
    const size_t d;

    QuantizerFP16(size_t d_in, const std::vector<float>& /* unused */)
            : d(d_in) {}

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

    QuantizerBF16(size_t d_in, const std::vector<float>& /* unused */)
            : d(d_in) {}

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

    Quantizer8bitDirect(size_t d_in, const std::vector<float>& /* unused */)
            : d(d_in) {}

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

    Quantizer8bitDirectSigned(
            size_t d_in,
            const std::vector<float>& /* unused */)
            : d(d_in) {}

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
