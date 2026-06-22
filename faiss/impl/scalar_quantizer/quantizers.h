/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>

// Hack for MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/utils.h>

extern "C" {
int sgemm_(
        const char* transa,
        const char* transb,
        int* m,
        int* n,
        int* k,
        const float* alpha,
        const float* a,
        int* lda,
        const float* b,
        int* ldb,
        float* beta,
        float* c,
        int* ldc);
}

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

    uint8_t select_index(float x) const {
        return static_cast<uint8_t>(
                std::upper_bound(
                        boundaries, boundaries + (kCentroidsCount - 1), x) -
                boundaries);
    }

    void encode_index(uint8_t idx, uint8_t* code, size_t i) const {
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

    uint8_t decode_index(const uint8_t* code, size_t i) const {
        const size_t bit_offset = i * NBits;
        const size_t byte_offset = bit_offset >> 3;
        const size_t bit_shift = bit_offset & 7;

        uint16_t packed = code[byte_offset];
        if (bit_shift + NBits > 8) {
            packed |= static_cast<uint16_t>(code[byte_offset + 1]) << 8;
        }
        return static_cast<uint8_t>((packed >> bit_shift) & kIndexMask);
    }

    void encode_vector(const float* x, uint8_t* code) const override {
        for (size_t i = 0; i < d; i++) {
            encode_index(select_index(x[i]), code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const override {
        for (size_t i = 0; i < d; i++) {
            x[i] = centroids[decode_index(code, i)];
        }
    }

    float reconstruct_component(const uint8_t* code, size_t i) const {
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

    void encode_vector(const float* x, uint8_t* code) const override {
        encode_bf16_simd(x, (uint16_t*)code, d);
    }

    void decode_vector(const uint8_t* code, float* x) const override {
        decode_bf16_simd((const uint16_t*)code, x, d);
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

template <>
struct QuantizerBF16<SIMDLevel::AVX512>;
template <>
struct QuantizerBF16<SIMDLevel::AVX512_SPR>;

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
 * Full TurboQuant (MSE + QJL) quantizer
 *
 * NBits = total bits per dimension (2-5).
 *   MSE bits = NBits - 1,  QJL bits = 1.
 *
 * Trained vector layout:
 *   [centroids (k floats), boundaries (k-1 floats),
 *    seed_lo (float), seed_hi (float), qjl_type (float)]
 * where k = 2^(NBits-1).
 *******************************************************************/

FAISS_PACK_STRUCTS_BEGIN
struct SQTurboQFactors {
    float norm = 0;
    float gamma = 0;
};
FAISS_PACK_STRUCTS_END

template <int NBits, SIMDLevel SL>
struct QuantizerTurboQuantFull;

template <int NBits>
struct QuantizerTurboQuantFull<NBits, SIMDLevel::NONE>
        : ScalarQuantizer::SQuantizer {
    static_assert(NBits >= 2 && NBits <= 5);

    static constexpr int kMSEBits = NBits - 1;
    static constexpr size_t kCentroidsCount = size_t(1) << kMSEBits;

    const size_t d;
    const float* centroids;
    const float* boundaries;

    // QJL projection type: 0 = FWHT, 2 = Random Rotation
    uint8_t qjl_type;

    // FWHT state (qjl_type == 0)
    size_t padded_d;
    std::vector<float> fwht_signs;

    // Random Rotation state (qjl_type == 2)
    std::vector<float> rr_matrix; // d x d orthogonal matrix (row-major)

    size_t mse_plane_bytes; // bytes for one bit-plane of d bits
    size_t mse_total_bytes; // kMSEBits * mse_plane_bytes
    size_t qjl_plane_bytes;

    QuantizerTurboQuantFull(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              centroids(trained.data()),
              boundaries(trained.data() + kCentroidsCount) {
        // trained = [centroids(k), boundaries(k-1), seed_lo, seed_hi, qjl_type]
        size_t k = kCentroidsCount;
        FAISS_THROW_IF_NOT(trained.size() == 2 * k - 1 + 3);

        mse_plane_bytes = (d + 7) / 8;
        mse_total_bytes = kMSEBits * mse_plane_bytes;
        qjl_plane_bytes = (d + 7) / 8;

        // Extract seed from trained
        uint64_t seed = ScalarQuantizer::TurboQuantRefine::unpack_seed(
                trained[2 * k - 1], trained[2 * k]);
        qjl_type = static_cast<uint8_t>(trained[2 * k + 1]);

        if (qjl_type == 0) {
            // FWHT mode
            padded_d = 1;
            while (padded_d < d) {
                padded_d <<= 1;
            }
            fwht_signs.resize(padded_d);
            RandomGenerator rng(seed);
            for (size_t i = 0; i < padded_d; i++) {
                fwht_signs[i] = (rng.rand_int(2) == 0) ? 1.0f : -1.0f;
            }
        } else {
            // Random Rotation mode
            padded_d = d; // no padding needed for dense multiply
            rr_matrix.resize(d * d);
            float_randn(rr_matrix.data(), d * d, static_cast<int64_t>(seed));
            matrix_qr(
                    static_cast<int>(d), static_cast<int>(d), rr_matrix.data());
        }
    }

    void fwht_inplace(float* x, size_t n) const {
        for (size_t h = 1; h < n; h <<= 1) {
            for (size_t i = 0; i < n; i += h << 1) {
                for (size_t j = i; j < i + h; j++) {
                    float a = x[j];
                    float b = x[j + h];
                    x[j] = a + b;
                    x[j + h] = a - b;
                }
            }
        }
    }

    /// Forward QJL projection: residual -> projected (d outputs)
    void project_forward(const float* residual, float* out) const {
        if (qjl_type == 0) {
            std::vector<float> fwht_buf(padded_d);
            for (size_t j = 0; j < d; j++) {
                fwht_buf[j] = residual[j] * fwht_signs[j];
            }
            for (size_t j = d; j < padded_d; j++) {
                fwht_buf[j] = 0.0f;
            }
            fwht_inplace(fwht_buf.data(), padded_d);
            for (size_t j = 0; j < d; j++) {
                out[j] = fwht_buf[j];
            }
        } else {
            rr_forward(residual, out);
        }
    }

    /// Inverse QJL projection: signs_buf -> reconstructed (d outputs)
    void project_inverse(float* signs_buf, float* out) const {
        if (qjl_type == 0) {
            fwht_inplace(signs_buf, padded_d);
            for (size_t j = 0; j < d; j++) {
                out[j] = signs_buf[j] * fwht_signs[j];
            }
        } else {
            rr_inverse(signs_buf, out);
        }
    }

    void rr_forward(const float* x, float* out) const {
        float alpha = 1.0f;
        float beta = 0.0f;
        int di = static_cast<int>(d);
        int one = 1;
        sgemm_("T",
               "N",
               &di,
               &one,
               &di,
               &alpha,
               rr_matrix.data(),
               &di,
               x,
               &di,
               &beta,
               out,
               &di);
    }

    void rr_inverse(const float* x, float* out) const {
        float alpha = 1.0f;
        float beta = 0.0f;
        int di = static_cast<int>(d);
        int one = 1;
        sgemm_("N",
               "N",
               &di,
               &one,
               &di,
               &alpha,
               rr_matrix.data(),
               &di,
               x,
               &di,
               &beta,
               out,
               &di);
    }

    /// Store MSE index for dimension j using BIT-PLANE layout.
    /// Plane p stores bit p of every dimension's index.
    void store_mse_index(uint8_t idx, uint8_t* code, size_t j) const {
        for (int p = 0; p < kMSEBits; p++) {
            if (idx & (1 << p)) {
                code[p * mse_plane_bytes + j / 8] |= (1 << (j % 8));
            }
        }
    }

    /// Load MSE index for dimension j from BIT-PLANE layout.
    uint8_t load_mse_index(const uint8_t* code, size_t j) const {
        uint8_t idx = 0;
        for (int p = 0; p < kMSEBits; p++) {
            if (code[p * mse_plane_bytes + j / 8] & (1 << (j % 8))) {
                idx |= (1 << p);
            }
        }
        return idx;
    }

    void encode_vector(const float* x, uint8_t* code) const final {
        float sqrt_d = std::sqrt(static_cast<float>(d));
        float inv_sqrt_d = 1.0f / sqrt_d;

        float x_norm = std::sqrt(fvec_norm_L2sqr(x, d));
        if (x_norm < 1e-30f) {
            x_norm = 1e-30f;
        }

        // MSE quantize in scaled space + compute residual
        std::vector<float> residual(padded_d);
        for (size_t j = 0; j < d; j++) {
            float v = x[j] / x_norm; // unit-normalized
            float val = v * sqrt_d;  // scaled for MSE lookup
            uint8_t idx = static_cast<uint8_t>(
                    std::upper_bound(
                            boundaries,
                            boundaries + (kCentroidsCount - 1),
                            val) -
                    boundaries);
            store_mse_index(idx, code, j);
            residual[j] = v - centroids[idx] * inv_sqrt_d;
        }

        // QJL: project residual, take signs
        std::vector<float> proj(d);
        project_forward(residual.data(), proj.data());

        uint8_t* qjl_code = code + mse_total_bytes;
        for (size_t j = 0; j < d; j++) {
            if (proj[j] > 0.0f) {
                rabitq_utils::set_bit_standard(qjl_code, j);
            }
        }

        // Store per-vector factors
        float gamma = std::sqrt(fvec_norm_L2sqr(residual.data(), d));
        auto* factors = reinterpret_cast<SQTurboQFactors*>(
                code + mse_total_bytes + qjl_plane_bytes);
        factors->norm = x_norm;
        factors->gamma = gamma;
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        float inv_sqrt_pd = 1.0f / std::sqrt(static_cast<float>(padded_d));

        const auto* factors = reinterpret_cast<const SQTurboQFactors*>(
                code + mse_total_bytes + qjl_plane_bytes);

        // MSE reconstruction
        for (size_t j = 0; j < d; j++) {
            uint8_t idx = load_mse_index(code, j);
            x[j] = centroids[idx] * inv_sqrt_d;
        }

        // QJL reconstruction: coeff * gamma * S^T * signs
        const uint8_t* qjl_code = code + mse_total_bytes;
        float coeff =
                std::sqrt(M_PI / 2.0f) / static_cast<float>(d) * factors->gamma;

        std::vector<float> signs_buf(padded_d);
        for (size_t j = 0; j < d; j++) {
            signs_buf[j] = rabitq_utils::extract_bit_standard(qjl_code, j)
                    ? inv_sqrt_pd
                    : -inv_sqrt_pd;
        }
        for (size_t j = d; j < padded_d; j++) {
            signs_buf[j] = 0.0f;
        }

        std::vector<float> reconstructed(d);
        project_inverse(signs_buf.data(), reconstructed.data());
        for (size_t j = 0; j < d; j++) {
            x[j] += coeff * reconstructed[j];
        }

        // Scale by norm
        for (size_t j = 0; j < d; j++) {
            x[j] *= factors->norm;
        }
    }
};

template <int NBits, SIMDLevel SL>
struct QuantizerTurboQuantFull
        : QuantizerTurboQuantFull<NBits, SIMDLevel::NONE> {
    using QuantizerTurboQuantFull<NBits, SIMDLevel::NONE>::
            QuantizerTurboQuantFull;
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
