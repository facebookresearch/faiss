/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/EDENQuantizer.h>
#include <faiss/impl/EDENQuantizerDistance.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/scalar_quantizer/training.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace faiss {

namespace {

const std::vector<float>& half_boundaries(size_t nb_bits) {
    return scalar_quantizer::lloyd_max_normal_half_boundaries(nb_bits);
}

const std::vector<float>& full_centroids(size_t nb_bits) {
    return scalar_quantizer::lloyd_max_normal_centroids(nb_bits);
}

uint8_t assign_lloyd_max(float normalized_value, size_t nb_bits) {
    const size_t half_count = size_t{1} << (nb_bits - 1);
    const float abs_value = std::abs(normalized_value);
    const auto& boundaries = half_boundaries(nb_bits);

    if (normalized_value < 0) {
        const size_t magnitude =
                std::lower_bound(
                        boundaries.begin(), boundaries.end(), abs_value) -
                boundaries.begin();
        return static_cast<uint8_t>(half_count - 1 - magnitude);
    }
    const size_t magnitude =
            std::upper_bound(boundaries.begin(), boundaries.end(), abs_value) -
            boundaries.begin();
    return static_cast<uint8_t>(half_count + magnitude);
}

} // namespace

namespace eden_utils {

size_t packed_code_size(size_t d, size_t nb_bits) {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 1 && nb_bits <= 8, "EDEN nb_bits must be in [1, 8]");
    return (d * nb_bits + 7) / 8;
}

float code_to_centroid(uint8_t code, size_t nb_bits) {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 1 && nb_bits <= 8, "EDEN nb_bits must be in [1, 8]");
    const size_t code_count = size_t{1} << nb_bits;
    FAISS_ASSERT(code < code_count);

    return full_centroids(nb_bits)[code];
}

uint8_t extract_code(const uint8_t* codes, size_t index, size_t nb_bits) {
    if (nb_bits == 8) {
        return codes[index];
    }

    const size_t bit_pos = index * nb_bits;
    BitstringReader reader(codes, (bit_pos + nb_bits + 7) / 8);
    reader.i = bit_pos;
    return static_cast<uint8_t>(reader.read(nb_bits));
}

} // namespace eden_utils

EDENQuantizer::EDENQuantizer(
        size_t d_in,
        MetricType metric,
        size_t nb_bits_in,
        EDENScaleType scale_type_in)
        : Quantizer(d_in, 0),
          metric_type(metric),
          scale_type(scale_type_in),
          nb_bits(nb_bits_in) {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 1 && nb_bits <= 8, "EDEN nb_bits must be in [1, 8]");
    FAISS_THROW_IF_NOT_MSG(
            scale_type == EDENScaleType_UNBIASED ||
                    scale_type == EDENScaleType_BIASED,
            "invalid EDEN scale type");
    code_size = compute_code_size(d, nb_bits);
}

size_t EDENQuantizer::compute_code_size(size_t d_in, size_t num_bits) const {
    return eden_utils::packed_code_size(d_in, num_bits) +
            sizeof(EDENCodeFactors);
}

void EDENQuantizer::train(size_t /*n*/, const float* /*x*/) {}

void EDENQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    compute_codes_core(x, codes, n, centroid);
}

void EDENQuantizer::compute_codes_core(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid_in) const {
    FAISS_ASSERT(x != nullptr);
    FAISS_ASSERT(codes != nullptr);
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2 ||
                    metric_type == MetricType::METRIC_INNER_PRODUCT,
            "EDEN supports only L2 and inner-product metrics");

    if (n == 0) {
        return;
    }

    const size_t packed_size = eden_utils::packed_code_size(d, nb_bits);
    const float sqrt_d = std::sqrt(static_cast<float>(d));
    const float* codebook = full_centroids(nb_bits).data();

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const float* xi = x + i * d;
        uint8_t* code = codes + i * code_size;
        memset(code, 0, code_size);

        float norm_sqr = 0.0f;
        for (size_t j = 0; j < d; j++) {
            const float c = centroid_in ? centroid_in[j] : 0.0f;
            const float r = xi[j] - c;
            norm_sqr += r * r;
        }

        EDENCodeFactors* factors =
                reinterpret_cast<EDENCodeFactors*>(code + packed_size);
        if (norm_sqr <= std::numeric_limits<float>::epsilon()) {
            factors->l2_norm_term = 0.0f;
            factors->scale = 0.0f;
            continue;
        }

        const float norm = std::sqrt(norm_sqr);
        const float inv_norm = 1.0f / norm;
        double code_norm_sqr = 0.0;
        double code_residual_ip = 0.0;
        BitstringWriter writer(code, nb_bits == 8 ? 0 : packed_size);

        for (size_t j = 0; j < d; j++) {
            const float c = centroid_in ? centroid_in[j] : 0.0f;
            const float r = xi[j] - c;
            const float z = r * sqrt_d * inv_norm;
            const uint8_t assignment = assign_lloyd_max(z, nb_bits);
            const float q = codebook[assignment];

            if (nb_bits == 8) {
                code[j] = assignment;
            } else {
                writer.write(assignment, nb_bits);
            }
            code_norm_sqr += double(q) * q;
            code_residual_ip += double(q) * r;
        }

        float scale = 0.0f;
        float l2_norm_term = 0.0f;
        // Unbiased EDEN uses ||r||^2 / <q, r>. The biased scale follows
        // DRIVE (NeurIPS 2021): <q, r> / ||q||^2.
        if (scale_type == EDENScaleType_BIASED) {
            scale = static_cast<float>(code_residual_ip / code_norm_sqr);
            l2_norm_term =
                    static_cast<float>(double(scale) * scale * code_norm_sqr);
        } else {
            scale = static_cast<float>(double(norm_sqr) / code_residual_ip);
            l2_norm_term = norm_sqr;
        }
        if (!std::isfinite(scale)) {
            scale = 0.0f;
            l2_norm_term = 0.0f;
        }

        factors->scale = scale;
        factors->l2_norm_term = l2_norm_term;
    }
}

void EDENQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    decode_core(codes, x, n, centroid);
}

void EDENQuantizer::decode_core(
        const uint8_t* codes,
        float* x,
        size_t n,
        const float* centroid_in) const {
    FAISS_ASSERT(codes != nullptr);
    FAISS_ASSERT(x != nullptr);

    const size_t packed_size = eden_utils::packed_code_size(d, nb_bits);
    const float* codebook = full_centroids(nb_bits).data();

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const uint8_t* code = codes + i * code_size;
        const EDENCodeFactors* factors =
                reinterpret_cast<const EDENCodeFactors*>(code + packed_size);
        float* xi = x + i * d;

        if (nb_bits == 8) {
            for (size_t j = 0; j < d; j++) {
                const float q = codebook[code[j]];
                const float c = centroid_in ? centroid_in[j] : 0.0f;
                xi[j] = c + factors->scale * q;
            }
        } else {
            BitstringReader reader(code, packed_size);
            for (size_t j = 0; j < d; j++) {
                const float q = codebook[reader.read(nb_bits)];
                const float c = centroid_in ? centroid_in[j] : 0.0f;
                xi[j] = c + factors->scale * q;
            }
        }
    }
}

namespace eden_distance {

template <>
EDENFlatCodesDistanceComputer* make_distance_computer<SIMDLevel::NONE>(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* centroid) {
    return make_reference_distance_computer(metric_type, d, nb_bits, centroid);
}

} // namespace eden_distance

EDENFlatCodesDistanceComputer* EDENQuantizer::get_distance_computer(
        const float* centroid_in) const {
    return with_simd_level(
            [&]<SIMDLevel SL>() -> EDENFlatCodesDistanceComputer* {
                return eden_distance::make_distance_computer<SL>(
                        metric_type, d, nb_bits, centroid_in);
            });
}

} // namespace faiss
