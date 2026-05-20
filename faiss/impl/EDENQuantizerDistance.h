/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/EDENQuantizer.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/scalar_quantizer/training.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/simd_levels.h>

#include <cstring>
#include <memory>
#include <vector>

namespace faiss {

namespace eden_distance {

inline const std::vector<float>& codebook(size_t nb_bits) {
    return scalar_quantizer::lloyd_max_normal_centroids(nb_bits);
}

inline float compute_code_dot_reference(
        const uint8_t* code,
        const float* query,
        size_t d,
        size_t nb_bits) {
    const float* centroids = codebook(nb_bits).data();
    BitstringReader reader(code, eden_utils::packed_code_size(d, nb_bits));
    float dot = 0.0f;
    for (size_t i = 0; i < d; i++) {
        dot += query[i] * centroids[reader.read(nb_bits)];
    }
    return dot;
}

struct EDENDistanceComputerBase : EDENFlatCodesDistanceComputer {
    size_t d = 0;
    size_t nb_bits = 1;
    const float* centroid = nullptr;
    MetricType metric_type = MetricType::METRIC_L2;

    size_t packed_size = 0;
    std::vector<float> dot_query;
    float query_base = 0.0f;

    float symmetric_dis(idx_t /*i*/, idx_t /*j*/) override {
        FAISS_THROW_MSG("Not implemented");
    }

    void set_query_common(const float* x) {
        q = x;
        FAISS_ASSERT(x != nullptr);

        dot_query.resize(d);
        if (metric_type == MetricType::METRIC_L2) {
            query_base = centroid ? fvec_L2sqr(x, centroid, d)
                                  : fvec_norm_L2sqr(x, d);
            for (size_t i = 0; i < d; i++) {
                dot_query[i] = x[i] - (centroid ? centroid[i] : 0.0f);
            }
        } else if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
            query_base = centroid ? fvec_inner_product(x, centroid, d) : 0.0f;
            memcpy(dot_query.data(), x, d * sizeof(float));
        } else {
            FAISS_THROW_MSG("EDEN supports only L2 and inner-product metrics");
        }
    }

    float distance_from_code_dot(
            const EDENCodeFactors* factors,
            float code_dot_query) const {
        if (metric_type == MetricType::METRIC_L2) {
            return query_base + factors->l2_norm_term -
                    2.0f * factors->scale * code_dot_query;
        }
        return query_base + factors->scale * code_dot_query;
    }
};

struct EDENReferenceDistanceComputer : EDENDistanceComputerBase {
    void set_query(const float* x) override {
        set_query_common(x);
    }

    float distance_to_code(const uint8_t* code) final {
        const EDENCodeFactors* factors =
                reinterpret_cast<const EDENCodeFactors*>(code + packed_size);
        const float code_dot_query = compute_code_dot_reference(
                code, dot_query.data(), d, nb_bits);
        return distance_from_code_dot(factors, code_dot_query);
    }
};

inline EDENFlatCodesDistanceComputer* make_reference_distance_computer(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* centroid) {
    auto dc = std::make_unique<EDENReferenceDistanceComputer>();
    dc->metric_type = metric_type;
    dc->d = d;
    dc->nb_bits = nb_bits;
    dc->centroid = centroid;
    dc->packed_size = eden_utils::packed_code_size(d, nb_bits);
    return dc.release();
}

template <SIMDLevel SL>
EDENFlatCodesDistanceComputer* make_distance_computer(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* centroid);

} // namespace eden_distance

} // namespace faiss
