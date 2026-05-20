/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/MetricType.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/Quantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

enum EDENScaleType {
    // Original EDEN unbiased scale.
    EDENScaleType_UNBIASED = 1,
    // Biased DRIVE scale (NeurIPS 2021, arXiv:2105.08339), generalized to
    // multi-bit EDEN.
    EDENScaleType_BIASED = 2,
};

FAISS_PACK_STRUCTS_BEGIN
struct FAISS_PACKED EDENCodeFactors {
    // L2 term used by the distance computer. For the unbiased scale this is
    // ||r||^2, yielding RaBitQ's 1-bit distance estimator. For the biased scale
    // this is S^2 * ||q_code||^2, the reconstructed-code norm.
    float l2_norm_term = 0;
    // Per-vector reconstruction scale.
    float scale = 0;
};
FAISS_PACK_STRUCTS_END

struct EDENFlatCodesDistanceComputer : FlatCodesDistanceComputer {
    using FlatCodesDistanceComputer::FlatCodesDistanceComputer;

    virtual void consecutive_distances_batch_8(idx_t first, float* distances) {
        distances_batch_4(
                first,
                first + 1,
                first + 2,
                first + 3,
                distances[0],
                distances[1],
                distances[2],
                distances[3]);
        distances_batch_4(
                first + 4,
                first + 5,
                first + 6,
                first + 7,
                distances[4],
                distances[5],
                distances[6],
                distances[7]);
    }

};

// EDEN Lloyd-Max quantizer from the EDEN ICML 2022 paper:
// https://proceedings.mlr.press/v162/vargaftik22a.html.
//
// Like RaBitQuantizer, this assumes any random rotation is provided externally
// (for example by IndexPreTransform). Codes are produced against an externally
// supplied centroid; nullptr means a zero centroid. The default stored scale is
// EDEN's original unbiased scale. The biased scale
// follows DRIVE's MSE-minimizing scale for the chosen Lloyd-Max codeword; see
// also https://arxiv.org/abs/2604.18555.
struct EDENQuantizer : Quantizer {
    float* centroid = nullptr;
    MetricType metric_type = MetricType::METRIC_L2;
    EDENScaleType scale_type = EDENScaleType_UNBIASED;

    // Integer Lloyd-Max bit budget. EDEN's published tables cover 1..8 bits.
    size_t nb_bits = 1;

    EDENQuantizer(
            size_t d = 0,
            MetricType metric = MetricType::METRIC_L2,
            size_t nb_bits = 1,
            EDENScaleType scale_type = EDENScaleType_UNBIASED);

    size_t compute_code_size(size_t d, size_t nb_bits) const;

    void train(size_t n, const float* x) override;

    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

    void compute_codes_core(
            const float* x,
            uint8_t* codes,
            size_t n,
            const float* centroid_in) const;

    void decode(const uint8_t* codes, float* x, size_t n) const override;

    void decode_core(
            const uint8_t* codes,
            float* x,
            size_t n,
            const float* centroid_in) const;

    EDENFlatCodesDistanceComputer* get_distance_computer(
            const float* centroid = nullptr) const;
};

namespace eden_utils {

size_t packed_code_size(size_t d, size_t nb_bits);

float code_to_centroid(uint8_t code, size_t nb_bits);

uint8_t extract_code(const uint8_t* codes, size_t index, size_t nb_bits);

} // namespace eden_utils

} // namespace faiss
