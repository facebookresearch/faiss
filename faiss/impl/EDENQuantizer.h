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
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

enum EDENScaleType {
    // Original EDEN unbiased scale.
    EDENScaleType_UNBIASED = 1,
    // Biased DRIVE scale (NeurIPS 2021, arXiv:2105.08339), generalized to
    // multi-bit EDEN; see also https://arxiv.org/abs/2604.18555.
    EDENScaleType_BIASED = 2,
};

FAISS_PACK_STRUCTS_BEGIN
struct FAISS_PACKED EDENCodeFactors {
    // L2 term used by the distance computer. For the unbiased scale this is
    // ||r||^2. For the biased scale this is S^2 * ||q_code||^2, the
    // reconstructed-code norm.
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

    virtual void consecutive_distances_batch_16(idx_t first, float* distances) {
        consecutive_distances_batch_8(first, distances);
        consecutive_distances_batch_8(first + 8, distances + 8);
    }
};

// EDEN Lloyd-Max quantizer from the EDEN ICML 2022 paper:
// https://proceedings.mlr.press/v162/vargaftik22a.html.
//
// EDEN operates on the vectors supplied to it. Optional preprocessing such as
// a random rotation can be applied externally with IndexPreTransform. The
// scalar assignment itself is a ScalarQuantizer::QT_*_eden qtype. EDEN adds
// per-vector scale factors after the packed scalar code and computes them
// against an externally supplied centroid; nullptr means a zero centroid. The
// default stored scale is EDEN's original unbiased scale. The biased scale
// follows DRIVE's MSE-minimizing scale for the chosen Lloyd-Max codeword; see
// also
// https://arxiv.org/abs/2604.18555.

namespace eden_utils {

ScalarQuantizer::QuantizerType quantizer_type_for_bits(size_t nb_bits);

bool is_eden_quantizer_type(ScalarQuantizer::QuantizerType qtype);

size_t nb_bits_for_qtype(ScalarQuantizer::QuantizerType qtype);

size_t packed_code_size(size_t d, size_t nb_bits);

size_t code_size(size_t d, size_t nb_bits);

uint8_t extract_code(const uint8_t* codes, size_t index, size_t nb_bits);

void compute_codes(
        const ScalarQuantizer& sq,
        MetricType metric_type,
        EDENScaleType scale_type,
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid = nullptr);

void decode(
        const ScalarQuantizer& sq,
        const uint8_t* codes,
        float* x,
        size_t n,
        const float* centroid = nullptr);

EDENFlatCodesDistanceComputer* get_distance_computer(
        const ScalarQuantizer& sq,
        MetricType metric_type,
        const float* centroid = nullptr);

} // namespace eden_utils

} // namespace faiss
