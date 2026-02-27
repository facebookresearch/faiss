/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

namespace scalar_quantizer {

/// Specialized in per-SIMD .cpp files. Returns nullptr if d doesn't align
/// for the given SIMD level.
template <SIMDLevel SL>
ScalarQuantizer::SQuantizer* sq_select_quantizer(
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

template <SIMDLevel SL>
ScalarQuantizer::SQDistanceComputer* sq_select_distance_computer(
        MetricType mt,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

template <SIMDLevel SL>
InvertedListScanner* sq_select_InvertedListScanner(
        ScalarQuantizer::QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual);

} // namespace scalar_quantizer

} // namespace faiss
