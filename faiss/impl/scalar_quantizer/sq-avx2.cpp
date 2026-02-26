/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <faiss/impl/scalar_quantizer/sq-inl.h>

namespace faiss {

namespace scalar_quantizer {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<SIMDLevel::AVX2>(
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    if (d % 8 != 0) {
        return nullptr;
    }
    return select_quantizer_1_body<SIMDLevel::AVX2>(qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQDistanceComputer* sq_select_distance_computer<
        SIMDLevel::AVX2>(
        MetricType mt,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    if (d % 8 != 0) {
        return nullptr;
    }
    if (mt == METRIC_L2) {
        return select_distance_computer_body<SimilarityL2<SIMDLevel::AVX2>>(
                qtype, d, trained);
    } else {
        return select_distance_computer_body<SimilarityIP<SIMDLevel::AVX2>>(
                qtype, d, trained);
    }
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
InvertedListScanner* sq_select_InvertedListScanner<SIMDLevel::AVX2>(
        ScalarQuantizer::QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (d % 8 != 0) {
        return nullptr;
    }
    return select_InvertedListScanner_body<SIMDLevel::AVX2>(
            qtype,
            mt,
            d,
            code_size,
            trained,
            quantizer,
            store_pairs,
            sel,
            by_residual);
}

} // namespace scalar_quantizer

} // namespace faiss

#endif // COMPILE_SIMD_AVX2
