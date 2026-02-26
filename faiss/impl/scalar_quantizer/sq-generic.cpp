/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/scalar_quantizer/sq-inl.h>

namespace faiss {

namespace scalar_quantizer {

// NONE specializations — always available, no dimension check.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<SIMDLevel::NONE>(
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return select_quantizer_1_body<SIMDLevel::NONE>(qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQDistanceComputer* sq_select_distance_computer<
        SIMDLevel::NONE>(
        MetricType mt,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    if (mt == METRIC_L2) {
        return select_distance_computer_body<SimilarityL2<SIMDLevel::NONE>>(
                qtype, d, trained);
    } else {
        return select_distance_computer_body<SimilarityIP<SIMDLevel::NONE>>(
                qtype, d, trained);
    }
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
InvertedListScanner* sq_select_InvertedListScanner<SIMDLevel::NONE>(
        ScalarQuantizer::QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    return select_InvertedListScanner_body<SIMDLevel::NONE>(
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
