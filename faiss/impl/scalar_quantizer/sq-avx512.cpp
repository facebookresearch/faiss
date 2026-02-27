/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/scalar_quantizer/sq-inl.h>

namespace faiss {

namespace scalar_quantizer {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<SIMDLevel::AVX512>(
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    if (d % 16 != 0) {
        return nullptr;
    }
    return select_quantizer_1_body<SIMDLevel::AVX512>(qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQDistanceComputer* sq_select_distance_computer<
        SIMDLevel::AVX512>(
        MetricType mt,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    if (d % 16 != 0) {
        return nullptr;
    }
    if (mt == METRIC_L2) {
        return select_distance_computer_body<SimilarityL2<SIMDLevel::AVX512>>(
                qtype, d, trained);
    } else {
        return select_distance_computer_body<SimilarityIP<SIMDLevel::AVX512>>(
                qtype, d, trained);
    }
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
InvertedListScanner* sq_select_InvertedListScanner<SIMDLevel::AVX512>(
        ScalarQuantizer::QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (d % 16 != 0) {
        return nullptr;
    }
    return select_InvertedListScanner_body<SIMDLevel::AVX512>(
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

#endif // COMPILE_SIMD_AVX512

#ifdef COMPILE_SIMD_AVX512_SPR
// AVX512_SPR: Sapphire Rapids is a superset of AVX512. Reuse the
// AVX512 implementation until a dedicated SPR specialization is written.

#include <faiss/impl/scalar_quantizer/sq-inl.h>

namespace faiss {

namespace scalar_quantizer {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<SIMDLevel::AVX512_SPR>(
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return sq_select_quantizer<SIMDLevel::AVX512>(qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQDistanceComputer* sq_select_distance_computer<
        SIMDLevel::AVX512_SPR>(
        MetricType mt,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return sq_select_distance_computer<SIMDLevel::AVX512>(
            mt, qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
InvertedListScanner* sq_select_InvertedListScanner<SIMDLevel::AVX512_SPR>(
        ScalarQuantizer::QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    return sq_select_InvertedListScanner<SIMDLevel::AVX512>(
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

#endif // COMPILE_SIMD_AVX512_SPR
