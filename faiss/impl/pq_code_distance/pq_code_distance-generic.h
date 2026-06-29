/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

namespace faiss {
namespace pq_code_distance {

// NONE: use scalar directly.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
inline float pq_code_distance_8bit_single_impl<SIMDLevel::NONE>(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    return PQCodeDistanceScalar<PQDecoder8>::distance_single_code(
            M, 8, sim_table, code);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
inline void pq_code_distance_8bit_four_impl<SIMDLevel::NONE>(
        size_t M,
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    PQCodeDistanceScalar<PQDecoder8>::distance_four_codes(
            M,
            8,
            sim_table,
            code0,
            code1,
            code2,
            code3,
            result0,
            result1,
            result2,
            result3);
}

#ifdef COMPILE_SIMD_ARM_NEON
// ARM_NEON: No NEON-optimized PQ code distance exists. Use scalar.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
inline float pq_code_distance_8bit_single_impl<SIMDLevel::ARM_NEON>(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    return PQCodeDistanceScalar<PQDecoder8>::distance_single_code(
            M, 8, sim_table, code);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
inline void pq_code_distance_8bit_four_impl<SIMDLevel::ARM_NEON>(
        size_t M,
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    PQCodeDistanceScalar<PQDecoder8>::distance_four_codes(
            M,
            8,
            sim_table,
            code0,
            code1,
            code2,
            code3,
            result0,
            result1,
            result2,
            result3);
}
#endif // COMPILE_SIMD_ARM_NEON

} // namespace pq_code_distance
} // namespace faiss
