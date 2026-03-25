/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This TU provides:
// 1. _impl specializations for NONE (and ARM_NEON), using scalar code.
// 2. Non-templated PQ code distance dispatch wrappers
//    (pq_code_distance_single, pq_code_distance_four) declared in
//    pq_code_distance.h. These use DISPATCH_SIMDLevel to route to the
//    best available SIMD implementation via pq_code_distance_*_impl
//    function template specializations defined in the per-SIMD .cpp files.

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

namespace faiss {
namespace pq_code_distance {

// NONE: use scalar directly.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
float pq_code_distance_single_impl<SIMDLevel::NONE>(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code) {
    return PQCodeDistanceScalar<PQDecoder8>::distance_single_code(
            M, nbits, sim_table, code);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void pq_code_distance_four_impl<SIMDLevel::NONE>(
        size_t M,
        size_t nbits,
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
            nbits,
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

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void pq_code_distance_batch_impl<SIMDLevel::NONE>(
        size_t M,
        size_t nbits,
        size_t ncode,
        const uint8_t* codes,
        const float* sim_table,
        float* dis,
        float dis0) {
    for (size_t i = 0; i < ncode; i++) {
        dis[i] = pq_code_distance_single_impl<SIMDLevel::NONE>(
                M, nbits, sim_table, codes + i * M) + dis0;
    }
}

// ARM_NEON NEON-optimized implementations are in pq_code_distance-neon.cpp.
// [DD-MIGRATION] Original code (scalar fallback for ARM_NEON, now superseded):
// #ifdef COMPILE_SIMD_ARM_NEON
// template <>
// float pq_code_distance_single_impl<SIMDLevel::ARM_NEON>(
//         size_t M, size_t nbits, const float* sim_table, const uint8_t* code) {
//     return PQCodeDistanceScalar<PQDecoder8>::distance_single_code(M, nbits, sim_table, code);
// }
// template <>
// void pq_code_distance_four_impl<SIMDLevel::ARM_NEON>(
//         size_t M, size_t nbits, const float* sim_table,
//         const uint8_t* __restrict code0, const uint8_t* __restrict code1,
//         const uint8_t* __restrict code2, const uint8_t* __restrict code3,
//         float& result0, float& result1, float& result2, float& result3) {
//     PQCodeDistanceScalar<PQDecoder8>::distance_four_codes(
//             M, nbits, sim_table, code0, code1, code2, code3,
//             result0, result1, result2, result3);
// }
// #endif // COMPILE_SIMD_ARM_NEON

float pq_code_distance_single(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code) {
    DISPATCH_SIMDLevel(pq_code_distance_single_impl, M, nbits, sim_table, code);
}

void pq_code_distance_four(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    DISPATCH_SIMDLevel(
            pq_code_distance_four_impl,
            M,
            nbits,
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

void pq_code_distance_batch(
        size_t M,
        size_t nbits,
        size_t ncode,
        const uint8_t* codes,
        const float* sim_table,
        float* dis,
        float dis0) {
    DISPATCH_SIMDLevel(
            pq_code_distance_batch_impl,
            M,
            nbits,
            ncode,
            codes,
            sim_table,
            dis,
            dis0);
}

} // namespace pq_code_distance
} // namespace faiss
