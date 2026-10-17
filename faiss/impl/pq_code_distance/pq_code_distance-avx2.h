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

// 8-bit PQ codes use the scalar kernel at this level too. The gather kernel
// was 1.15-1.8x slower than scalar code on AMD Zen 4 and Zen 5 at M >= 16 and
// 9-37% faster on Intel Skylake, Cooper Lake and Sapphire Rapids
// (perf_tests/bench_pq_distance_to_code). The level template argument keeps
// this TU's scalar instantiation a distinct symbol.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
float pq_code_distance_8bit_single_impl<SIMDLevel::AVX2>(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    return PQCodeDistanceScalar<PQDecoder8, SIMDLevel::AVX2>::
            distance_single_code(M, 8, sim_table, code);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void pq_code_distance_8bit_four_impl<SIMDLevel::AVX2>(
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
    PQCodeDistanceScalar<PQDecoder8, SIMDLevel::AVX2>::distance_four_codes(
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

} // namespace pq_code_distance
} // namespace faiss
