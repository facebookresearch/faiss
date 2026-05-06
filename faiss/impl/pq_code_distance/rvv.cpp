/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_RISCV_RVV

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

namespace faiss {
namespace pq_code_distance {

// RISCV_RVV: no RVV-optimized PQ code distance exists yet. Use scalar.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
float pq_code_distance_8bit_single_impl<SIMDLevel::RISCV_RVV>(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    return PQCodeDistanceScalar<PQDecoder8>::distance_single_code(
            M, 8, sim_table, code);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void pq_code_distance_8bit_four_impl<SIMDLevel::RISCV_RVV>(
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

} // namespace pq_code_distance
} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::RISCV_RVV

// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/hamming_distance/hamming_computer-rvv.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/PQDistanceComputer_impl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/IVFPQScanner_impl.h>

#undef THE_SIMD_LEVEL

#endif // COMPILE_SIMD_RISCV_RVV
