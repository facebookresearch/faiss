/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_RISCV_RVV

#include <riscv_vector.h>

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>
#include <faiss/impl/pq_code_distance/pq_rvv_kernel.h>

namespace faiss {
namespace pq_code_distance {

// The kernel lives in pq_rvv_kernel.h. See that header for the gather layout.

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
float pq_code_distance_8bit_single_impl<SIMDLevel::RISCV_RVV>(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    return pq_rvv_kernel::distance_8bit(M, sim_table, code);
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
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    size_t m = 0;
    while (m < M) {
        const size_t vl = __riscv_vsetvl_e32m4(M - m);
        // The row term is the same for all four codes, so compute it once.
        // That is the whole advantage of the four-code entry point.
        vuint32m4_t row_off = pq_rvv_kernel::row_offsets(m, vl);

        vfloat32m4_t v0 = __riscv_vluxei32_v_f32m4(
                sim_table,
                pq_rvv_kernel::byte_offsets(row_off, code0, m, vl),
                vl);
        vfloat32m4_t v1 = __riscv_vluxei32_v_f32m4(
                sim_table,
                pq_rvv_kernel::byte_offsets(row_off, code1, m, vl),
                vl);
        vfloat32m4_t v2 = __riscv_vluxei32_v_f32m4(
                sim_table,
                pq_rvv_kernel::byte_offsets(row_off, code2, m, vl),
                vl);
        vfloat32m4_t v3 = __riscv_vluxei32_v_f32m4(
                sim_table,
                pq_rvv_kernel::byte_offsets(row_off, code3, m, vl),
                vl);

        acc0 = __riscv_vfadd_vv_f32m4_tu(acc0, acc0, v0, vl);
        acc1 = __riscv_vfadd_vv_f32m4_tu(acc1, acc1, v1, vl);
        acc2 = __riscv_vfadd_vv_f32m4_tu(acc2, acc2, v2, vl);
        acc3 = __riscv_vfadd_vv_f32m4_tu(acc3, acc3, v3, vl);

        m += vl;
    }

    result0 = pq_rvv_kernel::reduce(acc0, vlmax);
    result1 = pq_rvv_kernel::reduce(acc1, vlmax);
    result2 = pq_rvv_kernel::reduce(acc2, vlmax);
    result3 = pq_rvv_kernel::reduce(acc3, vlmax);
}

} // namespace pq_code_distance
} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::RISCV_RVV

// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/pq_scan_impl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/hamming_distance/hamming_computer-rvv.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/PQDistanceComputer_impl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/IVFPQScanner_impl.h>

#undef THE_SIMD_LEVEL

#endif // COMPILE_SIMD_RISCV_RVV
