/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This TU provides non-templated PQ code distance dispatch wrappers
// (pq_code_distance_8bit_single, pq_code_distance_8bit_four) declared
// in pq_code_distance-inl.h. These use with_simd_level to route to the
// best available SIMD implementation via pq_code_distance_8bit_*_impl
// function template specializations.
//
// The NONE and ARM_NEON _impl specializations are defined inline in
// pq_code_distance-generic.h (included transitively). The AVX2, AVX512,
// and ARM_SVE specializations are in their respective per-SIMD files.

#include <faiss/impl/pq_code_distance/pq_code_distance-generic.h>

#define THE_SIMD_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/pq_scan_impl.h>
#undef THE_SIMD_LEVEL

namespace faiss {
namespace pq_code_distance {

void pq_scan_8bit(
        size_t M,
        const float* dis_table,
        const uint8_t* codes,
        size_t ncodes,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        bool max_heap) {
    with_simd_level([&]<SIMDLevel SL>() {
        pq_scan_8bit_impl<SL>(
                M, dis_table, codes, ncodes, k, heap_dis, heap_ids, max_heap);
    });
}

float pq_code_distance_8bit_single(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    return with_simd_level([&]<SIMDLevel SL>() {
        return pq_code_distance_8bit_single_impl<SL>(M, sim_table, code);
    });
}

void pq_code_distance_8bit_four(
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
    with_simd_level([&]<SIMDLevel SL>() {
        pq_code_distance_8bit_four_impl<SL>(
                M,
                sim_table,
                code0,
                code1,
                code2,
                code3,
                result0,
                result1,
                result2,
                result3);
    });
}

} // namespace pq_code_distance
} // namespace faiss
