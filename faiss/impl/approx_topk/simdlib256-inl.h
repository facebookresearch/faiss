/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Out-of-line definition of HeapWithBucketsCMaxFloat::bs_addn using
// simdlib types. Only included by per-ISA .cpp files (avx2.cpp, neon.cpp).
// Do NOT include this from common translation units.

#pragma once

#include <cstdint>
#include <limits>

#include <faiss/impl/approx_topk/approx_topk.h>
#include <faiss/impl/simdlib/simdlib.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

// Element-wise max of two simd8float32 vectors, implemented via
// cmplt_min_max_fast (which computes both min and max).
template <SIMDLevel SL>
inline simd8float32_tpl<SL> simd8float32_max(
        simd8float32_tpl<SL> a,
        simd8float32_tpl<SL> b) {
    simd8float32_tpl<SL> min_val, max_val;
    simd8uint32_tpl<SL> dummy(0u), dmin, dmax;
    cmplt_min_max_fast(a, dummy, b, dummy, min_val, dmin, max_val, dmax);
    return max_val;
}

template <uint32_t NBUCKETS, uint32_t N, SIMDLevel SL>
void HeapWithBucketsCMaxFloat<NBUCKETS, N, SL>::bs_addn(
        const uint32_t beam_size,
        const uint32_t n_per_beam,
        const float* const __restrict distances,
        const uint32_t k,
        float* const __restrict bh_val,
        int32_t* const __restrict bh_ids) {
    using C = CMax<float, int>;
    using simd_float = simd8float32_tpl<SL>;
    using simd_uint = simd8uint32_tpl<SL>;

    for (uint32_t beam_index = 0; beam_index < beam_size; beam_index++) {
        simd_float min_distances_i[NBUCKETS / 8][N];
        simd_uint min_indices_i[NBUCKETS / 8][N];

        for (uint32_t j = 0; j < NBUCKETS / 8; j++) {
            for (uint32_t p = 0; p < N; p++) {
                min_distances_i[j][p] =
                        simd_float(std::numeric_limits<float>::max());
                min_indices_i[j][p] = simd_uint(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u);
            }
        }

        simd_uint current_indices(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u);
        simd_uint indices_delta(NBUCKETS);

        const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;

        // put the data into buckets
        for (uint32_t ip = 0; ip < nb; ip += NBUCKETS) {
            for (uint32_t j = 0; j < NBUCKETS / 8; j++) {
                const simd_float distances_reg(
                        distances + j * 8 + ip + n_per_beam * beam_index);

                simd_float distance_candidate = distances_reg;
                simd_uint indices_candidate = current_indices;

                for (uint32_t p = 0; p < N; p++) {
                    // Use cmplt_min_max_fast for comparison, min values,
                    // min indices, and max indices.
                    simd_float min_d_new, max_d_unused;
                    simd_uint min_idx_new, max_idx_new;
                    cmplt_min_max_fast(
                            distance_candidate,
                            indices_candidate,
                            min_distances_i[j][p],
                            min_indices_i[j][p],
                            min_d_new,
                            min_idx_new,
                            max_d_unused,
                            max_idx_new);

                    // The max distance uses distances_reg (the original
                    // input), NOT distance_candidate. This is a deliberate
                    // approximation that breaks the data dependency chain.
                    simd_float max_d_new = simd8float32_max<SL>(
                            min_distances_i[j][p], distances_reg);

                    distance_candidate = max_d_new;
                    indices_candidate = max_idx_new;

                    min_distances_i[j][p] = min_d_new;
                    min_indices_i[j][p] = min_idx_new;
                }
            }

            current_indices = current_indices + indices_delta;
        }

        // fix the indices
        for (uint32_t j = 0; j < NBUCKETS / 8; j++) {
            const simd_uint offset(n_per_beam * beam_index + j * 8);
            for (uint32_t p = 0; p < N; p++) {
                min_indices_i[j][p] = min_indices_i[j][p] + offset;
            }
        }

        // merge every bucket into the regular heap
        for (uint32_t p = 0; p < N; p++) {
            for (uint32_t j = 0; j < NBUCKETS / 8; j++) {
                uint32_t min_indices_scalar[8];
                float min_distances_scalar[8];

                min_indices_i[j][p].storeu(min_indices_scalar);
                min_distances_i[j][p].storeu(min_distances_scalar);

                for (size_t j8 = 0; j8 < 8; j8++) {
                    const auto value = min_distances_scalar[j8];
                    const auto index =
                            static_cast<int32_t>(min_indices_scalar[j8]);
                    if (C::cmp2(bh_val[0], value, bh_ids[0], index)) {
                        heap_replace_top<C>(k, bh_val, bh_ids, value, index);
                    }
                }
            }
        }

        // process leftovers
        for (uint32_t ip = nb; ip < n_per_beam; ip++) {
            const int32_t index = ip + n_per_beam * beam_index;
            const float value = distances[index];

            if (C::cmp(bh_val[0], value)) {
                heap_replace_top<C>(k, bh_val, bh_ids, value, index);
            }
        }
    }
}

} // namespace faiss
