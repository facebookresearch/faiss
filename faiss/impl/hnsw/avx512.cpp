/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/hnsw/MinimaxHeap.h>

#include <immintrin.h>
#include <cassert>
#include <limits>
#include <type_traits>

namespace faiss {

namespace {

/// Templated AVX512 implementation of "pop best" for both CMax (returns
/// the smallest distance) and CMin (returns the largest similarity).
template <class HC>
int pop_best_avx512(MinimaxHeapT<HC>& heap, float* vmin_out) {
    using storage_idx_t = typename MinimaxHeapT<HC>::storage_idx_t;
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");
    assert(heap.k > 0);

    constexpr float worst_v = HC::is_max
            ? std::numeric_limits<float>::infinity()
            : -std::numeric_limits<float>::infinity();

    int32_t best_idx = -1;
    float best_dis = worst_v;

    __m512i best_indices = _mm512_set1_epi32(-1);
    __m512 best_distances = _mm512_set1_ps(worst_v);
    __m512i current_indices = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i offset = _mm512_set1_epi32(16);

    auto best_vs_cand_mask = [](__m512 best_d, __m512 cand_d) -> __mmask16 {
        // Returns the mask of lanes where the current best is already
        // (strictly) better than the candidate.
        if constexpr (HC::is_max) {
            return _mm512_cmp_ps_mask(best_d, cand_d, _CMP_LT_OS);
        } else {
            return _mm512_cmp_ps_mask(best_d, cand_d, _CMP_GT_OS);
        }
    };

    const size_t k16 = (heap.k / 16) * 16;
    for (size_t iii = 0; iii < k16; iii += 16) {
        __m512i indices =
                _mm512_loadu_si512((const __m512i*)(heap.ids.data() + iii));
        __m512 distances = _mm512_loadu_ps(heap.dis.data() + iii);

        __mmask16 m1mask =
                _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);
        __mmask16 dmask = best_vs_cand_mask(best_distances, distances);
        __mmask16 finalmask = m1mask | dmask;

        const __m512i best_indices_new = _mm512_mask_blend_epi32(
                finalmask, current_indices, best_indices);
        const __m512 best_distances_new =
                _mm512_mask_blend_ps(finalmask, distances, best_distances);

        best_indices = best_indices_new;
        best_distances = best_distances_new;

        current_indices = _mm512_add_epi32(current_indices, offset);
    }

    // Leftovers.
    if (k16 != static_cast<size_t>(heap.k)) {
        const __mmask16 kmask = (1 << (heap.k - k16)) - 1;

        __m512i indices = _mm512_mask_loadu_epi32(
                _mm512_set1_epi32(-1), kmask, heap.ids.data() + k16);
        __m512 distances = _mm512_maskz_loadu_ps(kmask, heap.dis.data() + k16);

        __mmask16 m1mask =
                _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);
        __mmask16 dmask = best_vs_cand_mask(best_distances, distances);
        __mmask16 finalmask = m1mask | dmask;

        const __m512i best_indices_new = _mm512_mask_blend_epi32(
                finalmask, current_indices, best_indices);
        const __m512 best_distances_new =
                _mm512_mask_blend_ps(finalmask, distances, best_distances);

        best_indices = best_indices_new;
        best_distances = best_distances_new;
    }

    // Horizontal best: min for CMax (distance), max for CMin (similarity).
    if constexpr (HC::is_max) {
        best_dis = _mm512_reduce_min_ps(best_distances);
    } else {
        best_dis = _mm512_reduce_max_ps(best_distances);
    }
    // Tiebreak by picking the rightmost (largest) index among lanes
    // matching the best distance, matching the original behavior.
    __mmask16 best_lane_mask =
            _mm512_cmpeq_ps_mask(best_distances, _mm512_set1_ps(best_dis));
    best_idx = _mm512_mask_reduce_max_epi32(best_lane_mask, best_indices);

    if (best_idx == -1) {
        return -1;
    }

    if (vmin_out) {
        *vmin_out = best_dis;
    }
    int ret = heap.ids[best_idx];
    heap.ids[best_idx] = -1;
    --heap.nvalid;
    return ret;
}

} // namespace

// Explicit specializations for AVX512
template <>
int pop_min_tpl<CMax<float, int32_t>, SIMDLevel::AVX512>(
        MinimaxHeapT<CMax<float, int32_t>>* heap,
        float* vmin_out) {
    return pop_best_avx512<CMax<float, int32_t>>(*heap, vmin_out);
}

template <>
int pop_min_tpl<CMin<float, int32_t>, SIMDLevel::AVX512>(
        MinimaxHeapT<CMin<float, int32_t>>* heap,
        float* vmin_out) {
    return pop_best_avx512<CMin<float, int32_t>>(*heap, vmin_out);
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX512
