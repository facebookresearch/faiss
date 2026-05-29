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

template <>
int MinimaxHeap::pop_min_tpl<SIMDLevel::AVX512>(float* vmin_out) {
    assert(k > 0);
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    __m512i min_indices = _mm512_set1_epi32(-1);
    __m512 min_distances =
            _mm512_set1_ps(std::numeric_limits<float>::infinity());
    __m512i current_indices = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i offset = _mm512_set1_epi32(16);

    // The following loop tracks the rightmost index with the min distance.
    // -1 index values are ignored.
    const size_t k16 = (k / 16) * 16;
    for (size_t iii = 0; iii < k16; iii += 16) {
        __m512i indices =
                _mm512_loadu_si512((const __m512i*)(ids.data() + iii));
        __m512 distances = _mm512_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __mmask16 m1mask =
                _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);

        __mmask16 dmask =
                _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        const __m512i min_indices_new = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        const __m512 min_distances_new =
                _mm512_mask_blend_ps(finalmask, distances, min_distances);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm512_add_epi32(current_indices, offset);
    }

    // leftovers
    if (k16 != static_cast<size_t>(k)) {
        const __mmask16 kmask = (1 << (k - k16)) - 1;

        __m512i indices = _mm512_mask_loadu_epi32(
                _mm512_set1_epi32(-1), kmask, ids.data() + k16);
        __m512 distances = _mm512_maskz_loadu_ps(kmask, dis.data() + k16);

        // This mask filters out -1 values among indices.
        __mmask16 m1mask =
                _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);

        __mmask16 dmask =
                _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        const __m512i min_indices_new = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        const __m512 min_distances_new =
                _mm512_mask_blend_ps(finalmask, distances, min_distances);

        min_indices = min_indices_new;
        min_distances = min_distances_new;
    }

    // grab min distance
    min_dis = _mm512_reduce_min_ps(min_distances);
    // blend
    __mmask16 mindmask =
            _mm512_cmpeq_ps_mask(min_distances, _mm512_set1_ps(min_dis));
    // pick the max one
    min_idx = _mm512_mask_reduce_max_epi32(mindmask, min_indices);

    if (min_idx == -1) {
        return -1;
    }

    if (vmin_out) {
        *vmin_out = min_dis;
    }
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX512
