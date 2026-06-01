/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#include <faiss/impl/hnsw/MinimaxHeap.h>

#include <immintrin.h>
#include <cassert>
#include <limits>
#include <type_traits>

namespace faiss {

namespace {

/// Templated AVX2 implementation of "pop best" for both CMax (returns
/// the smallest distance) and CMin (returns the largest similarity).
/// The only differences between the two flavors are: (1) the initial
/// "worst possible" value, (2) the running-best update comparison
/// (`_CMP_LT_OS` vs `_CMP_GT_OS`), and (3) the tiebreaker direction.
template <class HC>
int pop_best_avx2(MinimaxHeapT<HC>& heap, float* vmin_out) {
    using storage_idx_t = typename MinimaxHeapT<HC>::storage_idx_t;
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");
    assert(heap.k > 0);

    // For CMax (distance) the "best" candidate is the smallest value, so
    // we initialize the running best to +inf. For CMin (similarity) the
    // best is the largest value, so we initialize to -inf.
    constexpr float worst_v = HC::is_max
            ? std::numeric_limits<float>::infinity()
            : -std::numeric_limits<float>::infinity();

    int32_t best_idx = -1;
    float best_dis = worst_v;

    size_t iii = 0;

    __m256i best_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
    __m256 best_distances = _mm256_set1_ps(worst_v);
    __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i offset = _mm256_set1_epi32(8);

    // Track the rightmost index whose distance equals the running best.
    // -1 index values are filtered out via m1mask.
    const size_t k8 = (heap.k / 8) * 8;
    for (; iii < k8; iii += 8) {
        __m256i indices =
                _mm256_loadu_si256((const __m256i*)(heap.ids.data() + iii));
        __m256 distances = _mm256_loadu_ps(heap.dis.data() + iii);

        // Mask out -1 indices (invalid entries).
        __m256i m1mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);

        // dmask is "true where best is already (strictly) better than the
        // candidate" — entries the candidate should NOT update. For CMax,
        // best < candidate means we keep best (we want the smallest);
        // for CMin we keep best when best > candidate (we want the largest).
        __m256i dmask;
        if constexpr (HC::is_max) {
            dmask = _mm256_castps_si256(
                    _mm256_cmp_ps(best_distances, distances, _CMP_LT_OS));
        } else {
            dmask = _mm256_castps_si256(
                    _mm256_cmp_ps(best_distances, distances, _CMP_GT_OS));
        }
        __m256 finalmask = _mm256_castsi256_ps(_mm256_or_si256(m1mask, dmask));

        const __m256i best_indices_new = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(current_indices),
                _mm256_castsi256_ps(best_indices),
                finalmask));

        const __m256 best_distances_new =
                _mm256_blendv_ps(distances, best_distances, finalmask);

        best_indices = best_indices_new;
        best_distances = best_distances_new;

        current_indices = _mm256_add_epi32(current_indices, offset);
    }

    // Vectorizing the horizontal reduction is doable but not practical.
    int32_t vidx8[8];
    float vdis8[8];
    _mm256_storeu_ps(vdis8, best_distances);
    _mm256_storeu_si256((__m256i*)vidx8, best_indices);

    for (size_t j = 0; j < 8; j++) {
        const bool strictly_better =
                HC::is_max ? (best_dis > vdis8[j]) : (best_dis < vdis8[j]);
        if (strictly_better || (best_dis == vdis8[j] && best_idx < vidx8[j])) {
            best_idx = vidx8[j];
            best_dis = vdis8[j];
        }
    }

    // Tail (under 8 entries). Vectorizing is doable but not practical.
    for (; iii < static_cast<size_t>(heap.k); iii++) {
        if (heap.ids[iii] == -1) {
            continue;
        }
        const bool weakly_better = HC::is_max ? (best_dis >= heap.dis[iii])
                                              : (best_dis <= heap.dis[iii]);
        if (weakly_better) {
            best_dis = heap.dis[iii];
            best_idx = iii;
        }
    }

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

// Explicit specializations for AVX2
template <>
int pop_min_tpl<CMax<float, int32_t>, SIMDLevel::AVX2>(
        MinimaxHeapT<CMax<float, int32_t>>* heap,
        float* vmin_out) {
    return pop_best_avx2<CMax<float, int32_t>>(*heap, vmin_out);
}

template <>
int pop_min_tpl<CMin<float, int32_t>, SIMDLevel::AVX2>(
        MinimaxHeapT<CMin<float, int32_t>>* heap,
        float* vmin_out) {
    return pop_best_avx2<CMin<float, int32_t>>(*heap, vmin_out);
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX2
