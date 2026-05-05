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

template <>
int MinimaxHeap::pop_min_tpl<SIMDLevel::AVX2>(float* vmin_out) {
    assert(k > 0);
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    size_t iii = 0;

    __m256i min_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
    __m256 min_distances =
            _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i offset = _mm256_set1_epi32(8);

    // The baseline version is available in the NONE specialization.

    // The following loop tracks the rightmost index with the min distance.
    // -1 index values are ignored.
    const size_t k8 = (k / 8) * 8;
    for (; iii < k8; iii += 8) {
        __m256i indices =
                _mm256_loadu_si256((const __m256i*)(ids.data() + iii));
        __m256 distances = _mm256_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __m256i m1mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);

        __m256i dmask = _mm256_castps_si256(
                _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS));
        __m256 finalmask = _mm256_castsi256_ps(_mm256_or_si256(m1mask, dmask));

        const __m256i min_indices_new = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(current_indices),
                _mm256_castsi256_ps(min_indices),
                finalmask));

        const __m256 min_distances_new =
                _mm256_blendv_ps(distances, min_distances, finalmask);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm256_add_epi32(current_indices, offset);
    }

    // Vectorizing is doable, but is not practical
    int32_t vidx8[8];
    float vdis8[8];
    _mm256_storeu_ps(vdis8, min_distances);
    _mm256_storeu_si256((__m256i*)vidx8, min_indices);

    for (size_t j = 0; j < 8; j++) {
        if (min_dis > vdis8[j] || (min_dis == vdis8[j] && min_idx < vidx8[j])) {
            min_idx = vidx8[j];
            min_dis = vdis8[j];
        }
    }

    // process last values. Vectorizing is doable, but is not practical
    for (; iii < static_cast<size_t>(k); iii++) {
        if (ids[iii] != -1 && dis[iii] <= min_dis) {
            min_dis = dis[iii];
            min_idx = iii;
        }
    }

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

#endif // COMPILE_SIMD_AVX2
