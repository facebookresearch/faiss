/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AVX2 specialisation of Top1 add_results (8-wide branchless argmin/argmax).
// Reservoir stays on the NONE path — VPCOMPRESSPS requires AVX512F.

#ifdef COMPILE_SIMD_AVX2

#include <faiss/impl/ResultHandler.h>

#include <immintrin.h>
#include <type_traits>

namespace faiss {

namespace {

/// Templated AVX2 implementation of Top1 add_results for both CMax (keeps the
/// smallest distance) and CMin (keeps the largest similarity).
template <class C, bool use_sel>
void top1_add_results_avx2(
        Top1BlockResultHandler<C, use_sel>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab_in) {
    static_assert(
            std::is_same<typename C::T, float>::value,
            "This code expects float distances");
    using TI = typename C::TI;
    const __m256i vstep = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    for (size_t qi = self->i0; qi < self->i1; qi++) {
        const float* dis_tab_i = dis_tab_in + (j1 - j0) * (qi - self->i0) - j0;

        // Hoist best_dis / best_idx into locals so the compiler keeps them in
        // registers across the inner loop (no aliasing with dis_tab reads).
        float best_dis = self->dis_tab[qi];
        TI best_idx = self->ids_tab[qi];
        size_t j = j0;

        __m256 vbest = _mm256_set1_ps(best_dis);
        __m256i vbest_idx = _mm256_set1_epi32((int32_t)best_idx);

        for (; j + 8 <= j1; j += 8) {
            __m256 vdis = _mm256_loadu_ps(dis_tab_i + j);
            __m256i vidx =
                    _mm256_add_epi32(_mm256_set1_epi32((int32_t)j), vstep);

            // CMax (L2 nearest neighbour): keep lane if dis < best.
            // CMin (inner product):        keep lane if dis > best.
            __m256 mask;
            if constexpr (C::is_max) {
                mask = _mm256_cmp_ps(vdis, vbest, _CMP_LT_OS);
            } else {
                mask = _mm256_cmp_ps(vdis, vbest, _CMP_GT_OS);
            }
            vbest = _mm256_blendv_ps(vbest, vdis, mask);
            vbest_idx = _mm256_blendv_epi8(
                    vbest_idx, vidx, _mm256_castps_si256(mask));
        }

        // Horizontal reduction across 8 lanes.
        alignas(32) float best_arr[8];
        alignas(32) int32_t idx_arr[8];
        _mm256_store_ps(best_arr, vbest);
        _mm256_store_si256((__m256i*)idx_arr, vbest_idx);
        for (int k = 0; k < 8; k++) {
            if (C::cmp(best_dis, best_arr[k])) {
                best_dis = best_arr[k];
                best_idx = (TI)idx_arr[k];
            }
        }

        // Scalar tail.
        for (; j < j1; j++) {
            if (C::cmp(best_dis, dis_tab_i[j])) {
                best_dis = dis_tab_i[j];
                best_idx = (TI)j;
            }
        }

        self->dis_tab[qi] = best_dis;
        self->ids_tab[qi] = best_idx;
    }
}

} // namespace

// Explicit specialisations for AVX2

template <>
void top1_add_results_tpl<CMax<float, int64_t>, false, SIMDLevel::AVX2>(
        Top1BlockResultHandler<CMax<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_avx2<CMax<float, int64_t>, false>(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<CMax<float, int64_t>, true, SIMDLevel::AVX2>(
        Top1BlockResultHandler<CMax<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_avx2<CMax<float, int64_t>, true>(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<CMin<float, int64_t>, false, SIMDLevel::AVX2>(
        Top1BlockResultHandler<CMin<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_avx2<CMin<float, int64_t>, false>(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<CMin<float, int64_t>, true, SIMDLevel::AVX2>(
        Top1BlockResultHandler<CMin<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_avx2<CMin<float, int64_t>, true>(self, j0, j1, dis_tab);
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX2
