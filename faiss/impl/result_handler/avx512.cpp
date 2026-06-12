/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AVX-512 specialisations of Top1 and Reservoir add_results.
//
// Top-1:    16-wide branchless argmin/argmax via mask_blend.
// Reservoir: VPCOMPRESSPS / VPCOMPRESSD bulk-insert of passing elements,
//            eliminating the per-element threshold branch entirely.

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/popcount.h>

#include <immintrin.h>
#include <type_traits>

namespace faiss {

namespace {

/// Templated AVX-512 implementation of Top1 add_results for both CMax (keeps
/// the smallest distance) and CMin (keeps the largest similarity).
template <class C, bool use_sel>
void top1_add_results_avx512(
        Top1BlockResultHandler<C, use_sel>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab_in) {
    static_assert(
            std::is_same<typename C::T, float>::value,
            "This code expects float distances");
    using TI = typename C::TI;
    const __m512i vstep = _mm512_set_epi32(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    for (size_t qi = self->i0; qi < self->i1; qi++) {
        const float* dis_tab_i = dis_tab_in + (j1 - j0) * (qi - self->i0) - j0;

        // Hoist best_dis / best_idx into locals so the compiler keeps them in
        // registers across the inner loop (no aliasing with dis_tab reads).
        float best_dis = self->dis_tab[qi];
        TI best_idx = self->ids_tab[qi];
        size_t j = j0;

        __m512 vbest = _mm512_set1_ps(best_dis);
        __m512i vbest_idx = _mm512_set1_epi32((int32_t)best_idx);

        for (; j + 16 <= j1; j += 16) {
            __m512 vdis = _mm512_loadu_ps(dis_tab_i + j);
            __m512i vidx =
                    _mm512_add_epi32(_mm512_set1_epi32((int32_t)j), vstep);

            // CMax (L2 nearest neighbour): keep lane if dis < best.
            // CMin (inner product):        keep lane if dis > best.
            __mmask16 mask;
            if constexpr (C::is_max) {
                mask = _mm512_cmp_ps_mask(vdis, vbest, _CMP_LT_OS);
            } else {
                mask = _mm512_cmp_ps_mask(vdis, vbest, _CMP_GT_OS);
            }
            vbest = _mm512_mask_blend_ps(mask, vbest, vdis);
            vbest_idx = _mm512_mask_blend_epi32(mask, vbest_idx, vidx);
        }

        // Horizontal reduction across 16 lanes.
        alignas(64) float best_arr[16];
        alignas(64) int32_t idx_arr[16];
        _mm512_store_ps(best_arr, vbest);
        _mm512_store_si512((__m512i*)idx_arr, vbest_idx);
        for (int k = 0; k < 16; k++) {
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

/// Templated AVX-512 implementation of Reservoir add_results for both CMax
/// and CMin.  Uses VPCOMPRESSPS / VPCOMPRESSD to bulk-insert all elements that
/// beat the current threshold in a single pass, avoiding the per-element branch
/// that dominates the scalar path.
///
/// Falls back to the scalar NONE path for small reservoirs (capacity < 64)
/// where the compress-path setup cost outweighs its throughput benefit.
template <class C, bool use_sel>
void reservoir_add_results_avx512(
        ReservoirBlockResultHandler<C, use_sel>* self,
        size_t j0,
        size_t j1,
        const float* dis_in) {
    static_assert(
            std::is_same<typename C::T, float>::value,
            "This code expects float distances");
    static_assert(
            std::is_same<typename C::TI, int64_t>::value,
            "This code expects int64_t indices");

    // AVX-512 compress amortizes its setup cost only for large reservoirs.
    // Benchmarks show a ~9% regression at k=10 (capacity≈20) and a ~25%
    // gain at k=100 (capacity≈200).  32 = 2 × lane-width is the crossover.
    // All reservoirs in a handler share the same capacity, so check once.
    constexpr size_t AVX512_RESERVOIR_MIN_CAPACITY = 32;
    if (self->i0 < self->i1 &&
        self->reservoirs[0].capacity < AVX512_RESERVOIR_MIN_CAPACITY) {
        reservoir_add_results_tpl<C, use_sel, SIMDLevel::NONE>(
                self, j0, j1, dis_in);
        return;
    }

    const __m512i vstep = _mm512_set_epi32(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

#pragma omp parallel for
    for (int64_t qi = (int64_t)self->i0; qi < (int64_t)self->i1; qi++) {
        ReservoirTopN<C>& res = self->reservoirs[qi - (int64_t)self->i0];
        const float* dis_tab_i =
                dis_in + (j1 - j0) * (qi - (int64_t)self->i0) - j0;
        size_t j = j0;

        for (; j + 16 <= j1; j += 16) {
            // Near-capacity: fewer than 16 free slots remain.  Fall back to
            // the scalar add_result which handles overflow correctly.
            if (res.i + 16 > res.capacity) {
                for (size_t jj = j; jj < j + 16; jj++)
                    res.add_result(dis_tab_i[jj], jj);
                continue;
            }

            __m512 vthresh = _mm512_set1_ps(res.threshold);
            __m512 vdis = _mm512_loadu_ps(dis_tab_i + j);

            // CMax (L2): keep elements with dis < threshold.
            // CMin (IP): keep elements with dis > threshold.
            __mmask16 mask;
            if constexpr (C::is_max) {
                mask = _mm512_cmp_ps_mask(vdis, vthresh, _CMP_LT_OS);
            } else {
                mask = _mm512_cmp_ps_mask(vdis, vthresh, _CMP_GT_OS);
            }
            if (mask == 0)
                continue;

            int count = popcount32(mask);

            // Compress passing distances into a contiguous run (VPCOMPRESSPS).
            __m512 passing_dis = _mm512_maskz_compress_ps(mask, vdis);

            // Compress the sequential indices j..j+15 (VPCOMPRESSD).
            __m512i vidx32 =
                    _mm512_add_epi32(_mm512_set1_epi32((int32_t)j), vstep);
            __m512i passing_idx32 = _mm512_maskz_compress_epi32(mask, vidx32);

            // Unconditional 16-element stores are safe: the res.i + 16 <=
            // res.capacity check above guarantees slots res.i .. res.i+15
            // are all unused.  The (count .. 15) tail positions get garbage
            // that res.i never reaches.
            _mm512_storeu_ps(res.vals + res.i, passing_dis);

            // Widen int32 indices to int64 (TI = int64_t) in two 8-element
            // halves and store them.
            _mm512_storeu_si512(
                    (void*)(res.ids + res.i),
                    _mm512_cvtepi32_epi64(
                            _mm512_castsi512_si256(passing_idx32)));
            _mm512_storeu_si512(
                    (void*)(res.ids + res.i + 8),
                    _mm512_cvtepi32_epi64(
                            _mm512_extracti64x4_epi64(passing_idx32, 1)));

            res.i += count;
            if (res.i >= res.capacity) {
                res.shrink_fuzzy();
            }
        }

        // Scalar tail.
        for (; j < j1; j++)
            res.add_result(dis_tab_i[j], j);
    }
}

} // namespace

// Explicit specialisations for AVX-512

template <>
void top1_add_results_tpl<CMax<float, int64_t>, false, SIMDLevel::AVX512>(
        Top1BlockResultHandler<CMax<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_avx512<CMax<float, int64_t>, false>(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<CMax<float, int64_t>, true, SIMDLevel::AVX512>(
        Top1BlockResultHandler<CMax<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_avx512<CMax<float, int64_t>, true>(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<CMin<float, int64_t>, false, SIMDLevel::AVX512>(
        Top1BlockResultHandler<CMin<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_avx512<CMin<float, int64_t>, false>(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<CMin<float, int64_t>, true, SIMDLevel::AVX512>(
        Top1BlockResultHandler<CMin<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_avx512<CMin<float, int64_t>, true>(self, j0, j1, dis_tab);
}

template <>
void reservoir_add_results_tpl<CMax<float, int64_t>, false, SIMDLevel::AVX512>(
        ReservoirBlockResultHandler<CMax<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_in) {
    reservoir_add_results_avx512<CMax<float, int64_t>, false>(
            self, j0, j1, dis_in);
}

template <>
void reservoir_add_results_tpl<CMax<float, int64_t>, true, SIMDLevel::AVX512>(
        ReservoirBlockResultHandler<CMax<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_in) {
    reservoir_add_results_avx512<CMax<float, int64_t>, true>(
            self, j0, j1, dis_in);
}

template <>
void reservoir_add_results_tpl<CMin<float, int64_t>, false, SIMDLevel::AVX512>(
        ReservoirBlockResultHandler<CMin<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_in) {
    reservoir_add_results_avx512<CMin<float, int64_t>, false>(
            self, j0, j1, dis_in);
}

template <>
void reservoir_add_results_tpl<CMin<float, int64_t>, true, SIMDLevel::AVX512>(
        ReservoirBlockResultHandler<CMin<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_in) {
    reservoir_add_results_avx512<CMin<float, int64_t>, true>(
            self, j0, j1, dis_in);
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX512
