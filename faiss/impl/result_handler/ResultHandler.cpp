/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic (NONE) implementations of Top1 and Reservoir add_results, plus the
// runtime-dispatch method bodies.  SIMD specialisations live in
// result_handler_avx2.cpp and result_handler_avx512.cpp.

#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/simd_dispatch.h>

namespace faiss {

// ----------------------------------------------------------------
// SIMD-level masks
// ----------------------------------------------------------------

// Top-1: scalar fallback + AVX2 (8-wide) + AVX512 (16-wide).
constexpr int TOP1_SIMD_LEVELS = (1 << int(SIMDLevel::NONE)) |
        (1 << int(SIMDLevel::AVX2)) | (1 << int(SIMDLevel::AVX512));

// Reservoir: scalar fallback + AVX512 compress path.
// VPCOMPRESSPS/VPCOMPRESSD require AVX512F so there is no AVX2 path.
// On non-AVX512 hosts the dispatch falls back to NONE automatically.
constexpr int RESERVOIR_SIMD_LEVELS =
        (1 << int(SIMDLevel::NONE)) | (1 << int(SIMDLevel::AVX512));

// ----------------------------------------------------------------
// Scalar (NONE) helper implementations
// ----------------------------------------------------------------

namespace {

template <class C, bool use_sel>
void top1_add_results_none(
        Top1BlockResultHandler<C, use_sel>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab_in) {
    using T = typename C::T;
    using TI = typename C::TI;

    for (size_t qi = self->i0; qi < self->i1; qi++) {
        const T* dis_tab_i = dis_tab_in + (j1 - j0) * (qi - self->i0) - j0;

        // Hoist best_dis / best_idx into locals so the compiler keeps them in
        // registers across the inner loop (no aliasing with dis_tab reads).
        T best_dis = self->dis_tab[qi];
        TI best_idx = self->ids_tab[qi];

        for (size_t j = j0; j < j1; j++) {
            if (C::cmp(best_dis, dis_tab_i[j])) {
                best_dis = dis_tab_i[j];
                best_idx = (TI)j;
            }
        }

        self->dis_tab[qi] = best_dis;
        self->ids_tab[qi] = best_idx;
    }
}

template <class C, bool use_sel>
void reservoir_add_results_none(
        ReservoirBlockResultHandler<C, use_sel>* self,
        size_t j0,
        size_t j1,
        const float* dis_in) {
    using T = typename C::T;
    using TI = typename C::TI;

#pragma omp parallel for
    for (int64_t qi = (int64_t)self->i0; qi < (int64_t)self->i1; qi++) {
        ReservoirTopN<C>& res = self->reservoirs[qi - (int64_t)self->i0];
        const T* dis_tab_i = dis_in + (j1 - j0) * (qi - (int64_t)self->i0) - j0;

        // Hoist res.i and res.threshold into locals so the compiler keeps
        // them in registers.
        size_t ri = res.i;
        T thresh = res.threshold;

        for (size_t j = j0; j < j1; j++) {
            T dis = dis_tab_i[j];
            if (C::cmp(thresh, dis)) {
                res.vals[ri] = dis;
                res.ids[ri] = (TI)j;
                ri++;
                if (ri >= res.capacity) {
                    res.i = ri;
                    res.shrink_fuzzy();
                    ri = res.i;
                    thresh = res.threshold;
                }
            }
        }
        res.i = ri;
    }
}

} // namespace

// ----------------------------------------------------------------
// SIMDLevel::NONE explicit specialisations
// ----------------------------------------------------------------

// Instantiate top1_add_results_tpl<C, use_sel, SIMDLevel::NONE> and
// reservoir_add_results_tpl<C, use_sel, SIMDLevel::NONE> for all
// (C, use_sel) combinations that the rest of FAISS uses.
#define INSTANTIATE_NONE(C, use_sel)                                  \
    template <>                                                       \
    void top1_add_results_tpl<C, use_sel, SIMDLevel::NONE>(           \
            Top1BlockResultHandler<C, use_sel> * self,                \
            size_t j0,                                                \
            size_t j1,                                                \
            const float* dis_tab) {                                   \
        top1_add_results_none<C, use_sel>(self, j0, j1, dis_tab);     \
    }                                                                 \
    template <>                                                       \
    void reservoir_add_results_tpl<C, use_sel, SIMDLevel::NONE>(      \
            ReservoirBlockResultHandler<C, use_sel> * self,           \
            size_t j0,                                                \
            size_t j1,                                                \
            const float* dis_in) {                                    \
        reservoir_add_results_none<C, use_sel>(self, j0, j1, dis_in); \
    }

// Type aliases so the comma in CMax<float, int64_t> doesn't split macro args.
using CMaxFI = CMax<float, int64_t>;
using CMinFI = CMin<float, int64_t>;

INSTANTIATE_NONE(CMaxFI, false)
INSTANTIATE_NONE(CMaxFI, true)
INSTANTIATE_NONE(CMinFI, false)
INSTANTIATE_NONE(CMinFI, true)

#undef INSTANTIATE_NONE

// ----------------------------------------------------------------
// add_results method definitions — dispatch to the right SL kernel
// ----------------------------------------------------------------

template <class C, bool use_sel>
void Top1BlockResultHandler<C, use_sel>::add_results(
        size_t j0,
        size_t j1,
        const T* dis_tab_2) {
    with_selected_simd_levels<TOP1_SIMD_LEVELS>([&]<SIMDLevel SL>() {
        top1_add_results_tpl<C, use_sel, SL>(this, j0, j1, dis_tab_2);
    });
}

template <class C, bool use_sel>
void ReservoirBlockResultHandler<C, use_sel>::add_results(
        size_t j0,
        size_t j1,
        const T* dis_in) {
    with_selected_simd_levels<RESERVOIR_SIMD_LEVELS>([&]<SIMDLevel SL>() {
        reservoir_add_results_tpl<C, use_sel, SL>(this, j0, j1, dis_in);
    });
}

// ----------------------------------------------------------------
// Explicit class-template instantiations (force linkage)
// ----------------------------------------------------------------

template void Top1BlockResultHandler<CMax<float, int64_t>, false>::add_results(
        size_t,
        size_t,
        const float*);
template void Top1BlockResultHandler<CMax<float, int64_t>, true>::add_results(
        size_t,
        size_t,
        const float*);
template void Top1BlockResultHandler<CMin<float, int64_t>, false>::add_results(
        size_t,
        size_t,
        const float*);
template void Top1BlockResultHandler<CMin<float, int64_t>, true>::add_results(
        size_t,
        size_t,
        const float*);

template void ReservoirBlockResultHandler<CMax<float, int64_t>, false>::
        add_results(size_t, size_t, const float*);
template void ReservoirBlockResultHandler<CMax<float, int64_t>, true>::
        add_results(size_t, size_t, const float*);
template void ReservoirBlockResultHandler<CMin<float, int64_t>, false>::
        add_results(size_t, size_t, const float*);
template void ReservoirBlockResultHandler<CMin<float, int64_t>, true>::
        add_results(size_t, size_t, const float*);

} // namespace faiss
