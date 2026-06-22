/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/hnsw/MinimaxHeap.h>

#include <faiss/impl/simd_dispatch.h>

namespace faiss {

// Runtime-dispatched pop_min (NONE + AVX2 + AVX512 only).
constexpr int MINIMAX_HEAP_SIMD_LEVELS = (1 << int(SIMDLevel::NONE)) |
        (1 << int(SIMDLevel::AVX2)) | (1 << int(SIMDLevel::AVX512));

template <class HC_>
int MinimaxHeapT<HC_>::pop_min(float* vmin_out) {
    return with_selected_simd_levels<MINIMAX_HEAP_SIMD_LEVELS>(
            [&]<SIMDLevel SL>() {
                return pop_min_tpl<HC_, SL>(this, vmin_out);
            });
}

// Primary-template scalar implementation. Used directly when SL==NONE
template <class HC>
int pop_min_simd_none(MinimaxHeapT<HC>* heap, float* vmin_out) {
    int k = heap->k;
    int* ids = heap->ids.data();
    float* dis = heap->dis.data();
    assert(k > 0);
    // Returns the "best" entry. This is an O(n) operation.
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1) {
            break;
        }
        i--;
    }
    if (i == -1) {
        return -1;
    }
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        // HC::cmp(vmin, dis[i]) → "dis[i] is better than vmin".
        if (ids[i] != -1 && HC::cmp(vmin, dis[i])) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) {
        *vmin_out = vmin;
    }
    int ret = ids[imin];
    ids[imin] = -1;
    --heap->nvalid;
    return ret;
}

// declare for min and max heap at simd level NONE
template <>
int pop_min_tpl<CMin<float, int32_t>, SIMDLevel::NONE>(
        MinimaxHeapT<CMin<float, int32_t>>* heap,
        float* vmin_out) {
    return pop_min_simd_none(heap, vmin_out);
}

template <>
int pop_min_tpl<CMax<float, int32_t>, SIMDLevel::NONE>(
        MinimaxHeapT<CMax<float, int32_t>>* heap,
        float* vmin_out) {
    return pop_min_simd_none(heap, vmin_out);
}

// Explicit instantiations of pop_min for the two HC variants
template int MinimaxHeapT<CMax<float, int32_t>>::pop_min(float*);
template int MinimaxHeapT<CMin<float, int32_t>>::pop_min(float*);

} // namespace faiss
