/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <faiss/impl/hnsw/MinimaxHeap.h>

#include <cassert>

#include <faiss/impl/simd_dispatch.h>

namespace faiss {

void MinimaxHeap::push(storage_idx_t i, float v) {
    // Treat NaN distances as infinitely far away so heap ordering is preserved.
    if (std::isnan(v)) {
        v = HC::neutral();
    }
    if (k == n) {
        if (v >= dis[0]) {
            return;
        }
        if (ids[0] != -1) {
            --nvalid;
        }
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

// Scalar (NONE) specialization of pop_min_tpl
template <>
int MinimaxHeap::pop_min_tpl<SIMDLevel::NONE>(float* vmin_out) {
    assert(k > 0);
    // returns min. This is an O(n) operation
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
        if (ids[i] != -1 && dis[i] < vmin) {
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
    --nvalid;

    return ret;
}

// Runtime-dispatched pop_min (NONE + AVX2 + AVX512 only)
constexpr int MINIMAX_HEAP_SIMD_LEVELS = (1 << int(SIMDLevel::NONE)) |
        (1 << int(SIMDLevel::AVX2)) | (1 << int(SIMDLevel::AVX512));

int MinimaxHeap::pop_min(float* vmin_out) {
    return with_selected_simd_levels<MINIMAX_HEAP_SIMD_LEVELS>(
            [&]<SIMDLevel SL>() { return pop_min_tpl<SL>(vmin_out); });
}

int MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss
