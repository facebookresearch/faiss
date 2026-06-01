/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include <faiss/utils/Heap.h>
#include <faiss/utils/ordered_key_value.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

/** Heap structure that allows fast access and updates.
 *
 * Templated on the comparator HC_ so that the same data structure can
 * service both distance-style searches (HC_ = CMax<float, int32_t>, smaller
 * is better) and similarity-style searches (HC_ = CMin<float, int32_t>,
 * larger is better). For the distance variant the underlying heap is a
 * max-heap and "pop_min" returns the closest element; for similarity the
 * underlying heap is a min-heap and "pop_min" returns the most similar
 * element.
 */
template <class HC_ = CMax<float, int32_t>>
struct MinimaxHeapT {
    using HC = HC_;
    using storage_idx_t = int32_t;

    int n;
    int k;
    int nvalid;

    std::vector<storage_idx_t> ids;
    std::vector<float> dis;

    explicit MinimaxHeapT(int n_in)
            : n(n_in), k(0), nvalid(0), ids(n_in), dis(n_in) {}

    void push(storage_idx_t i, float v) {
        // Treat NaN distances as the "worst" value so heap ordering is
        // preserved (insertion is then guaranteed to fall through the
        // not-better-than-top early-reject branch when the heap is full).
        if (std::isnan(v)) {
            v = HC::neutral();
        }
        if (k == n) {
            // top of the heap is the "worst" entry under HC. If the new
            // value is not strictly better than the worst, drop it.
            // HC::cmp(top, v) means "v is better than top" for both CMax
            // (cmp = a > b → top > v → v < top) and CMin (cmp = a < b →
            // top < v → v > top).
            if (!HC::cmp(dis[0], v)) {
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

    float max() const {
        return dis[0];
    }

    int size() const {
        return nvalid;
    }

    void clear() {
        nvalid = k = 0;
    }

    /// Runtime-dispatched best-element extraction (NONE + AVX2 + AVX512).
    int pop_min(float* vmin_out = nullptr);

    int count_below(float thresh) {
        int n_below = 0;
        for (int i = 0; i < k; i++) {
            // Count entries that are strictly "better than" thresh.
            // HC::cmp(thresh, dis[i]) → for CMax: thresh > dis[i]
            // (i.e., dis[i] < thresh, the historical L2 semantics);
            // for CMin: thresh < dis[i] (similarity above threshold).
            if (HC::cmp(thresh, dis[i])) {
                n_below++;
            }
        }
        return n_below;
    }
};

// Default `MinimaxHeap` keeps the historical max-heap semantics (smaller
// distance is better). The CMin instantiation is used when the owning
// HNSW has `is_similarity = true`. The alias itself is declared once,
// alongside the forward declaration in HNSW.h, to avoid duplicate
// `using` declarations that SWIG treats as redundant.

// Forward declarations of the SIMD specializations. The actual bodies live
// in the SIMD-specific translation units (avx2.cpp, avx512.cpp) and are
// resolved at link time.
template <class HC_, SIMDLevel SL>
int pop_min_tpl(MinimaxHeapT<HC_>* heap, float* vmin_out);

} // namespace faiss
