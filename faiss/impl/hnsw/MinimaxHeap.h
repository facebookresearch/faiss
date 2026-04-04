/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/utils/Heap.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

/** Heap structure that allows fast access and updates.
 *
 * Supports both max-heap operations (via the underlying CMax heap)
 * and efficient min extraction via linear scan (with optional SIMD
 * acceleration).
 */
struct MinimaxHeap {
    using storage_idx_t = int32_t;

    int n;
    int k;
    int nvalid;

    std::vector<storage_idx_t> ids;
    std::vector<float> dis;
    using HC = faiss::CMax<float, storage_idx_t>;

    explicit MinimaxHeap(int n_in)
            : n(n_in), k(0), nvalid(0), ids(n_in), dis(n_in) {}

    void push(storage_idx_t i, float v);

    float max() const {
        return dis[0];
    }

    int size() const {
        return nvalid;
    }

    void clear() {
        nvalid = k = 0;
    }

    /// SIMD-templated pop_min implementation.
    /// Specializations exist for NONE, AVX2, and AVX512.
    template <SIMDLevel SL>
    int pop_min_tpl(float* vmin_out = nullptr);

    /// Runtime-dispatched pop_min (calls pop_min_tpl with best available
    /// SIMD level).
    int pop_min(float* vmin_out = nullptr);

    int count_below(float thresh);
};

} // namespace faiss
