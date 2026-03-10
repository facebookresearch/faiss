/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <immintrin.h>
#include "partition.h"

#include <algorithm>
#include <vector>

namespace faiss {

/**
 * AVX-512 optimized reservoir result handler for top-k selection.
 *
 * This handler collects results one by one and uses the AVX-512
 * optimized partition algorithm when compaction is needed.
 *
 * Handles distances where lower is better (e.g., L2 distance).
 * Keeps the k smallest distances.
 *
 * Usage:
 *   ReservoirResultHandlerAVX512 handler(k);
 *   for (size_t q = 0; q < nq; q++) {
 *       handler.begin();
 *       index->search1(xq + q * d, handler);
 *       handler.end(D + q * k, I + q * k);
 *   }
 */
struct ReservoirResultHandlerAVX512 : ResultHandler {
    // total size (rounded up to multiple of 16)
    size_t capacity;
    // number we want to keep
    size_t k;
    // number currently stored
    size_t n;

    // Storage arrays
    std::vector<float> vals;
    std::vector<int32_t> idxs;

    /**
     * Constructor
     * @param k           Number of results to keep
     * @param capacity_in Optional capacity for reservoir (default: 2*k rounded
     * up to 16)
     */
    explicit ReservoirResultHandlerAVX512(size_t k, size_t capacity_in = 0)
            : k(k) {
        capacity = capacity_in > 0 ? capacity_in : 2 * k;
        if (capacity < k + 16) {
            capacity = k + 16;
        }
        capacity = (capacity + 15) & ~15; // Round up to multiple of 16
        vals.resize(capacity);
        idxs.resize(capacity);
        begin();
    }

    /**
     * Begin a new query. Resets the handler state.
     */
    void begin() {
        n = 0;
        threshold = HUGE_VALF;
    }

    /**
     * Compact the reservoir by partitioning to keep only k smallest elements.
     */
    void compact() {
        assert(n >= k);
        // argpartition(n, ..., n-k) puts (n-k) largest at [k, n), leaving k
        // smallest at [0, k)
        argpartition(n, vals.data(), idxs.data(), n - k);
        // threshold is max of k smallest values
        threshold = *std::max_element(vals.data(), vals.data() + k);
        n = k;
    }

    /**
     * Add one result.
     * Checks that the idx fits in 32 bits.
     *
     * @param dis  Distance value (lower is better)
     * @param idx  64-bit index (must fit in 32 bits)
     * @return true if threshold was updated
     */
    bool add_result(float dis, idx_t idx) override {
        FAISS_THROW_IF_NOT_MSG(idx >> 32 == 0, "Index does not fit in 32 bits");

        // Early threshold check: reject if dis >= threshold
        if (dis >= threshold) {
            return false;
        }

        if (n == capacity) {
            compact();
            if (dis >= threshold) {
                return false;
            }
        }

        float old_threshold = threshold;
        vals[n] = dis;
        idxs[n] = static_cast<int32_t>(idx);
        n++;

        return threshold != old_threshold;
    }

    /**
     * Finalize results: sort the reservoir and copy to output arrays.
     * Results are sorted with smallest distances first.
     *
     * @param heap_dis  Output array for distances (size k)
     * @param heap_ids  Output array for ids (size k)
     */
    void end(float* heap_dis, idx_t* heap_ids) {
        // Sort the reservoir (ascending order)
        if (n > 0) {
            argsort(n, vals.data(), idxs.data());
        }

        // Copy results to output (up to k elements)
        size_t n_results = std::min(n, k);
        for (size_t i = 0; i < n_results; i++) {
            heap_dis[i] = vals[i];
            heap_ids[i] = static_cast<idx_t>(idxs[i]);
        }

        // Fill remaining slots with neutral values
        for (size_t i = n_results; i < k; i++) {
            heap_dis[i] = HUGE_VALF;
            heap_ids[i] = -1;
        }
    }
};

} // namespace faiss
