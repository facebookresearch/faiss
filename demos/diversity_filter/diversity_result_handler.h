/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexHNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/random.h>

using TheC = faiss::CMax<float, int64_t>;

/** There is a group id associated to each vector id. We want no more than
 * max_per_group results per group but of course we also want to keep the best
 * results for each group.
 * This implementation is a stateless handler that always keeps results sorted,
 * which means updates are O(k) */
struct DiversityResultHandlerBubble : faiss::ResultHandlerT<TheC> {
    using group_t = int;

    // records the group for each database vector
    const std::vector<group_t>& id_to_group;

    // we want no more than this many results per group
    size_t max_per_group;

    // the distances and labels of the results
    size_t K; // number of results to keep
    float* distances;
    int64_t* labels;

    // found so far
    size_t nresults = 0;

    DiversityResultHandlerBubble(
            const std::vector<group_t>& id_to_group,
            size_t max_per_group,
            size_t K,
            float* distances,
            int64_t* labels)
            : id_to_group(id_to_group),
              max_per_group(max_per_group),
              K(K),
              distances(distances),
              labels(labels) {}

    /// find where the element distance should be inserted in the sorted array
    /// distances
    size_t bissection(float distance) {
        size_t lo = 0, hi = nresults;
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (distances[mid] < distance) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /// shift results one step to the right in [i0, i1), ie. i1 is overwritten
    void shift_results(size_t i0, size_t i1) {
        if (i0 >= i1) {
            return;
        }
        memmove(&distances[i0 + 1], &distances[i0], (i1 - i0) * sizeof(float));
        memmove(&labels[i0 + 1], &labels[i0], (i1 - i0) * sizeof(int64_t));
    }

    virtual bool add_result(float distance, faiss::idx_t i) final {
        group_t group = id_to_group[i];
        if (nresults == K && distance >= distances[K - 1]) {
            return false;
        }
        // count number of occurrences of this group and keep track of the worst
        // result of this group
        int n = 0;
        size_t worst_i = 0;
        for (int j = 0; j < nresults; j++) {
            if (id_to_group[labels[j]] == group) {
                n++;
                worst_i = j; // is the worst since sorted
            }
        }
        if (n >= max_per_group && distance >= distances[worst_i]) {
            return false;
        }
        // we can add the new result
        size_t insertion_point = bissection(distance);
        if (n >= max_per_group) {
            // replace worst_i with new result
            // shift elements between insertion_point and worst_i
            if (insertion_point < worst_i) {
                shift_results(insertion_point, worst_i);
            }
        } else if (nresults < (int)K) {
            // add new result
            shift_results(insertion_point, nresults);
            nresults++;
        } else {
            // replace the last result
            shift_results(insertion_point, K - 1);
        }
        distances[insertion_point] = distance;
        labels[insertion_point] = i;
        return true;
    }
};

/** TODO implement a version with a double heap: the first heap keeps track of
 * the worst result of each group, K other heaps keep track of the results of
 * each group. This would ensure log(k) update time. */
