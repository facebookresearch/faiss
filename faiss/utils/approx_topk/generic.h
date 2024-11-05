/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <limits>
#include <utility>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>

namespace faiss {

// This is the implementation of the idea and it is very slow,
// because a compiler is unable to vectorize it properly.

template <typename C, uint32_t NBUCKETS, uint32_t N>
struct HeapWithBuckets {
    // this case was not implemented yet.
};

template <uint32_t NBUCKETS, uint32_t N>
struct HeapWithBuckets<CMax<float, int>, NBUCKETS, N> {
    static void addn(
            // number of elements
            const uint32_t n,
            // distances. It is assumed to have n elements.
            const float* const __restrict distances,
            // number of best elements to keep
            const uint32_t k,
            // output distances
            float* const __restrict bh_val,
            // output indices, each being within [0, n) range
            int32_t* const __restrict bh_ids) {
        // forward a call to bs_addn with 1 beam
        bs_addn(1, n, distances, k, bh_val, bh_ids);
    }

    static void bs_addn(
            // beam_size parameter of Beam Search algorithm
            const uint32_t beam_size,
            // number of elements per beam
            const uint32_t n_per_beam,
            // distances. It is assumed to have (n_per_beam * beam_size)
            // elements.
            const float* const __restrict distances,
            // number of best elements to keep
            const uint32_t k,
            // output distances
            float* const __restrict bh_val,
            // output indices, each being within [0, n_per_beam * beam_size)
            // range
            int32_t* const __restrict bh_ids) {
        // // Basically, the function runs beam_size iterations.
        // // Every iteration NBUCKETS * N elements are added to a regular heap.
        // // So, maximum number of added elements is beam_size * NBUCKETS * N.
        // // This number is expected to be less or equal than k.
        // FAISS_THROW_IF_NOT_FMT(
        //         beam_size * NBUCKETS * N >= k,
        //         "Cannot pick %d elements, only %d. "
        //         "Check the function and template arguments values.",
        //         k,
        //         beam_size * NBUCKETS * N);

        using C = CMax<float, int>;

        // main loop
        for (uint32_t beam_index = 0; beam_index < beam_size; beam_index++) {
            float min_distances_i[N][NBUCKETS];
            int min_indices_i[N][NBUCKETS];

            for (uint32_t p = 0; p < N; p++) {
                for (uint32_t j = 0; j < NBUCKETS; j++) {
                    min_distances_i[p][j] = std::numeric_limits<float>::max();
                    min_indices_i[p][j] = 0;
                }
            }

            const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;

            // put the data into buckets
            for (uint32_t ip = 0; ip < nb; ip += NBUCKETS) {
                for (uint32_t j = 0; j < NBUCKETS; j++) {
                    const int index = j + ip + n_per_beam * beam_index;
                    const float distance = distances[index];

                    int index_candidate = index;
                    float distance_candidate = distance;

                    for (uint32_t p = 0; p < N; p++) {
                        if (distance_candidate < min_distances_i[p][j]) {
                            std::swap(
                                    distance_candidate, min_distances_i[p][j]);
                            std::swap(index_candidate, min_indices_i[p][j]);
                        }
                    }
                }
            }

            // merge every bucket into the regular heap
            for (uint32_t p = 0; p < N; p++) {
                for (uint32_t j = 0; j < NBUCKETS; j++) {
                    // this exact way is needed to maintain the order as if the
                    // input elements were pushed to the heap sequentially

                    if (C::cmp2(bh_val[0],
                                min_distances_i[p][j],
                                bh_ids[0],
                                min_indices_i[p][j])) {
                        heap_replace_top<C>(
                                k,
                                bh_val,
                                bh_ids,
                                min_distances_i[p][j],
                                min_indices_i[p][j]);
                    }
                }
            }

            // process leftovers
            for (uint32_t ip = nb; ip < n_per_beam; ip++) {
                const int32_t index = ip + n_per_beam * beam_index;
                const float value = distances[index];

                if (C::cmp(bh_val[0], value)) {
                    heap_replace_top<C>(k, bh_val, bh_ids, value, index);
                }
            }
        }
    }
};

} // namespace faiss
