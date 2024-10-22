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

#include <faiss/utils/Heap.h>
#include <faiss/utils/simdlib.h>

namespace faiss {

// HeapWithBucketsForHamming32 uses simd8uint32 under the hood.

template <typename C, uint32_t NBUCKETS, uint32_t N, typename HammingComputerT>
struct HeapWithBucketsForHamming32 {
    // this case was not implemented yet.
};

template <uint32_t NBUCKETS, uint32_t N, typename HammingComputerT>
struct HeapWithBucketsForHamming32<
        CMax<int, int64_t>,
        NBUCKETS,
        N,
        HammingComputerT> {
    static constexpr uint32_t NBUCKETS_8 = NBUCKETS / 8;
    static_assert(
            (NBUCKETS) > 0 && ((NBUCKETS % 8) == 0),
            "Number of buckets needs to be 8, 16, 24, ...");

    static void addn(
            // number of elements
            const uint32_t n,
            // Hamming computer
            const HammingComputerT& hc,
            // n elements that can be used with hc
            const uint8_t* const __restrict binaryVectors,
            // number of best elements to keep
            const uint32_t k,
            // output distances
            int* const __restrict bh_val,
            // output indices, each being within [0, n) range
            int64_t* const __restrict bh_ids) {
        // forward a call to bs_addn with 1 beam
        bs_addn(1, n, hc, binaryVectors, k, bh_val, bh_ids);
    }

    static void bs_addn(
            // beam_size parameter of Beam Search algorithm
            const uint32_t beam_size,
            // number of elements per beam
            const uint32_t n_per_beam,
            // Hamming computer
            const HammingComputerT& hc,
            // n elements that can be used against hc
            const uint8_t* const __restrict binary_vectors,
            // number of best elements to keep
            const uint32_t k,
            // output distances
            int* const __restrict bh_val,
            // output indices, each being within [0, n_per_beam * beam_size)
            // range
            int64_t* const __restrict bh_ids) {
        //
        using C = CMax<int, int64_t>;

        // Hamming code size
        const size_t code_size = hc.get_code_size();

        // main loop
        for (uint32_t beam_index = 0; beam_index < beam_size; beam_index++) {
            simd8uint32 min_distances_i[NBUCKETS_8][N];
            simd8uint32 min_indices_i[NBUCKETS_8][N];

            for (uint32_t j = 0; j < NBUCKETS_8; j++) {
                for (uint32_t p = 0; p < N; p++) {
                    min_distances_i[j][p] =
                            simd8uint32(std::numeric_limits<int32_t>::max());
                    min_indices_i[j][p] = simd8uint32(0, 1, 2, 3, 4, 5, 6, 7);
                }
            }

            simd8uint32 current_indices(0, 1, 2, 3, 4, 5, 6, 7);
            const simd8uint32 indices_delta(NBUCKETS);

            const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;

            // put the data into buckets
            for (uint32_t ip = 0; ip < nb; ip += NBUCKETS) {
                for (uint32_t j = 0; j < NBUCKETS_8; j++) {
                    uint32_t hamming_distances[8];
                    for (size_t j8 = 0; j8 < 8; j8++) {
                        hamming_distances[j8] = hc.hamming(
                                binary_vectors +
                                (j8 + j * 8 + ip + n_per_beam * beam_index) *
                                        code_size);
                    }

                    // loop. Compiler should get rid of unneeded ops
                    simd8uint32 distance_candidate;
                    distance_candidate.loadu(hamming_distances);
                    simd8uint32 indices_candidate = current_indices;

                    for (uint32_t p = 0; p < N; p++) {
                        simd8uint32 min_distances_new;
                        simd8uint32 min_indices_new;
                        simd8uint32 max_distances_new;
                        simd8uint32 max_indices_new;

                        faiss::cmplt_min_max_fast(
                                distance_candidate,
                                indices_candidate,
                                min_distances_i[j][p],
                                min_indices_i[j][p],
                                min_distances_new,
                                min_indices_new,
                                max_distances_new,
                                max_indices_new);

                        distance_candidate = max_distances_new;
                        indices_candidate = max_indices_new;

                        min_distances_i[j][p] = min_distances_new;
                        min_indices_i[j][p] = min_indices_new;
                    }
                }

                current_indices += indices_delta;
            }

            // fix the indices
            for (uint32_t j = 0; j < NBUCKETS_8; j++) {
                const simd8uint32 offset(n_per_beam * beam_index + j * 8);
                for (uint32_t p = 0; p < N; p++) {
                    min_indices_i[j][p] += offset;
                }
            }

            // merge every bucket into the regular heap
            for (uint32_t p = 0; p < N; p++) {
                for (uint32_t j = 0; j < NBUCKETS_8; j++) {
                    uint32_t min_indices_scalar[8];
                    uint32_t min_distances_scalar[8];

                    min_indices_i[j][p].storeu(min_indices_scalar);
                    min_distances_i[j][p].storeu(min_distances_scalar);

                    // this exact way is needed to maintain the order as if the
                    // input elements were pushed to the heap sequentially
                    for (size_t j8 = 0; j8 < 8; j8++) {
                        const auto value = min_distances_scalar[j8];
                        const auto index = min_indices_scalar[j8];

                        if (C::cmp2(bh_val[0], value, bh_ids[0], index)) {
                            heap_replace_top<C>(
                                    k, bh_val, bh_ids, value, index);
                        }
                    }
                }
            }

            // process leftovers
            for (uint32_t ip = nb; ip < n_per_beam; ip++) {
                const auto index = ip + n_per_beam * beam_index;
                const auto value =
                        hc.hamming(binary_vectors + (index)*code_size);

                if (C::cmp(bh_val[0], value)) {
                    heap_replace_top<C>(k, bh_val, bh_ids, value, index);
                }
            }
        }
    }
};

// HeapWithBucketsForHamming16 uses simd16uint16 under the hood.
// Less registers needed in total, so higher values of NBUCKETS/N can be used,
//   but somewhat slower.
// No more than 32K elements currently, but it can be reorganized a bit
//   to be limited to 32K elements per beam.

template <typename C, uint32_t NBUCKETS, uint32_t N, typename HammingComputerT>
struct HeapWithBucketsForHamming16 {
    // this case was not implemented yet.
};

template <uint32_t NBUCKETS, uint32_t N, typename HammingComputerT>
struct HeapWithBucketsForHamming16<
        CMax<int, int64_t>,
        NBUCKETS,
        N,
        HammingComputerT> {
    static constexpr uint32_t NBUCKETS_16 = NBUCKETS / 16;
    static_assert(
            (NBUCKETS) > 0 && ((NBUCKETS % 16) == 0),
            "Number of buckets needs to be 16, 32, 48...");

    static void addn(
            // number of elements
            const uint32_t n,
            // Hamming computer
            const HammingComputerT& hc,
            // n elements that can be used with hc
            const uint8_t* const __restrict binaryVectors,
            // number of best elements to keep
            const uint32_t k,
            // output distances
            int* const __restrict bh_val,
            // output indices, each being within [0, n) range
            int64_t* const __restrict bh_ids) {
        // forward a call to bs_addn with 1 beam
        bs_addn(1, n, hc, binaryVectors, k, bh_val, bh_ids);
    }

    static void bs_addn(
            // beam_size parameter of Beam Search algorithm
            const uint32_t beam_size,
            // number of elements per beam
            const uint32_t n_per_beam,
            // Hamming computer
            const HammingComputerT& hc,
            // n elements that can be used against hc
            const uint8_t* const __restrict binary_vectors,
            // number of best elements to keep
            const uint32_t k,
            // output distances
            int* const __restrict bh_val,
            // output indices, each being within [0, n_per_beam * beam_size)
            // range
            int64_t* const __restrict bh_ids) {
        //
        using C = CMax<int, int64_t>;

        // Hamming code size
        const size_t code_size = hc.get_code_size();

        // main loop
        for (uint32_t beam_index = 0; beam_index < beam_size; beam_index++) {
            simd16uint16 min_distances_i[NBUCKETS_16][N];
            simd16uint16 min_indices_i[NBUCKETS_16][N];

            for (uint32_t j = 0; j < NBUCKETS_16; j++) {
                for (uint32_t p = 0; p < N; p++) {
                    min_distances_i[j][p] =
                            simd16uint16(std::numeric_limits<int16_t>::max());
                    min_indices_i[j][p] = simd16uint16(
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15);
                }
            }

            simd16uint16 current_indices(
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            const simd16uint16 indices_delta((uint16_t)NBUCKETS);

            const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;

            // put the data into buckets
            for (uint32_t ip = 0; ip < nb; ip += NBUCKETS) {
                for (uint32_t j = 0; j < NBUCKETS_16; j++) {
                    uint16_t hamming_distances[16];
                    for (size_t j16 = 0; j16 < 16; j16++) {
                        hamming_distances[j16] = hc.hamming(
                                binary_vectors +
                                (j16 + j * 16 + ip + n_per_beam * beam_index) *
                                        code_size);
                    }

                    // loop. Compiler should get rid of unneeded ops
                    simd16uint16 distance_candidate;
                    distance_candidate.loadu(hamming_distances);
                    simd16uint16 indices_candidate = current_indices;

                    for (uint32_t p = 0; p < N; p++) {
                        simd16uint16 min_distances_new;
                        simd16uint16 min_indices_new;
                        simd16uint16 max_distances_new;
                        simd16uint16 max_indices_new;

                        faiss::cmplt_min_max_fast(
                                distance_candidate,
                                indices_candidate,
                                min_distances_i[j][p],
                                min_indices_i[j][p],
                                min_distances_new,
                                min_indices_new,
                                max_distances_new,
                                max_indices_new);

                        distance_candidate = max_distances_new;
                        indices_candidate = max_indices_new;

                        min_distances_i[j][p] = min_distances_new;
                        min_indices_i[j][p] = min_indices_new;
                    }
                }

                current_indices += indices_delta;
            }

            // fix the indices
            for (uint32_t j = 0; j < NBUCKETS_16; j++) {
                const simd16uint16 offset(
                        (uint16_t)(n_per_beam * beam_index + j * 16));
                for (uint32_t p = 0; p < N; p++) {
                    min_indices_i[j][p] += offset;
                }
            }

            // merge every bucket into the regular heap
            for (uint32_t p = 0; p < N; p++) {
                for (uint32_t j = 0; j < NBUCKETS_16; j++) {
                    uint16_t min_indices_scalar[16];
                    uint16_t min_distances_scalar[16];

                    min_indices_i[j][p].storeu(min_indices_scalar);
                    min_distances_i[j][p].storeu(min_distances_scalar);

                    // this exact way is needed to maintain the order as if the
                    // input elements were pushed to the heap sequentially
                    for (size_t j16 = 0; j16 < 16; j16++) {
                        const auto value = min_distances_scalar[j16];
                        const auto index = min_indices_scalar[j16];

                        if (C::cmp2(bh_val[0], value, bh_ids[0], index)) {
                            heap_replace_top<C>(
                                    k, bh_val, bh_ids, value, index);
                        }
                    }
                }
            }

            // process leftovers
            for (uint32_t ip = nb; ip < n_per_beam; ip++) {
                const auto index = ip + n_per_beam * beam_index;
                const auto value =
                        hc.hamming(binary_vectors + (index)*code_size);

                if (C::cmp(bh_val[0], value)) {
                    heap_replace_top<C>(k, bh_val, bh_ids, value, index);
                }
            }
        }
    }
};

} // namespace faiss
