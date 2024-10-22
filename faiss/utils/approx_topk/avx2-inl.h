/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <immintrin.h>

#include <limits>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>

namespace faiss {

template <typename C, uint32_t NBUCKETS, uint32_t N>
struct HeapWithBuckets {
    // this case was not implemented yet.
};

template <uint32_t NBUCKETS, uint32_t N>
struct HeapWithBuckets<CMax<float, int>, NBUCKETS, N> {
    static constexpr uint32_t NBUCKETS_8 = NBUCKETS / 8;
    static_assert(
            (NBUCKETS) > 0 && ((NBUCKETS % 8) == 0),
            "Number of buckets needs to be 8, 16, 24, ...");

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
            __m256 min_distances_i[NBUCKETS_8][N];
            __m256i min_indices_i[NBUCKETS_8][N];

            for (uint32_t j = 0; j < NBUCKETS_8; j++) {
                for (uint32_t p = 0; p < N; p++) {
                    min_distances_i[j][p] =
                            _mm256_set1_ps(std::numeric_limits<float>::max());
                    min_indices_i[j][p] =
                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                }
            }

            __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i indices_delta = _mm256_set1_epi32(NBUCKETS);

            const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;

            // put the data into buckets
            for (uint32_t ip = 0; ip < nb; ip += NBUCKETS) {
                for (uint32_t j = 0; j < NBUCKETS_8; j++) {
                    const __m256 distances_reg = _mm256_loadu_ps(
                            distances + j * 8 + ip + n_per_beam * beam_index);

                    // loop. Compiler should get rid of unneeded ops
                    __m256 distance_candidate = distances_reg;
                    __m256i indices_candidate = current_indices;

                    for (uint32_t p = 0; p < N; p++) {
                        const __m256 comparison = _mm256_cmp_ps(
                                min_distances_i[j][p],
                                distance_candidate,
                                _CMP_LE_OS);

                        // // blend seems to be slower that min
                        // const __m256 min_distances_new = _mm256_blendv_ps(
                        //         distance_candidate,
                        //         min_distances_i[j][p],
                        //         comparison);
                        const __m256 min_distances_new = _mm256_min_ps(
                                distance_candidate, min_distances_i[j][p]);
                        const __m256i min_indices_new =
                                _mm256_castps_si256(_mm256_blendv_ps(
                                        _mm256_castsi256_ps(indices_candidate),
                                        _mm256_castsi256_ps(
                                                min_indices_i[j][p]),
                                        comparison));

                        // // blend seems to be slower that min
                        // const __m256 max_distances_new = _mm256_blendv_ps(
                        //         min_distances_i[j][p],
                        //         distance_candidate,
                        //         comparison);
                        const __m256 max_distances_new = _mm256_max_ps(
                                min_distances_i[j][p], distances_reg);
                        const __m256i max_indices_new =
                                _mm256_castps_si256(_mm256_blendv_ps(
                                        _mm256_castsi256_ps(
                                                min_indices_i[j][p]),
                                        _mm256_castsi256_ps(indices_candidate),
                                        comparison));

                        distance_candidate = max_distances_new;
                        indices_candidate = max_indices_new;

                        min_distances_i[j][p] = min_distances_new;
                        min_indices_i[j][p] = min_indices_new;
                    }
                }

                current_indices =
                        _mm256_add_epi32(current_indices, indices_delta);
            }

            // fix the indices
            for (uint32_t j = 0; j < NBUCKETS_8; j++) {
                const __m256i offset =
                        _mm256_set1_epi32(n_per_beam * beam_index + j * 8);
                for (uint32_t p = 0; p < N; p++) {
                    min_indices_i[j][p] =
                            _mm256_add_epi32(min_indices_i[j][p], offset);
                }
            }

            // merge every bucket into the regular heap
            for (uint32_t p = 0; p < N; p++) {
                for (uint32_t j = 0; j < NBUCKETS_8; j++) {
                    int32_t min_indices_scalar[8];
                    float min_distances_scalar[8];

                    _mm256_storeu_si256(
                            (__m256i*)min_indices_scalar, min_indices_i[j][p]);
                    _mm256_storeu_ps(
                            min_distances_scalar, min_distances_i[j][p]);

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
