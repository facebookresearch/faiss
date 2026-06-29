/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/Clustering.h>
#include <faiss/Index.h>

namespace faiss {
namespace detail {

/** Resolve the actual RNG seed for clustering helpers.
 *
 * If `seed >= 0`, returns `seed`. Otherwise returns a high-resolution
 * timestamp so that callers get a non-deterministic seed.
 *
 * @param seed  user-provided seed; negative values request a time-based seed
 * @return      the resolved seed
 */
uint64_t get_actual_rng_seed(const int seed);

/** Subsample a training set down to `clus.k * clus.max_points_per_centroid`
 * rows.
 *
 * Allocates `*x_out` (and `*weights_out` when `weights` is non-null) with
 * `new[]`; ownership is transferred to the caller.
 *
 * @param clus        clustering parameters (reads `k`,
 * `max_points_per_centroid`, `use_faster_subsampling`, `seed`, `verbose`)
 * @param nx          number of input training rows
 * @param x           input training data, row-major, `nx * line_size` bytes
 * @param line_size   bytes per training row
 * @param weights     optional per-row weights (length `nx`), or null
 * @param x_out       output: newly allocated subsampled rows
 * @param weights_out output: newly allocated subsampled weights, or null
 * @return            number of rows in the subsampled set
 */
idx_t subsample_training_set(
        const Clustering& clus,
        idx_t nx,
        const uint8_t* x,
        size_t line_size,
        const float* weights,
        uint8_t** x_out,
        float** weights_out);

/** compute centroids as (weighted) sum of training points
 *
 * @param x            training vectors, size n * code_size (from codec)
 * @param codec        how to decode the vectors (if NULL then cast to float*)
 * @param weights      per-training vector weight, size n (or NULL)
 * @param assign       nearest centroid for each training vector, size n
 * @param k_frozen     do not update the k_frozen first centroids
 * @param centroids    centroid vectors (output only), size k * d
 * @param hassign      histogram of assignments per centroid (size k),
 *                     should be 0 on input
 *
 */
void compute_centroids(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const uint8_t* x,
        const Index* codec,
        const int64_t* assign,
        const float* weights,
        float* hassign,
        float* centroids);

/** Handle empty clusters by splitting larger ones.
 *
 * It works by slightly changing the centroids to make 2 clusters from
 * a single one. Takes the same arguments as compute_centroids.
 *
 * @return           nb of splitting operations (larger is worse)
 */
int split_clusters(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        float* hassign,
        float* centroids);

} // namespace detail
} // namespace faiss
