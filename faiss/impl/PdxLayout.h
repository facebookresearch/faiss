/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace faiss {
namespace detail {

/** Reorder a row-major (k, d_trail) matrix into PDX block-column-major
 * layout. Inside each block of `pdx_block_size` dims the layout is
 * column-major across centroids, so all k centroids' values for the same
 * dim are contiguous — the access pattern that makes progressive pruning
 * cache-friendly. Trailing block (size `d_trail % pdx_block_size`) uses
 * the same convention. `Y_pdx` must already be sized to `k * d_trail`. */
void pdxify(
        const float* Y,
        int k,
        int d_trail,
        int pdx_block_size,
        float* Y_pdx);

/** Inverse of pdxify (used in tests for the bit-identical round-trip
 * check). */
void de_pdxify(
        const float* Y_pdx,
        int k,
        int d_trail,
        int pdx_block_size,
        float* Y);

/** norms[i] = sum_{m<p} X[i, m]^2  for row-major X of shape (n, d).
 * Parallel over rows. Used by SuperKMeans to keep partial-norm caches
 * in sync with the current d_prime. */
void compute_partial_norms(const float* X, int n, int d, int p, float* norms);

} // namespace detail
} // namespace faiss
