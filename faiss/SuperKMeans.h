/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SuperKMeans — a faster k-means alternative to faiss::Clustering for IVF
// training and other large-k workloads.
//
// Based on:
//   Kuffo, L., Hepkema, S., & Boncz, P. (2026).
//   "A Super Fast K-means for Indexing Vector Embeddings."
//   arXiv preprint arXiv:2603.20009.
//
// Use when: L2 metric, k >= 1024, d >= 128, dense float embeddings.
// Do not use for: IP/cosine (use Clustering with cp.spherical=true), small k,
// binary data (use IndexBinaryIVF), or near-unit-sphere embeddings with
// k < 4096 (chi-squared assumption breaks down).
//
// `cp`, `d`, `k` are public mutable; mutating them post-construction bypasses
// validation and may produce undefined behavior.

#pragma once

#include <vector>

#include <faiss/Clustering.h>
#include <faiss/Index.h>

namespace faiss {

struct SuperKMeansParameters : public ClusteringParameters {
    /// Initial d_prime as a fraction of d (GEMM/PRUNING split).
    float d_prime_fraction = 0.125f;

    /// Matches block_l2 SIMD width.
    int pdx_block_size = 64;

    /// ADSampling significance: epsilon = ad_epsilon_factor / d.
    float ad_epsilon_factor = 1.0f;

    /// Adaptive d_prime stay-in-band controller. Per iteration:
    ///   pruning > high → shrink d_prime  (over-pruning, cheaper threshold)
    ///   pruning < low  → grow   d_prime  (under-pruning, more GEMM work)
    ///   else: hold.
    float pruning_target_low = 0.95f;
    float pruning_target_high = 0.97f;

    /// Relative step size for d_prime adjustments.
    float d_prime_adjust = 0.20f;

    /// Floor on d_prime; below this the chi-squared bound is unvalidated.
    int d_prime_min = 16;

    int x_batch = 4096;
    int y_batch = 1024;

    /// OpenMP dynamic-schedule chunk size for the pruning loop.
    int omp_chunk = 8;
};

/** Drop-in faster k-means: same interface as faiss::Clustering. Per iteration:
 *   iter 0:           full GEMM over all d dims (vanilla Lloyd's).
 *   iter 1..niter-1:  GEMM over front d_prime dims, then ADSampling
 *                     progressive pruning over PDX-laid trailing dims.
 *
 * Trains in a randomly-rotated space; centroids are un-rotated before return.
 */
struct SuperKMeans {
    SuperKMeansParameters cp;
    int d;
    int k;

    /// Output: k * d floats, row-major, un-rotated.
    std::vector<float> centroids;

    /// Per-iter stats. `obj`, `time`, and `nsplit` are populated faithfully.
    /// `time_search` mirrors `time` (phases not timed separately) and
    /// `imbalance_factor` is set to NaN (not computed).
    std::vector<ClusteringIterationStats> iteration_stats;

    /// Per-iter fraction of (vector, centroid) pairs pruned at the d_prime
    /// GEMM-boundary chi-squared check. Per-PDX-block early-exits are NOT
    /// counted. iter 0 uses full GEMM, so gemm_pruning_rates[0] == 0.0f.
    std::vector<float> gemm_pruning_rates;

    SuperKMeans(int d, int k, const SuperKMeansParameters& cp = {});

    /// Train on `n` row-major vectors of dimension `d`. Honors the applicable
    /// ClusteringParameters fields (niter, seed, verbose,
    /// max_points_per_centroid, use_faster_subsampling,
    /// check_input_data_for_NaNs).
    void train(idx_t n, const float* x);
};

} // namespace faiss
