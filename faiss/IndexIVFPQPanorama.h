/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_INDEX_IVFPQ_PANORAMA_H
#define FAISS_INDEX_IVFPQ_PANORAMA_H

#include <vector>

#include <faiss/IndexIVFPQ.h>

namespace faiss {

/// Panorama adaptation of IndexIVFPQ following
/// https://www.arxiv.org/pdf/2510.00566.
///
/// IDEA:
/// Panorama adapts the storage layout within each cluster and uses
/// Cauchy-Schwarz pruning to skip unnecessary distance computations.
/// Combined with orthogonal transforms upstream that concentrate signal
/// energy in the early PQ subquantizers (like PCA), Panorama can prune
/// the majority of candidates after computing only a fraction of the
/// full PQ distance.
///
/// STORAGE LAYOUT:
/// Standard IVFPQ stores codes row-major: [point0_code, point1_code, ...].
/// Panorama transposes codes into column-major within each batch:
/// for each batch of `batch_size` points, codes are stored as
/// M columns of `batch_size` bytes each. The M columns are grouped
/// into `n_levels` levels of `chunk_size` columns, enabling incremental
/// distance computation level-by-level.
///
/// Storage is managed by ArrayInvertedListsPanorama with a PanoramaPQ
/// instance that handles code transposition and cumulative sum computation
/// (via PQ decoding) on insertion.
///
/// OVERHEAD:
/// Panorama precomputes per-point cumulative residual norms at insertion
/// time. Storage overhead is (n_levels + 1) floats per point for
/// cum_sums. Initial exact distances are computed on-the-fly during
/// search using the precomputed_table (no extra per-point storage).
///
/// CONSTRAINTS:
/// - Only L2 metric is supported.
/// - Only 8-bit PQ codes (nbits_per_idx == 8).
/// - M must be divisible by n_levels.
/// - batch_size must be a multiple of 64.
/// - use_precomputed_table must be 1.
///
/// NOTE:
/// We inherit from IndexIVFPQ and override only get_InvertedListScanner().
/// The base IndexIVF::search_preassigned() handles all search
/// orchestration — no search code is duplicated.
/// Storage (transposition + cum_sums) is handled by
/// ArrayInvertedListsPanorama, so no add() override is needed.
struct IndexIVFPQPanorama : public IndexIVFPQ {
    int n_levels;
    size_t batch_size;

    size_t chunk_size;
    size_t levels_size;

    IndexIVFPQPanorama(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            int n_levels,
            size_t batch_size = 128,
            MetricType metric = METRIC_L2,
            bool own_invlists = true);

    IndexIVFPQPanorama() = default;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;
};

} // namespace faiss

#endif
