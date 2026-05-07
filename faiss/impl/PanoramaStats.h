/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_PANORAMA_STATS_H
#define FAISS_PANORAMA_STATS_H

#include <faiss/impl/platform_macros.h>

namespace faiss {

/// Statistics are not robust to internal threading nor to
/// concurrent Panorama searches. Use these values in a
/// single-threaded context to accurately gauge Panorama's
/// pruning effectiveness.
struct PanoramaStats {
    uint64_t total_dims_scanned = 0; // total dimensions scanned
    uint64_t total_dims = 0;         // total dimensions
    float ratio_dims_scanned = 1.0f; // fraction of dimensions actually scanned

    /// Per-level survivor counters (HNSW Panorama search): how many
    /// candidates entered level 0, how many survived past level 0, and
    /// how many got the full L2 distance computed (i.e. survived all
    /// levels). Lets us measure pruning rate per level apples-to-apples
    /// across runs.
    uint64_t n_visited = 0;       // candidates that entered level 0
    uint64_t n_after_level0 = 0;  // survived past level 0
    uint64_t n_full_dist = 0;     // survived all levels (got exact L2)

    PanoramaStats() {
        reset();
    }
    void reset();
    void add(const PanoramaStats& other);
};

// Single global var for all Panorama indexes
FAISS_API extern PanoramaStats indexPanorama_stats;

} // namespace faiss

#endif
