/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/PanoramaStats.h>

namespace faiss {

void PanoramaStats::reset() {
    total_dims_scanned = 0;
    total_dims = 0;
    ratio_dims_scanned = 1.0f;
    n_visited = 0;
    n_after_level0 = 0;
    n_full_dist = 0;
}

void PanoramaStats::add(const PanoramaStats& other) {
    total_dims_scanned += other.total_dims_scanned;
    total_dims += other.total_dims;
    if (total_dims > 0) {
        ratio_dims_scanned =
                static_cast<float>(total_dims_scanned) / total_dims;
    } else {
        ratio_dims_scanned = 1.0f;
    }
    n_visited += other.n_visited;
    n_after_level0 += other.n_after_level0;
    n_full_dist += other.n_full_dist;
}

PanoramaStats indexPanorama_stats;

} // namespace faiss
