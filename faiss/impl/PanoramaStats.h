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
    uint64_t total_dims_scanned; // total dimensions scanned
    uint64_t total_dims;  	     // total dimensions
    float ratio_dims_scanned;    // fraction of dimensions actually scanned
    
    PanoramaStats() { reset(); }
    void reset();
    void add(const PanoramaStats& other);
};

// Single global var for all Panorama indexes
FAISS_API extern PanoramaStats indexPanorama_stats;

} // namespace faiss

#endif
