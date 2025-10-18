/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/PanoramaStats.h>

#include <cstring>

namespace faiss {

void PanoramaStats::reset() {
    memset((void*)this, 0, sizeof(*this));
}

void PanoramaStats::add(const PanoramaStats& other) {
    total_dims_scanned += other.total_dims_scanned;
    total_dims += other.total_dims;
    pct_dims_scanned = static_cast<float>(total_dims_scanned) / total_dims;
}

PanoramaStats indexPanorama_stats;

} // namespace faiss
