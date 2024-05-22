/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>
#include <unordered_set>
#include <vector>

#include <faiss/MetricType.h>

/** IDGrouper is intended to define a group of vectors to include only
 * the nearest vector of each group during search */

namespace faiss {

/** Encapsulates a group id of ids */
struct IDGrouper {
    const idx_t NO_MORE_DOCS = std::numeric_limits<idx_t>::max();
    virtual idx_t get_group(idx_t id) const = 0;
    virtual ~IDGrouper() {}
};

/** One bit per element. Constructed with a bitmap, size ceil(n / 8).
 */
struct IDGrouperBitmap : IDGrouper {
    // length of the bitmap array
    size_t n;

    // Array of uint64_t holding the bits
    // Using uint64_t to leverage function __builtin_ctzll which is defined in
    // faiss/impl/platform_macros.h Group id of a given id is next set bit in
    // the bitmap
    uint64_t* bitmap;

    /** Construct with a binary mask
     *
     * @param n size of the bitmap array
     * @param bitmap group id of a given id is next set bit in the bitmap
     */
    IDGrouperBitmap(size_t n, uint64_t* bitmap);
    idx_t get_group(idx_t id) const final;
    void set_group(idx_t group_id);
    ~IDGrouperBitmap() override {}
};

} // namespace faiss
