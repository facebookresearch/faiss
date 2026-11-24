/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Rank-based 2-way split of an oversized coarse bucket.
 *
 * The bucket members are ordered by their projection score `diff` (ties and
 * non-finite scores broken by vector id) and cut at the median. A band of
 * `overlap` (fraction of the bucket, centred on the cut) is copied into both
 * halves so neighbors across the cut stay connected. Because the cut is by
 * rank, each half holds at most ceil(nb * (1 + overlap) / 2) members whatever
 * the geometry: skewed clusters, ties and fully duplicated vectors all shrink
 * the bucket, and degenerate scores fall back to an even split by id.
 *
 * Header-only and free of SIMD intrinsics so it can be unit-tested directly.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace faiss {
namespace rabitq_batch {

/// Splits `bucket` (member ids, any order) into `out0` (high-`diff` side) and
/// `out1` (low-`diff` side). Both outputs keep the members' original order in
/// `bucket`. `diff[i]` is the score of `bucket[i]`.
inline void split_bucket_by_rank(
        const std::vector<uint32_t>& bucket,
        const std::vector<float>& diff,
        double overlap,
        std::vector<uint32_t>& out0,
        std::vector<uint32_t>& out1) {
    const size_t nb = bucket.size();
    out0.clear();
    out1.clear();
    if (nb == 0) {
        return;
    }
    // Non-finite scores carry no geometric information: treat them as 0 so the
    // comparator stays a strict weak ordering.
    std::vector<float> key(nb);
    for (size_t i = 0; i < nb; ++i) {
        key[i] = std::isfinite(diff[i]) ? diff[i] : 0.0f;
    }
    std::vector<uint32_t> ord(nb);
    std::iota(ord.begin(), ord.end(), uint32_t(0));
    std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
        if (key[a] != key[b]) {
            return key[a] < key[b];
        }
        return bucket[a] < bucket[b];
    });

    const size_t half = nb / 2;
    const size_t band = std::min(
            half, size_t(double(nb) * std::clamp(overlap, 0.0, 1.0) / 2.0));
    const size_t lo = half - band;               // out0 = ord[lo, nb)
    const size_t hi = std::min(nb, half + band); // out1 = ord[0, hi)

    // side bit 1: in out0, bit 2: in out1 (positions in `bucket` order).
    std::vector<uint8_t> side(nb, 0);
    for (size_t r = 0; r < nb; ++r) {
        uint8_t s = 0;
        if (r >= lo)
            s |= 1;
        if (r < hi)
            s |= 2;
        side[ord[r]] = s;
    }
    out0.reserve(nb - lo);
    out1.reserve(hi);
    for (size_t i = 0; i < nb; ++i) {
        if (side[i] & 1)
            out0.push_back(bucket[i]);
        if (side[i] & 2)
            out1.push_back(bucket[i]);
    }
}

} // namespace rabitq_batch
} // namespace faiss
