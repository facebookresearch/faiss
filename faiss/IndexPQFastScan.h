/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFastScan.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

/** Fast scan version of IndexPQ. Works for 4-bit PQ for now.
 *
 * The codes are not stored sequentially but grouped in blocks of size bbs.
 * This makes it possible to compute distances quickly with SIMD instructions.
 *
 * Implementations:
 * 12: blocked loop with internal loop on Q with qbs
 * 13: same with reservoir accumulator to store results
 * 14: no qbs with heap accumulator
 * 15: no qbs with reservoir accumulator
 */

struct IndexPQFastScan : IndexFastScan {
    ProductQuantizer pq;

    IndexPQFastScan(
            int d,
            size_t M,
            size_t nbits,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexPQFastScan() = default;

    /// build from an existing IndexPQ
    explicit IndexPQFastScan(const IndexPQ& orig, int bbs = 32);

    void train(idx_t n, const float* x) override;

    void compute_codes(uint8_t* codes, idx_t n, const float* x) const override;

    void compute_float_LUT(float* lut, idx_t n, const float* x) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

} // namespace faiss
