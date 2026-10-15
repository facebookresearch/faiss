/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCodes.h>

namespace faiss {

/** Flat index that stores each vector as row-wise int8: d int8 codes plus
 * one float32 scale per vector.
 *
 *    s = max_j |x_j| / 127   (s = 1 for an all-zero vector)
 *    c_j = round(x_j / s), in [-127, 127]
 *    x ~ c * s
 *
 * code layout: [c_0 .. c_{d-1}][s, 4 bytes], code_size = d + 4.
 *
 * Search is exhaustive and exact on the int8 codes: queries are encoded the
 * same way and
 *
 *    q . b       = s_q s_b (c_q . c_b)
 *    |q - b|^2   = s_q^2 |c_q|^2 + s_b^2 |c_b|^2 - 2 q . b
 *
 * where c_q . c_b is an exact int32 sum (AVX-512 VNNI when the CPU has it).
 * Supports METRIC_L2 and METRIC_INNER_PRODUCT.
 */
struct IndexRowwiseInt8 : IndexFlatCodes {
    explicit IndexRowwiseInt8(idx_t d, MetricType metric = METRIC_L2);

    IndexRowwiseInt8();

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /// add vectors that are already row-wise int8: codes (n, d), scales (n)
    void add_int8(idx_t n, const int8_t* codes, const float* scales);

    /// copy out the codes (n, d) and scales (n) of vectors i0 .. i0 + n - 1
    void get_int8(idx_t i0, idx_t n, int8_t* codes, float* scales) const;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// search with queries that are already row-wise int8: codes (n, d),
    /// scales (n)
    void search_int8(
            idx_t n,
            const int8_t* codes,
            const float* scales,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;
};

/// row-wise int8 encoding of n vectors of dimension d (also used on the GPU
/// side to check results)
void rowwise_int8_encode(
        size_t d,
        size_t n,
        const float* x,
        int8_t* codes,
        float* scales);

} // namespace faiss
