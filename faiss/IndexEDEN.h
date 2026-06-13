/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/EDENQuantizer.h>

namespace faiss {

struct IndexEDEN : IndexFlatCodes {
    EDENQuantizer eden;

    // Center of all training points. Random rotation, if desired, is expected
    // to be supplied externally via IndexPreTransform.
    std::vector<float> center;

    IndexEDEN();

    // Factory strings: EDEN, EDEN<n>, EDENBIASED, EDEN<n>BIASED.
    // <n> is the Lloyd-Max bit budget, 1..8. The default is EDEN's original
    // unbiased scale; the BIASED suffix selects the MSE-minimizing scale.
    explicit IndexEDEN(
            idx_t d,
            MetricType metric = METRIC_L2,
            uint8_t nb_bits = 1,
            EDENScaleType scale_type = EDENScaleType_UNBIASED);

    void train(idx_t n, const float* x) override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss
