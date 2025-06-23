/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/RaBitQuantizer.h>

namespace faiss {

struct RaBitQSearchParameters : SearchParameters {
    uint8_t qb = 0;
};

struct IndexRaBitQ : IndexFlatCodes {
    RaBitQuantizer rabitq;

    // center of all points
    std::vector<float> center;

    // the default number of bits to quantize a query with.
    // use '0' to disable quantization and use raw fp32 values.
    uint8_t qb = 0;

    IndexRaBitQ();

    IndexRaBitQ(idx_t d, MetricType metric = METRIC_L2);

    void train(idx_t n, const float* x) override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    // returns a quantized-to-qb bits DC if qb > 0
    // returns a default fp32-based DC if qb == 0
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    // returns a quantized-to-qb bits DC if qb_in > 0
    // returns a default fp32-based DC if qb_in == 0
    FlatCodesDistanceComputer* get_quantized_distance_computer(
            const uint8_t qb_in) const;

    // Don't rely on sa_decode(), bcz it is good for IP, but not for L2.
    //   As a result, use get_FlatCodesDistanceComputer() for the search.
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
