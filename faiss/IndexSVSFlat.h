/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <numeric>
#include <variant>
#include <vector>

#include "faiss/Index.h"

#include "svs/core/distance.h"

namespace svs {
class Flat;
}

namespace faiss {

struct IndexSVSFlat : Index {
    // sequential labels
    size_t nlabels{0};
    size_t num_threads = 1;

    IndexSVSFlat() = default;
    IndexSVSFlat(idx_t d, MetricType metric = METRIC_L2);

    ~IndexSVSFlat();

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    /* The actual SVS implementation */
    svs::Flat* impl{nullptr};

    /* Initializes the implementation, using the provided data */
    virtual void init_impl(idx_t n, const float* x);
};

} // namespace faiss
