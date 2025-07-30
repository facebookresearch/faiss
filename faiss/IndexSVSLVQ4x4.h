/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexSVS.h>

#include "svs/extensions/vamana/lvq.h"

namespace faiss {

// LVQ 4x0
// LVQ 4x4
// LVQ 4x8
//
//
// allocator
// threadpool

struct IndexSVSLVQ4x4 : IndexSVS {
    using blocked_alloc_type =
            svs::data::Blocked<svs::lib::Allocator<std::byte>>;

    using strategy_type = svs::quantization::lvq::Turbo<16, 8>;

    using storage_type = svs::quantization::lvq::
            LVQDataset<4, 4, svs::Dynamic, strategy_type, blocked_alloc_type>;

    IndexSVSLVQ4x4() = default;
    IndexSVSLVQ4x4(idx_t d, MetricType metric = METRIC_L2);

    ~IndexSVSLVQ4x4() override;

    void init_impl(idx_t n, const float* x) override;

    void deserialize_impl(std::istream& in) override;
};

} // namespace faiss
