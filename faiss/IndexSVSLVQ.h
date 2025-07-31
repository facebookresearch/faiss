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

enum LVQLevel {
  LVQ_4x0,
  LVQ_4x4,
  LVQ_4x8
};

struct IndexSVSLVQ : IndexSVS {
    using blocked_alloc_type =
            svs::data::Blocked<svs::lib::Allocator<std::byte>>;

    using strategy_type_4 = svs::quantization::lvq::Turbo<16, 8>;

    using storage_type_4x0 = svs::quantization::lvq::
            LVQDataset<4, 0, svs::Dynamic, strategy_type_4, blocked_alloc_type>;
    using storage_type_4x4 = svs::quantization::lvq::
            LVQDataset<4, 4, svs::Dynamic, strategy_type_4, blocked_alloc_type>;
    using storage_type_4x8 = svs::quantization::lvq::
            LVQDataset<4, 8, svs::Dynamic, strategy_type_4, blocked_alloc_type>;

    IndexSVSLVQ() = default;
    IndexSVSLVQ(idx_t d, MetricType metric = METRIC_L2, LVQLevel lvq_level = LVQLevel::LVQ_4x4);

    ~IndexSVSLVQ() override = default;

    void init_impl(idx_t n, const float* x) override;

    void deserialize_impl(std::istream& in) override;

    LVQLevel lvq_level;
};

} // namespace faiss
