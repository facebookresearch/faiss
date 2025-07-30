
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/IndexSVSLVQ4x4.h>

namespace faiss {

struct IndexSVSLVQ4x8 : IndexSVSLVQ4x4 {
    using blocked_alloc_type =
            svs::data::Blocked<svs::lib::Allocator<std::byte>>;

    using strategy_type = svs::quantization::lvq::Turbo<16, 8>;

    using storage_type = svs::quantization::lvq::
            LVQDataset<4, 8, svs::Dynamic, strategy_type, blocked_alloc_type>;

    IndexSVSLVQ4x8() = default;

    IndexSVSLVQ4x8(idx_t d, MetricType metric = METRIC_L2);
};

} // namespace faiss
