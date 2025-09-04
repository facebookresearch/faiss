/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexSVSVamana.h>

#include <svs/extensions/vamana/lvq.h>
#include <svs/quantization/scalar/scalar.h>

namespace faiss {

enum LVQLevel { LVQ4x0, LVQ4x4, LVQ4x8 };

struct IndexSVSVamanaLVQ : IndexSVSVamana {
    using blocked_alloc_type =
            svs::data::Blocked<svs::lib::Allocator<std::byte>>;
    using blocked_alloc_type_sq =
            svs::data::Blocked<svs::lib::Allocator<std::int8_t>>;

    using strategy_type_4 = svs::quantization::lvq::Turbo<16, 8>;

    using storage_type_4x0 = svs::quantization::lvq::
            LVQDataset<4, 0, svs::Dynamic, strategy_type_4, blocked_alloc_type>;
    using storage_type_4x4 = svs::quantization::lvq::
            LVQDataset<4, 4, svs::Dynamic, strategy_type_4, blocked_alloc_type>;
    using storage_type_4x8 = svs::quantization::lvq::
            LVQDataset<4, 8, svs::Dynamic, strategy_type_4, blocked_alloc_type>;
    using storage_type_sq = svs::quantization::scalar::
            SQDataset<std::int8_t, svs::Dynamic, blocked_alloc_type_sq>;

    IndexSVSVamanaLVQ() = default;
    IndexSVSVamanaLVQ(
            idx_t d,
            size_t degree,
            MetricType metric = METRIC_L2,
            LVQLevel lvq_level = LVQLevel::LVQ4x4);

    ~IndexSVSVamanaLVQ() override = default;

    void init_impl(idx_t n, const float* x) override;

    void deserialize_impl(std::istream& in) override;

    LVQLevel lvq_level;
};

} // namespace faiss
