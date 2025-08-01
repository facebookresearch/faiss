/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexSVS.h>

#include "svs/extensions/vamana/leanvec.h"

namespace faiss {

enum LeanVecLevel { LeanVec_4x4, LeanVec_4x8, LeanVec_8x8 };

struct IndexSVSLeanVec : IndexSVS {
    using blocked_alloc_type =
            svs::data::Blocked<svs::lib::Allocator<std::byte>>;
    using storage_type_4x4 = svs::leanvec::LeanDataset<
            svs::leanvec::UsingLVQ<4>,
            svs::leanvec::UsingLVQ<4>,
            svs::Dynamic,
            svs::Dynamic,
            blocked_alloc_type>;
    using storage_type_4x8 = svs::leanvec::LeanDataset<
            svs::leanvec::UsingLVQ<4>,
            svs::leanvec::UsingLVQ<8>,
            svs::Dynamic,
            svs::Dynamic,
            blocked_alloc_type>;
    using storage_type_8x8 = svs::leanvec::LeanDataset<
            svs::leanvec::UsingLVQ<8>,
            svs::leanvec::UsingLVQ<8>,
            svs::Dynamic,
            svs::Dynamic,
            blocked_alloc_type>;

    IndexSVSLeanVec() = default;

    IndexSVSLeanVec(
            idx_t d,
            MetricType metric = METRIC_L2,
            size_t leanvec_dims = 0,
            LeanVecLevel leanvec_level = LeanVecLevel::LeanVec_4x4);

    ~IndexSVSLeanVec() override = default;

    void reset() override;

    void train(idx_t n, const float* x) override;

    void init_impl(idx_t n, const float* x) override;

    size_t leanvec_d;

    LeanVecLevel leanvec_level;

    svs::leanvec::LeanVecMatrices<svs::Dynamic> leanvec_matrix;

    void deserialize_impl(std::istream& in) override;
};

} // namespace faiss
