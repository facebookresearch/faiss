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

struct IndexSVSLeanVec4x4 : IndexSVS {
    using blocked_alloc_type =
            svs::data::Blocked<svs::lib::Allocator<std::byte>>;

    using storage_type = svs::leanvec::LeanDataset<
            svs::leanvec::UsingLVQ<4>,
            svs::leanvec::UsingLVQ<4>,
            svs::Dynamic,
            svs::Dynamic,
            blocked_alloc_type>;

    using leanvec_matrix_type = svs::data::SimpleData<float>;

    IndexSVSLeanVec4x4(
            idx_t d,
            MetricType metric = METRIC_L2,
            size_t num_threads = 32,
            size_t graph_max_degree = 64,
            size_t leanvec_dims = 0);

    ~IndexSVSLeanVec4x4() override;

    void reset() override;

    void train(idx_t n, const float* x) override;

    void init_impl(idx_t n, const float* x) override;

    size_t leanvec_d;

    leanvec_matrix_type* leanvec_matrix;
};

} // namespace faiss
