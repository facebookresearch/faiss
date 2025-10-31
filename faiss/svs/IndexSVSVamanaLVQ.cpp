/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <faiss/svs/IndexSVSFaissUtils.h>
#include <faiss/svs/IndexSVSVamanaLVQ.h>

namespace faiss {

IndexSVSVamanaLVQ::IndexSVSVamanaLVQ(
        idx_t d,
        size_t degree,
        MetricType metric,
        LVQLevel lvq_level)
        : IndexSVSVamana(d, degree, metric), lvq_level{lvq_level} {}

void IndexSVSVamanaLVQ::create_impl() {
    FAISS_THROW_IF_NOT(!impl);
    ntotal = 0;

    auto svs_metric = to_svs_metric(metric_type);
    svs::runtime::IndexSVSVamanaImpl::BuildParams build_params;
    build_params.storage_kind = storage_kind;
    build_params.graph_max_degree = graph_max_degree;
    build_params.prune_to = prune_to;
    build_params.alpha = alpha;
    build_params.construction_window_size = construction_window_size;
    build_params.max_candidate_pool_size = max_candidate_pool_size;
    impl = svs::runtime::IndexSVSVamanaLVQImpl::build(
            d,
            svs_metric,
            build_params,
            static_cast<svs::runtime::IndexSVSVamanaLVQImpl::LVQLevel>(
                    lvq_level));
    FAISS_THROW_IF_NOT(impl);
}

} // namespace faiss
