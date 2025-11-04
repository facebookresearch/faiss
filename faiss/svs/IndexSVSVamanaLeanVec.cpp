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
#include <faiss/svs/IndexSVSVamanaLeanVec.h>
#include <svs/runtime/IndexSVSVamanaLeanVecImpl.h>

#include <memory>
#include <span>
#include "faiss/impl/FaissAssert.h"

namespace faiss {

IndexSVSVamanaLeanVec::IndexSVSVamanaLeanVec(
        idx_t d,
        size_t degree,
        MetricType metric,
        size_t leanvec_dims,
        LeanVecLevel leanvec_level)
        : IndexSVSVamana(d, degree, metric), leanvec_level{leanvec_level} {
    leanvec_d = leanvec_dims == 0 ? d / 2 : leanvec_dims;
}

void IndexSVSVamanaLeanVec::add(idx_t n, const float* x) {
    if (state == LeanVecState::EmptyAndUntrained) {
        FAISS_THROW_MSG("Index must be trained before adding data.");
    }

    auto status = svs::runtime::Status_Ok;
    if (!impl) {
        build_impl(n, x);
    } else {
        status = impl->add(n, x);
    }
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
    ntotal += n;
}

void IndexSVSVamanaLeanVec::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            state == LeanVecState::EmptyAndUntrained,
            "Index already trained or contains data.");

    auto* info =
            svs::runtime::IndexSVSVamanaLeanVecImpl::build_leanvec_training(
                    n, x, d, leanvec_d);
    FAISS_THROW_IF_NOT_MSG(info, "Failed to build leanvec training info.");
    training_info = info;
    state = LeanVecState::EmptyAndTrained;
    is_trained = true;
}

void IndexSVSVamanaLeanVec::deserialize_impl(std::istream& in) {
    IndexSVSVamana::deserialize_impl(in);
    auto limpl = leanvec_impl();
    // is_trained = limpl->is_trained();
}

void IndexSVSVamanaLeanVec::build_impl(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            state != LeanVecState::EmptyAndUntrained,
            "Index must be trained before adding data.");
    FAISS_THROW_IF_NOT(!impl); // impl should be null here
    ntotal = 0;

    auto svs_metric = to_svs_metric(metric_type);
    svs::runtime::IndexSVSVamanaImpl::BuildParams build_params;
    build_params.storage_kind = storage_kind;
    build_params.graph_max_degree = graph_max_degree;
    build_params.prune_to = prune_to;
    build_params.alpha = alpha;
    build_params.construction_window_size = construction_window_size;
    build_params.max_candidate_pool_size = max_candidate_pool_size;
    auto limpl = svs::runtime::IndexSVSVamanaLeanVecImpl::build_leanvec(
            d,
            svs_metric,
            build_params,
            leanvec_d,
            static_cast<svs::runtime::IndexSVSVamanaLeanVecImpl::LeanVecLevel>(
                    leanvec_level),
            n,
            x,
            training_info);
    FAISS_THROW_IF_NOT(limpl);
    impl = limpl;
    state = LeanVecState::NonEmptyAndTrained;
}

svs::runtime::IndexSVSVamanaLeanVecImpl* IndexSVSVamanaLeanVec::leanvec_impl()
        const {
    auto limpl = dynamic_cast<svs::runtime::IndexSVSVamanaLeanVecImpl*>(impl);
    FAISS_ASSERT(limpl != nullptr);
    return limpl;
}

} // namespace faiss
