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

#include <memory>
#include <span>

namespace faiss {

IndexSVSVamanaLeanVec::IndexSVSVamanaLeanVec() : IndexSVSVamana() {
    is_trained = false;
}

IndexSVSVamanaLeanVec::IndexSVSVamanaLeanVec(
        idx_t d,
        size_t degree,
        MetricType metric,
        size_t leanvec_dims,
        LeanVecLevel leanvec_level)
        : IndexSVSVamana(d, degree, metric), leanvec_level{leanvec_level} {
    leanvec_d = leanvec_dims == 0 ? d / 2 : leanvec_dims;
    is_trained = false;
}

void IndexSVSVamanaLeanVec::train(idx_t n, const float* x) {
    if (!impl) {
        create_impl();
    }
    auto limpl = leanvec_impl();
    FAISS_THROW_IF_NOT(!limpl->is_trained());
    auto status = limpl->train(static_cast<size_t>(n), x);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
    is_trained = limpl->is_trained();
}

void IndexSVSVamanaLeanVec::reset() {
    is_trained = false;
    IndexSVSVamana::reset();
}

void IndexSVSVamanaLeanVec::deserialize_impl(std::istream& in) {
    IndexSVSVamana::deserialize_impl(in);
    auto limpl = leanvec_impl();
    is_trained = limpl->is_trained();
}

void IndexSVSVamanaLeanVec::create_impl() {
    FAISS_THROW_IF_NOT(!impl);
    ntotal = 0;

    auto svs_metric = to_svs_metric(metric_type);
    svs::faiss_bind::IndexSVSVamanaImpl::BuildParams build_params;
    build_params.storage_kind = storage_kind;
    build_params.graph_max_degree = graph_max_degree;
    build_params.prune_to = prune_to;
    build_params.alpha = alpha;
    build_params.construction_window_size = construction_window_size;
    build_params.max_candidate_pool_size = max_candidate_pool_size;
    auto limpl = svs::faiss_bind::IndexSVSVamanaLeanVecImpl::build(
            d,
            svs_metric,
            build_params,
            leanvec_d,
            static_cast<
                    svs::faiss_bind::IndexSVSVamanaLeanVecImpl::LeanVecLevel>(
                    leanvec_level));
    FAISS_THROW_IF_NOT(limpl);
    impl = limpl;
    is_trained = limpl->is_trained();
}

svs::faiss_bind::IndexSVSVamanaLeanVecImpl* IndexSVSVamanaLeanVec::
        leanvec_impl() const {
    auto limpl =
            dynamic_cast<svs::faiss_bind::IndexSVSVamanaLeanVecImpl*>(impl);
    FAISS_ASSERT(limpl != nullptr);
    return limpl;
}

} // namespace faiss
