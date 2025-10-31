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

#pragma once

#include <faiss/svs/IndexSVSVamana.h>
#include <svs/runtime/IndexSVSVamanaLeanVecImpl.h>

namespace faiss {

// Enum redefinition to avoid including IndexSVSVamanaLeanVecImpl.h in swigfaiss
enum LeanVecLevel {
    LeanVec4x4 = svs::runtime::IndexSVSVamanaLeanVecImpl::LeanVec4x4,
    LeanVec4x8 = svs::runtime::IndexSVSVamanaLeanVecImpl::LeanVec4x8,
    LeanVec8x8 = svs::runtime::IndexSVSVamanaLeanVecImpl::LeanVec8x8
};

struct IndexSVSVamanaLeanVec : IndexSVSVamana {
    IndexSVSVamanaLeanVec();

    IndexSVSVamanaLeanVec(
            idx_t d,
            size_t degree,
            MetricType metric = METRIC_L2,
            size_t leanvec_dims = 0,
            LeanVecLevel leanvec_level = LeanVecLevel::LeanVec4x4);

    ~IndexSVSVamanaLeanVec() override = default;

    void reset() override;

    void train(idx_t n, const float* x) override;

    size_t leanvec_d;

    LeanVecLevel leanvec_level;

    void deserialize_impl(std::istream& in) override;

   protected:
    void create_impl() override;
    svs::runtime::IndexSVSVamanaLeanVecImpl* leanvec_impl() const;
};

} // namespace faiss
