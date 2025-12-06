/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2025 Intel Corporation
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

namespace faiss {

struct IndexSVSVamanaLeanVec : IndexSVSVamana {
    IndexSVSVamanaLeanVec();

    IndexSVSVamanaLeanVec(
            idx_t d,
            size_t degree,
            MetricType metric = METRIC_L2,
            size_t leanvec_dims = 0,
            SVSStorageKind storage = SVSStorageKind::SVS_LeanVec4x4);

    ~IndexSVSVamanaLeanVec() override;

    void add(idx_t n, const float* x) override;

    void train(idx_t n, const float* x) override;

    void serialize_training_data(std::ostream& out) const;
    void deserialize_training_data(std::istream& in);

    size_t leanvec_d;

    /* Training information */
    svs_runtime::LeanVecTrainingData* training_data{nullptr};

   protected:
    void create_impl() override;
};

} // namespace faiss
