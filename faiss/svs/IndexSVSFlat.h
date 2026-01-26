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

#include <faiss/Index.h>
#include <faiss/svs/IndexSVSFaissUtils.h>

#include <iostream>

namespace faiss {

struct IndexSVSFlat : Index {
    // sequential labels
    size_t nlabels{0};

    IndexSVSFlat() = default;
    IndexSVSFlat(idx_t d, MetricType metric = METRIC_L2);

    ~IndexSVSFlat() override;

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    /* The actual SVS implementation */
    svs_runtime::FlatIndex* impl{nullptr};

    /* Serialization */
    void serialize_impl(std::ostream& out) const;
    void deserialize_impl(std::istream& in);

   protected:
    /* Initializes the implementation*/
    virtual void create_impl();
};

} // namespace faiss
