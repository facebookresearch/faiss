// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <faiss/cppcontrib/knowhere/IndexWrapper.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

IndexWrapper::IndexWrapper(Index* underlying_index)
        : Index{underlying_index->d, underlying_index->metric_type},
          index{underlying_index} {
    ntotal = underlying_index->ntotal;
    is_trained = underlying_index->is_trained;
    verbose = underlying_index->verbose;
    metric_arg = underlying_index->metric_arg;
}

IndexWrapper::~IndexWrapper() {}

void IndexWrapper::train(idx_t n, const float* x) {
    index->train(n, x);
    is_trained = index->is_trained;
}

void IndexWrapper::add(idx_t n, const float* x) {
    index->add(n, x);
    this->ntotal = index->ntotal;
}

void IndexWrapper::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    index->search(n, x, k, distances, labels, params);
}

void IndexWrapper::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    index->range_search(n, x, radius, result, params);
}

void IndexWrapper::reset() {
    index->reset();
    this->ntotal = 0;
}

void IndexWrapper::merge_from(Index& otherIndex, idx_t add_id) {
    index->merge_from(otherIndex, add_id);
}

DistanceComputer* IndexWrapper::get_distance_computer() const {
    return index->get_distance_computer();
}

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
