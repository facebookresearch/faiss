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
#include <faiss/svs/IndexSVSFlat.h>

#include <iostream>

#include <faiss/Index.h>

namespace faiss {

IndexSVSFlat::IndexSVSFlat(idx_t d, MetricType metric) : Index(d, metric) {}

void IndexSVSFlat::add(idx_t n, const float* x) {
    if (!impl) {
        create_impl();
    }

    auto status = impl->add(n, x);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
    ntotal += n;
}

void IndexSVSFlat::reset() {
    if (impl) {
        impl->reset();
    }
    ntotal = 0;
}

IndexSVSFlat::~IndexSVSFlat() {
    svs::runtime::IndexSVSFlatImpl::destroy(impl);
    impl = nullptr;
}

void IndexSVSFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(impl);
    auto status = impl->search(
            n,
            x,
            static_cast<size_t>(k),
            distances,
            reinterpret_cast<size_t*>(labels));
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
}

/* Initializes the implementation*/
void IndexSVSFlat::create_impl() {
    FAISS_ASSERT(impl == nullptr);
    auto svs_metric = to_svs_metric(metric_type);
    impl = svs::runtime::IndexSVSFlatImpl::build(d, svs_metric);
    FAISS_THROW_IF_NOT(impl);
}

/* Serialization */
void IndexSVSFlat::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    auto status = impl->serialize(out);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
}

void IndexSVSFlat::deserialize_impl(std::istream& in) {
    if (!impl) {
        create_impl();
    }
    auto status = impl->deserialize(in);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
}

} // namespace faiss
