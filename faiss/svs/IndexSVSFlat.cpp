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

#include <faiss/Index.h>
#include <faiss/svs/IndexSVSFaissUtils.h>
#include <faiss/svs/IndexSVSFlat.h>

#include <svs/runtime/flat_index.h>

#include <iostream>

namespace faiss {

IndexSVSFlat::IndexSVSFlat(idx_t d, MetricType metric) : Index(d, metric) {}

IndexSVSFlat::~IndexSVSFlat() {
    if (impl) {
        auto status = svs_runtime::FlatIndex::destroy(impl);
        FAISS_ASSERT(status.ok());
        impl = nullptr;
    }
}

void IndexSVSFlat::add(idx_t n, const float* x) {
    if (!impl) {
        create_impl();
    }

    auto status = impl->add(n, x);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    ntotal += n;
}

void IndexSVSFlat::reset() {
    if (impl) {
        auto status = impl->reset();
        if (!status.ok()) {
            FAISS_THROW_MSG(status.message());
        }
    }
    ntotal = 0;
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
        FAISS_THROW_MSG(status.message());
    }
}

/* Initializes the implementation*/
void IndexSVSFlat::create_impl() {
    FAISS_ASSERT(impl == nullptr);
    auto svs_metric = to_svs_metric(metric_type);
    auto status = svs_runtime::FlatIndex::build(&impl, d, svs_metric);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT(impl);
}

/* Serialization */
void IndexSVSFlat::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    auto status = impl->save(out);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

void IndexSVSFlat::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(impl, "Cannot deserialize: SVS index already loaded.");
    auto metric = to_svs_metric(metric_type);
    auto status = impl->load(&impl, in, metric);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT(impl);
}

} // namespace faiss
