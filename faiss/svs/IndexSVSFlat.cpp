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
#include <faiss/impl/mapped_io.h>
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
    mmap_owner.reset(); // Release the memory mapping
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
    auto status = svs_runtime::FlatIndex::load(&impl, in, metric);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT_MSG(impl, "Failed to load SVS Flat index.");
}

void IndexSVSFlat::map_to(MappedFileIOReader* mf) {
    FAISS_THROW_IF_MSG(impl, "Cannot map_to: SVS index already loaded.");
    FAISS_THROW_IF_NOT(mf);

    // Get current position and remaining size
    size_t pos = mf->pos;
    size_t size_to_end = mf->mmap_owner->size() - pos;

    // Get memory-mapped pointer
    void* data = nullptr;
    size_t actual_size = mf->mmap(&data, 1, size_to_end);
    FAISS_THROW_IF_NOT_FMT(
            actual_size == size_to_end,
            "mmap() returned unexpected size: %zu (expected %zu)",
            actual_size,
            size_to_end);

    auto svs_metric = to_svs_metric(metric_type);

    size_t read_bytes = 0;
    auto status = svs_runtime::FlatIndex::map_to_memory(
            &impl, data, actual_size, svs_metric, &read_bytes);

    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }

    FAISS_THROW_IF_NOT_FMT(
            read_bytes > 0 && read_bytes <= size_to_end,
            "VamanaIndex::map_to_memory returned invalid read_bytes: %zu",
            read_bytes);

    // Store the mmap_owner to keep the memory mapping alive for the lifetime
    // of this index. This ensures the memory buffer remains mapped while SVS
    // is using the pointers we obtained from it.
    mmap_owner = mf->mmap_owner;

    // Adjust the position to account for the actual bytes read by SVS.
    // The mmap() call advanced the position by size_to_end, but SVS may have
    // consumed less data. We need to set it to the correct position for
    // subsequent read operations.
    mf->pos = pos + read_bytes;
}

} // namespace faiss
