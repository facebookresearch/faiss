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

#include <svs/runtime/api_defs.h>

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/mapped_io.h>

#include <algorithm>
#include <concepts>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

// validate FAISS_SVS_RUNTIME_VERSION is set
#ifndef FAISS_SVS_RUNTIME_VERSION
#error "FAISS_SVS_RUNTIME_VERSION is not defined"
#endif
// create svs_runtime as alias for svs::runtime::FAISS_SVS_RUNTIME_VERSION
SVS_RUNTIME_CREATE_API_ALIAS(svs_runtime, FAISS_SVS_RUNTIME_VERSION);

namespace faiss {

inline svs_runtime::MetricType to_svs_metric(faiss::MetricType metric) {
    switch (metric) {
        case METRIC_INNER_PRODUCT:
            return svs_runtime::MetricType::INNER_PRODUCT;
        case METRIC_L2:
            return svs_runtime::MetricType::L2;
        default:
            FAISS_THROW_MSG("not supported SVS distance");
    }
}

struct FaissIDFilter : public svs_runtime::IDFilter {
    FaissIDFilter(const faiss::IDSelector& sel) : selector(sel) {}

    bool is_member(size_t id) const override {
        return selector.is_member(static_cast<faiss::idx_t>(id));
    }

   private:
    const faiss::IDSelector& selector;
};

inline std::unique_ptr<FaissIDFilter> make_faiss_id_filter(
        const SearchParameters* params = nullptr) {
    if (params && params->sel) {
        return std::make_unique<FaissIDFilter>(*params->sel);
    }
    return nullptr;
}

template <typename T, typename U, typename = void>
struct InputBufferConverter {
    InputBufferConverter(std::span<const U> data = {}) : buffer(data.size()) {
        FAISS_ASSERT(
                false &&
                "InputBufferConverter: there is no suitable user code for this type conversion");
        std::transform(
                data.begin(), data.end(), buffer.begin(), [](const U& val) {
                    return static_cast<T>(val);
                });
    }

    operator T*() {
        return buffer.data();
    }
    operator const T*() const {
        return buffer.data();
    }

    operator std::span<T>() {
        return buffer;
    }
    operator std::span<const T>() const {
        return buffer;
    }

   private:
    std::vector<T> buffer;
};

// Specialization for reinterpret cast when types are integral and have
// the same size
template <typename T, typename U>
struct InputBufferConverter<
        T,
        U,
        std::enable_if_t<
                std::is_same_v<T, U> ||
                (std::is_integral_v<T> && std::is_integral_v<U> &&
                 sizeof(T) == sizeof(U))>> {
    InputBufferConverter(std::span<const U> data = {}) : data_span(data) {}
    operator T*() {
        return reinterpret_cast<T*>(data_span.data());
    }
    operator const T*() const {
        return reinterpret_cast<const T*>(data_span.data());
    }
    operator std::span<T>() {
        return std::span<T>(
                reinterpret_cast<T*>(data_span.data()), data_span.size());
    }
    operator std::span<const T>() const {
        return std::span<const T>(
                reinterpret_cast<const T*>(data_span.data()), data_span.size());
    }

   private:
    std::span<const U> data_span;
};

template <typename T, typename U, typename = void>
struct OutputBufferConverter {
    OutputBufferConverter(std::span<U> data = {})
            : data_span(data), buffer(data.size()) {
        FAISS_ASSERT(
                false &&
                "OutputBufferConverter: there is no suitable user code for this type conversion");
    }

    ~OutputBufferConverter() {
        std::transform(
                buffer.begin(),
                buffer.end(),
                data_span.begin(),
                [](const T& val) { return static_cast<U>(val); });
    }

    operator T*() {
        return buffer.data();
    }
    operator std::span<T>() {
        return buffer;
    }

   private:
    std::span<U> data_span;
    std::vector<T> buffer;
};

// Specialization for reinterpret cast when types are integral and have
// the same size
template <typename T, typename U>
struct OutputBufferConverter<
        T,
        U,
        std::enable_if_t<
                std::is_same_v<T, U> ||
                (std::is_integral_v<T> && std::is_integral_v<U> &&
                 sizeof(T) == sizeof(U))>> {
    OutputBufferConverter(std::span<U> data = {}) : data_span(data) {}
    operator T*() {
        return reinterpret_cast<T*>(data_span.data());
    }
    operator std::span<T>() {
        return std::span<T>(
                reinterpret_cast<T*>(data_span.data()), data_span.size());
    }

   private:
    std::span<U> data_span;
};

template <typename T, typename U>
auto convert_input_buffer(std::span<const U> data) {
    // Create temporary buffer and convert input data
    // to target type T in the temporary buffer
    // The temporary buffer will be destroyed
    // when going out of scope
    return InputBufferConverter<T, U>(data);
}

template <typename T, typename U>
auto convert_input_buffer(const U* data, size_t size) {
    return convert_input_buffer<T, U>(std::span<const U>(data, size));
}

// Output buffer conversion
template <typename T, typename U>
auto convert_output_buffer(std::span<U> data) {
    // Create temporary buffer for output data
    // The temporary buffer will be destroyed
    // when going out of scope, copying back
    // the converted data to original buffer
    return OutputBufferConverter<T, U>(data);
}

template <typename T, typename U>
auto convert_output_buffer(U* data, size_t size) {
    return convert_output_buffer<T, U>(std::span<U>(data, size));
}

struct FaissResultsAllocator : public svs_runtime::ResultsAllocator {
    FaissResultsAllocator(faiss::RangeSearchResult* result) : result(result) {
        FAISS_ASSERT(result != nullptr);
    }

    svs_runtime::SearchResultsStorage allocate(
            std::span<size_t> result_counts) const override {
        FAISS_ASSERT(result != nullptr);
        FAISS_ASSERT(result_counts.size() == result->nq);

        // RangeSearchResult .ctor() allows unallocated lims
        if (result->lims == nullptr) {
            result->lims = new size_t[result->nq + 1];
        }

        std::copy(result_counts.begin(), result_counts.end(), result->lims);
        result->do_allocation();
        this->labels_converter = LabelsConverter{
                std::span(result->labels, result->lims[result_counts.size()])};
        return svs_runtime::SearchResultsStorage{
                labels_converter,
                std::span<float>(
                        result->distances, result->lims[result_counts.size()])};
    }

   private:
    faiss::RangeSearchResult* result;
    using LabelsConverter = OutputBufferConverter<size_t, faiss::idx_t>;
    mutable LabelsConverter labels_converter;
};

// Helper for memory-mapped SVS index loading.
// Acquires a pointer to the remaining mapped region and returns the data
// pointer, size in bytes, and starting position. Validates that mmap returns
// expected byte count. MappedFileIOReader::mmap returns number of items (not
// bytes); we always request size=1 so items == bytes, but we name variables
// explicitly to avoid confusion.
struct MmapSpan {
    void* data = nullptr;
    size_t size_bytes = 0;
    size_t start_pos = 0;
};

inline MmapSpan acquire_mmap_span(MappedFileIOReader* mf) {
    FAISS_THROW_IF_NOT(mf);
    FAISS_THROW_IF_NOT(mf->mmap_owner);
    size_t pos = mf->pos;
    size_t size_to_end = mf->mmap_owner->size() - pos;
    // Reject an empty span: MappedFileIOReader::mmap() returns 0 without
    // writing *ptr when no bytes remain, which would otherwise pass the
    // size check below and hand a nullptr / zero size to the SVS runtime
    // (e.g. for a truncated or corrupt file).
    FAISS_THROW_IF_NOT_FMT(
            size_to_end > 0,
            "acquire_mmap_span: no mapped bytes remain at reader position %zu",
            pos);
    void* data = nullptr;
    // mmap returns actual_nitems; with size=1 this equals bytes
    size_t actual_nitems = mf->mmap(&data, 1, size_to_end);
    FAISS_THROW_IF_NOT_FMT(
            actual_nitems == size_to_end,
            "mmap() returned unexpected size: %zu items (expected %zu bytes)",
            actual_nitems,
            size_to_end);
    return {data, size_to_end, pos};
}

// Adjusts MappedFileIOReader position after SVS consumes read_bytes,
// storing mmap_owner reference to keep mapping alive.
inline void finalize_mmap_span(
        MappedFileIOReader* mf,
        const MmapSpan& span,
        size_t read_bytes,
        std::shared_ptr<MmappedFileMappingOwner>& mmap_owner_out) {
    // Take ownership of the mapping BEFORE validating read_bytes. By this point
    // the caller's impl already holds pointers into the mapped region, so the
    // index must keep the mapping alive even on the throw path. Otherwise the
    // index would be left with a live impl but no mmap_owner, and once the
    // reader's mapping reference is released the index destructor (which calls
    // destroy(impl)) would touch unmapped memory.
    mmap_owner_out = mf->mmap_owner;
    FAISS_THROW_IF_NOT_FMT(
            read_bytes > 0 && read_bytes <= span.size_bytes,
            "map_to_memory returned invalid read_bytes: %zu (span size %zu)",
            read_bytes,
            span.size_bytes);
    mf->pos = span.start_pos + read_bytes;
}

} // namespace faiss
