/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "faiss/Index.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <streambuf>
#include <utility>
#include <vector>

namespace faiss {
struct IOReader;
struct IOWriter;
} // namespace faiss

namespace svs {
class DynamicVamana;
}

namespace faiss {
struct IOWriter;
}

namespace faiss {

struct IndexSVS : Index {
    size_t num_threads = 1;
    size_t graph_max_degree = 64;
    float alpha = 1.2;
    size_t search_window_size = 10;
    size_t search_buffer_capacity = 10;
    size_t construction_window_size = 40;
    size_t max_candidate_pool_size = 200;
    size_t prune_to = 60;
    bool use_full_search_history = true;

    IndexSVS();

    IndexSVS(idx_t d, MetricType metric = METRIC_L2);

    virtual ~IndexSVS() override;

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    /* Serialization and deserialization helpers */
    void serialize_impl(std::ostream& out) const;
    virtual void deserialize_impl(std::istream& in);

    /* The actual SVS implementation */
    svs::DynamicVamana* impl{nullptr};

    /* Initializes the implementation, using the provided data */
    virtual void init_impl(idx_t n, const float* x);
};

// We provide some helpers for efficient I/O in the svs_io namespace
// These can be excluded from the Python bindings
namespace svs_io {
// Bridges IOWriter to std::ostream (used for streaming payload out)
struct WriterStreambuf : std::streambuf {
    IOWriter* w; // not owning
    explicit WriterStreambuf(IOWriter* w_);
    ~WriterStreambuf() override; // out-of-line def in .cpp
   protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override;
    int overflow(int ch) override;
};

// Bridges IOReader to std::istream (used to read payload to EOF)
struct ReaderStreambuf : std::streambuf {
    IOReader* r;           // not owning
    std::vector<char> buf; // ring buffer (default 1 MiB)
    explicit ReaderStreambuf(IOReader* rr, size_t bsz = (1u << 20));
    ~ReaderStreambuf() override; // out-of-line def in .cpp
   protected:
    int_type underflow() override;
};

/* temporary directory for SVS Vamana indices that tries to always clean up */
struct SVSTempDirectory {
    std::filesystem::path root;
    std::filesystem::path config;
    std::filesystem::path graph;
    std::filesystem::path data;

    SVSTempDirectory();
    ~SVSTempDirectory();

    void write_files_to_stream(std::ostream& out) const;
    void write_stream_to_files(std::istream& in) const;
};
} // namespace svs_io

} // namespace faiss
