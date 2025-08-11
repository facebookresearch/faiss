/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "faiss/Index.h"
#include "faiss/impl/FaissAssert.h"

#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <filesystem>
#include <streambuf>

#ifdef _WIN32
#include <io.h> // _lseeki64, _write
#else
#include <unistd.h> // lseek
#endif
#include <cstdint>

namespace svs {
class DynamicVamana;
}

namespace faiss {
struct IOWriter;
}

namespace faiss {

namespace svs_io {

/* helpers to get and seek positions in fd for back-patching blob size */
static inline bool fd_cur_pos(int fd, uint64_t& pos) {
#ifdef _WIN32
    __int64 p = _lseeki64(fd, 0, SEEK_CUR);
    if (p < 0)
        return false;
    pos = (uint64_t)p;
    return true;
#else
    off_t p = ::lseek(fd, 0, SEEK_CUR);
    if (p < 0)
        return false;
    pos = (uint64_t)p;
    return true;
#endif
}

static inline bool fd_seek(int fd, uint64_t pos) {
#ifdef _WIN32
    return _lseeki64(fd, (__int64)pos, SEEK_SET) >= 0;
#else
    return ::lseek(fd, (off_t)pos, SEEK_SET) >= 0;
#endif
}

struct WriterStreambuf : std::streambuf {
    faiss::IOWriter* w;
    uint64_t written = 0;
    explicit WriterStreambuf(faiss::IOWriter* w_);
    std::streamsize xsputn(const char* s, std::streamsize n) override;
    int overflow(int ch) override;
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

} // namespace faiss
