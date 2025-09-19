/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/svs_io.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>

#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace svs_io {

WriterStreambuf::WriterStreambuf(IOWriter* w_) : w(w_) {}

WriterStreambuf::~WriterStreambuf() = default;

std::streamsize WriterStreambuf::xsputn(const char* s, std::streamsize n) {
    if (n <= 0)
        return 0;
    size_t wrote = (*w)(s, 1, static_cast<size_t>(n));
    return static_cast<std::streamsize>(wrote);
}

int WriterStreambuf::overflow(int ch) {
    if (ch == traits_type::eof())
        return 0;
    char c = static_cast<char>(ch);
    size_t wrote = (*w)(&c, 1, 1);
    return wrote == 1 ? ch : traits_type::eof();
}

ReaderStreambuf::ReaderStreambuf(IOReader* rr)
        : r(rr), buf(1 << 20 /* 1 MiB */) {
    setg(buf.data(), buf.data(), buf.data());
}

ReaderStreambuf::~ReaderStreambuf() = default;

std::streambuf::int_type ReaderStreambuf::underflow() {
    if (gptr() < egptr()) {
        return traits_type::to_int_type(*gptr());
    }
    size_t got = (*r)(buf.data(), 1, buf.size());
    if (got == 0) {
        return traits_type::eof();
    }
    setg(buf.data(), buf.data(), buf.data() + got);
    return traits_type::to_int_type(*gptr());
}

SVSTempDirectory::SVSTempDirectory() {
    root = std::filesystem::temp_directory_path() /
            ("faiss_svs_" + std::to_string(std::rand()));
    config = root / "config";
    graph = root / "graph";
    data = root / "data";

    std::filesystem::create_directories(config);
    std::filesystem::create_directories(graph);
    std::filesystem::create_directories(data);
}

SVSTempDirectory::~SVSTempDirectory() {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

static inline void read_exact(std::istream& in, void* p, size_t n) {
    char* c = static_cast<char*>(p);
    size_t got = 0;
    while (got < n) {
        in.read(c + got, n - got);
        std::streamsize r = in.gcount();
        if (r <= 0) {
            FAISS_THROW_IF_NOT_MSG(false, "Unexpected EOF while reading");
        }
        got += size_t(r);
    }
}

void SVSTempDirectory::write_files_to_stream(std::ostream& out) const {
    // First pass: compute total number of bytes that will be written AFTER
    // the length prefix itself. The format per file is:
    // [uint64_t dir_len][dir_name bytes]
    // [uint64_t file_len][file_name bytes]
    // [uint64_t file_size][file bytes]
    // We sum these for all files in the (config, graph, data) subdirectories.

    uint64_t total_bytes = 0;
    for (const auto& dir : {config, graph, data}) {
        const std::string dir_name = dir.filename().string();
        const uint64_t dir_len = dir_name.size();
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            const std::string filename = entry.path().filename().string();
            const uint64_t file_len = filename.size();
            const uint64_t file_size = std::filesystem::file_size(entry.path());

            total_bytes += sizeof(uint64_t) + dir_len;  // dir_len + dir_name
            total_bytes += sizeof(uint64_t) + file_len; // file_len + filename
            total_bytes +=
                    sizeof(uint64_t) + file_size; // file_size + file contents
        }
    }

    // Write the length prefix (number of bytes that follow).
    out.write(reinterpret_cast<const char*>(&total_bytes), sizeof(total_bytes));

    for (const auto& dir : {config, graph, data}) {
        const std::string dir_name = dir.filename().string();
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            const std::string filename = entry.path().filename().string();

            const uint64_t dir_len = dir_name.size();
            const uint64_t file_len = filename.size();
            const uint64_t file_size = std::filesystem::file_size(entry.path());

            out.write(reinterpret_cast<const char*>(&dir_len), sizeof(dir_len));
            out.write(dir_name.data(), dir_len);

            out.write(
                    reinterpret_cast<const char*>(&file_len), sizeof(file_len));
            out.write(filename.data(), file_len);

            out.write(
                    reinterpret_cast<const char*>(&file_size),
                    sizeof(file_size));

            std::ifstream in(entry.path(), std::ios::binary);
            FAISS_THROW_IF_NOT_MSG(
                    in, "Failed to open temp SVS file for reading");

            out << in.rdbuf();
        }
    }
}

void SVSTempDirectory::write_stream_to_files(std::istream& in) const {
    constexpr size_t BUFSZ = 1 << 20; // 1 MiB
    std::vector<char> buf(BUFSZ);
    // Read the prefixed total byte count produced by write_files_to_stream.
    uint64_t total_bytes = 0;
    read_exact(in, &total_bytes, sizeof(total_bytes));

    uint64_t consumed = 0; // number of payload bytes consumed so far

    while (consumed < total_bytes) {
        // Each record begins with dir_len
        uint64_t dir_len = 0;
        read_exact(in, &dir_len, sizeof(dir_len));
        consumed += sizeof(dir_len);

        std::string dir_name(dir_len, '\0');
        read_exact(in, dir_name.data(), dir_len);
        consumed += dir_len;

        uint64_t file_len = 0;
        read_exact(in, &file_len, sizeof(file_len));
        consumed += sizeof(file_len);
        std::string filename(file_len, '\0');
        read_exact(in, filename.data(), file_len);
        consumed += file_len;

        uint64_t file_size = 0;
        read_exact(in, &file_size, sizeof(file_size));
        consumed += sizeof(file_size);

        // Bounds check before allocating / reading file contents
        if (consumed + file_size > total_bytes) {
            FAISS_THROW_IF_NOT_MSG(
                    false,
                    "Corrupt stream: declared file_size exceeds remaining bytes");
        }

        std::filesystem::path base;
        if (dir_name == "config") {
            base = config;
        } else if (dir_name == "graph") {
            base = graph;
        } else if (dir_name == "data") {
            base = data;
        } else {
            FAISS_THROW_IF_NOT_MSG(false, "Unknown SVS subdirectory name");
        }

        std::filesystem::path full_path = base / filename;
        std::ofstream out(full_path, std::ios::binary);
        FAISS_THROW_IF_NOT_MSG(out, "Failed to open temp SVS file for writing");

        uint64_t remaining = file_size;
        while (remaining > 0) {
            size_t want = size_t(std::min<uint64_t>(remaining, buf.size()));
            read_exact(in, buf.data(), want);
            out.write(buf.data(), want);
            FAISS_THROW_IF_NOT_MSG(out, "Short write to temp SVS file");
            remaining -= want;
            consumed += want;
        }
    }
}

} // namespace svs_io
} // namespace faiss
