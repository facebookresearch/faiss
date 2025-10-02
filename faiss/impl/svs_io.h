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

#pragma once

#include <filesystem>
#include <iostream>
#include <streambuf>
#include <vector>

#include <faiss/impl/io.h>

namespace faiss {
namespace svs_io {

// Bridges IOWriter to std::ostream (used for streaming payload out)
struct WriterStreambuf : std::streambuf {
    IOWriter* w;
    explicit WriterStreambuf(IOWriter* w_);
    ~WriterStreambuf() override;

   protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override;
    int overflow(int ch) override;
};

// Bridges IOReader to std::istream (used to read payload to EOF)
struct ReaderStreambuf : std::streambuf {
    IOReader* r;
    std::vector<char> buf;
    explicit ReaderStreambuf(IOReader* rr);
    ~ReaderStreambuf() override;

   protected:
    int_type underflow() override;
};

/* temporary directory for SVS indices that tries to always clean up */
struct SVSTempDirectory {
    std::filesystem::path root;
    std::filesystem::path config;
    std::filesystem::path graph;
    std::filesystem::path data;

    SVSTempDirectory();
    ~SVSTempDirectory();

    /* Write files written by SVS to a stream owned by FAISS for index_write */
    void write_files_to_stream(std::ostream& out) const;
    /* Write a stream owned by FAISS to SVS-compatible files for index_read */
    void write_stream_to_files(std::istream& in) const;
};

} // namespace svs_io
} // namespace faiss
