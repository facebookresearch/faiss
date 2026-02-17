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

#include <faiss/impl/svs_io.h>

#include <algorithm>
#include <cstdlib>

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

ReaderStreambuf::ReaderStreambuf(IOReader* rr) : r(rr), single_char_buffer(0) {
    // Initialize with empty get area
    setg(nullptr, nullptr, nullptr);
}

ReaderStreambuf::~ReaderStreambuf() = default;

std::streambuf::int_type ReaderStreambuf::underflow() {
    // Called by std::istream for single-character operations (get, peek, etc.)
    // when the get area is exhausted. Reads one byte from IOReader.
    size_t got = (*r)(&single_char_buffer, 1, 1);
    if (got == 0) {
        return traits_type::eof();
    }

    // Configure get area to expose the single buffered character
    setg(&single_char_buffer, &single_char_buffer, &single_char_buffer + 1);
    return traits_type::to_int_type(single_char_buffer);
}

std::streamsize ReaderStreambuf::xsgetn(char* s, std::streamsize n) {
    // Called by std::istream for bulk reads (read, readsome, etc.).
    // Forwards directly to IOReader without intermediate buffering to avoid
    // advancing IOReader beyond what the stream consumer requested.
    if (n <= 0) {
        return 0;
    }

    size_t got = (*r)(s, 1, n);
    return static_cast<std::streamsize>(got);
}

} // namespace svs_io
} // namespace faiss
