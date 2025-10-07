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

} // namespace svs_io
} // namespace faiss
