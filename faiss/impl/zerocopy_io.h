/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <faiss/impl/io.h>

namespace faiss {

// ZeroCopyIOReader just maps the data from a given pointer.
struct ZeroCopyIOReader : public faiss::IOReader {
    const uint8_t* data_;
    size_t rp_ = 0;
    size_t total_ = 0;

    ZeroCopyIOReader(const uint8_t* data, size_t size);
    ~ZeroCopyIOReader();

    void reset();
    size_t get_data_view(void** ptr, size_t size, size_t nitems);
    size_t operator()(void* ptr, size_t size, size_t nitems) override;

    int filedescriptor() override;
};

} // namespace faiss
