/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/zerocopy_io.h>
#include <cstring>

namespace faiss {

ZeroCopyIOReader::ZeroCopyIOReader(const uint8_t* data, size_t size)
        : data_(data), rp_(0), total_(size) {}

ZeroCopyIOReader::~ZeroCopyIOReader() {}

size_t ZeroCopyIOReader::get_data_view(void** ptr, size_t size, size_t nitems) {
    if (size == 0) {
        return nitems;
    }

    size_t actual_size = size * nitems;
    if (rp_ + size * nitems > total_) {
        actual_size = total_ - rp_;
    }

    size_t actual_nitems = (actual_size + size - 1) / size;
    if (actual_nitems == 0) {
        return 0;
    }

    // get an address
    *ptr = (void*)(reinterpret_cast<const char*>(data_ + rp_));

    // alter pos
    rp_ += size * actual_nitems;

    return actual_nitems;
}

void ZeroCopyIOReader::reset() {
    rp_ = 0;
}

size_t ZeroCopyIOReader::operator()(void* ptr, size_t size, size_t nitems) {
    if (size * nitems == 0) {
        return 0;
    }

    if (rp_ >= total_) {
        return 0;
    }
    size_t nremain = (total_ - rp_) / size;
    if (nremain < nitems) {
        nitems = nremain;
    }
    memcpy(ptr, (data_ + rp_), size * nitems);
    rp_ += size * nitems;
    return nitems;
}

int ZeroCopyIOReader::filedescriptor() {
    return -1; // Indicating no file descriptor available for memory buffer
}

} // namespace faiss
