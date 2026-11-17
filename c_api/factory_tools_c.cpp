/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "factory_tools_c.h"
#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/factory_tools.h>
#include <faiss/impl/FaissAssert.h>
#include <cstring>
#include <string>
#include "macros_impl.h"

namespace {

void copy_to_buffer(const std::string& s, char* buffer, size_t size) {
    FAISS_THROW_IF_NOT_FMT(
            s.size() < size,
            "buffer of size %zu is too small for factory string of length %zu",
            size,
            s.size());
    std::memcpy(buffer, s.c_str(), s.size() + 1);
}

} // namespace

int faiss_reverse_index_factory(
        const FaissIndex* index,
        char* buffer,
        size_t size) {
    try {
        copy_to_buffer(
                faiss::reverse_index_factory(
                        reinterpret_cast<const faiss::Index*>(index)),
                buffer,
                size);
    }
    CATCH_AND_HANDLE
}

int faiss_reverse_index_binary_factory(
        const FaissIndexBinary* index,
        char* buffer,
        size_t size) {
    try {
        copy_to_buffer(
                faiss::reverse_index_factory(
                        reinterpret_cast<const faiss::IndexBinary*>(index)),
                buffer,
                size);
    }
    CATCH_AND_HANDLE
}
