/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "io_c.h"
#include <faiss/impl/io.h>
#include "../macros_impl.h"

using faiss::IOReader;
using faiss::IOWriter;

struct CustomIOReader : IOReader {
    size_t (*func)(void* ptr, size_t size, size_t nitems) = nullptr;

    CustomIOReader(size_t (*func_in)(void* ptr, size_t size, size_t nitems));

    size_t operator()(void* ptr, size_t size, size_t nitems) override;
};

CustomIOReader::CustomIOReader(
        size_t (*func_in)(void* ptr, size_t size, size_t nitems))
        : func(func_in) {}

size_t CustomIOReader::operator()(void* ptr, size_t size, size_t nitems) {
    return func(ptr, size, nitems);
}

int faiss_CustomIOReader_new(
        FaissCustomIOReader** p_out,
        size_t (*func_in)(void* ptr, size_t size, size_t nitems)) {
    try {
        *p_out = reinterpret_cast<FaissCustomIOReader*>(
                new CustomIOReader(func_in));
    }
    CATCH_AND_HANDLE
}

void faiss_CustomIOReader_free(FaissCustomIOReader* obj) {
    delete reinterpret_cast<CustomIOReader*>(obj);
}

struct CustomIOWriter : IOWriter {
    size_t (*func)(const void* ptr, size_t size, size_t nitems) = nullptr;

    CustomIOWriter(
            size_t (*func_in)(const void* ptr, size_t size, size_t nitems));

    size_t operator()(const void* ptr, size_t size, size_t nitems) override;
};

CustomIOWriter::CustomIOWriter(
        size_t (*func_in)(const void* ptr, size_t size, size_t nitems))
        : func(func_in) {}

size_t CustomIOWriter::operator()(const void* ptr, size_t size, size_t nitems) {
    return func(ptr, size, nitems);
}

int faiss_CustomIOWriter_new(
        FaissCustomIOWriter** p_out,
        size_t (*func_in)(const void* ptr, size_t size, size_t nitems)) {
    try {
        *p_out = reinterpret_cast<FaissCustomIOWriter*>(
                new CustomIOWriter(func_in));
    }
    CATCH_AND_HANDLE
}

void faiss_CustomIOWriter_free(FaissCustomIOWriter* obj) {
    delete reinterpret_cast<CustomIOWriter*>(obj);
}
