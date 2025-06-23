/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
// I/O code for indexes

#include "index_io_c.h"
#include <faiss/index_io.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexBinary;
using faiss::IOReader;
using faiss::IOWriter;
using faiss::VectorTransform;

int faiss_write_index(const FaissIndex* idx, FILE* f) {
    try {
        faiss::write_index(reinterpret_cast<const Index*>(idx), f);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_fname(const FaissIndex* idx, const char* fname) {
    try {
        faiss::write_index(reinterpret_cast<const Index*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_custom(
        const FaissIndex* idx,
        FaissIOWriter* io_writer,
        int io_flags) {
    try {
        faiss::write_index(
                reinterpret_cast<const Index*>(idx),
                reinterpret_cast<IOWriter*>(io_writer),
                io_flags);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index(FILE* f, int io_flags, FaissIndex** p_out) {
    try {
        auto out = faiss::read_index(f, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_fname(
        const char* fname,
        int io_flags,
        FaissIndex** p_out) {
    try {
        auto out = faiss::read_index(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_custom(
        FaissIOReader* io_reader,
        int io_flags,
        FaissIndex** p_out) {
    try {
        auto out = faiss::read_index(
                reinterpret_cast<IOReader*>(io_reader), io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary(const FaissIndexBinary* idx, FILE* f) {
    try {
        faiss::write_index_binary(reinterpret_cast<const IndexBinary*>(idx), f);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary_fname(
        const FaissIndexBinary* idx,
        const char* fname) {
    try {
        faiss::write_index_binary(
                reinterpret_cast<const IndexBinary*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary_custom(
        const FaissIndexBinary* idx,
        FaissIOWriter* io_writer) {
    try {
        faiss::write_index_binary(
                reinterpret_cast<const IndexBinary*>(idx),
                reinterpret_cast<IOWriter*>(io_writer));
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary(FILE* f, int io_flags, FaissIndexBinary** p_out) {
    try {
        auto out = faiss::read_index_binary(f, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary_fname(
        const char* fname,
        int io_flags,
        FaissIndexBinary** p_out) {
    try {
        auto out = faiss::read_index_binary(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary_custom(
        FaissIOReader* io_reader,
        int io_flags,
        FaissIndexBinary** p_out) {
    try {
        auto out = faiss::read_index_binary(
                reinterpret_cast<IOReader*>(io_reader), io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_VectorTransform_fname(
        const char* fname,
        FaissVectorTransform** p_out) {
    try {
        auto out = faiss::read_VectorTransform(fname);
        *p_out = reinterpret_cast<FaissVectorTransform*>(out);
    }
    CATCH_AND_HANDLE
}
