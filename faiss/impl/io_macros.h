/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/maybe_owned_vector.h>

/*************************************************************
 * I/O macros
 *
 * we use macros so that we have a line number to report in abort
 * (). This makes debugging a lot easier. The IOReader or IOWriter is
 * always called f and thus is not passed in as a macro parameter.
 **************************************************************/

namespace faiss {
size_t get_deserialization_vector_byte_limit();
} // namespace faiss

#define READANDCHECK(ptr, n)                         \
    {                                                \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);   \
        FAISS_THROW_IF_NOT_FMT(                      \
                ret == (n),                          \
                "read error in %s: %zd != %zd (%s)", \
                f->name.c_str(),                     \
                ret,                                 \
                size_t(n),                           \
                strerror(errno));                    \
    }

#define READ1(x) READANDCHECK(&(x), 1)

#define READ1_DUMMY(x_type) \
    {                       \
        x_type x = {};      \
        READ1(x);           \
    }

// Rejects vectors whose total allocation would exceed the configurable
// byte limit (default 1 TB).
#define READVECTOR(vec)                                                  \
    {                                                                    \
        size_t size;                                                     \
        READANDCHECK(&size, 1);                                          \
        FAISS_THROW_IF_NOT(                                              \
                size < (faiss::get_deserialization_vector_byte_limit() / \
                        sizeof(*(vec).data())));                         \
        FAISS_THROW_IF_NOT_FMT(                                          \
                size <= SIZE_MAX / sizeof((vec)[0]),                     \
                "READVECTOR: size %zu would overflow for element "       \
                "size %zu",                                              \
                size,                                                    \
                sizeof((vec)[0]));                                       \
        (vec).resize(size);                                              \
        READANDCHECK((vec).data(), size);                                \
    }

#define WRITEANDCHECK(ptr, n)                         \
    {                                                 \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);    \
        FAISS_THROW_IF_NOT_FMT(                       \
                ret == (n),                           \
                "write error in %s: %zd != %zd (%s)", \
                f->name.c_str(),                      \
                ret,                                  \
                size_t(n),                            \
                strerror(errno));                     \
    }

#define WRITE1(x) WRITEANDCHECK(&(x), 1)

#define WRITEVECTOR(vec)                   \
    {                                      \
        size_t size = (vec).size();        \
        WRITEANDCHECK(&size, 1);           \
        WRITEANDCHECK((vec).data(), size); \
    }

// read/write xb vector for backwards compatibility of IndexFlat

#define WRITEXBVECTOR(vec)                         \
    {                                              \
        FAISS_THROW_IF_NOT((vec).size() % 4 == 0); \
        size_t size = (vec).size() / 4;            \
        WRITEANDCHECK(&size, 1);                   \
        WRITEANDCHECK((vec).data(), size * 4);     \
    }

#define READXBVECTOR(vec)                                                \
    {                                                                    \
        size_t size;                                                     \
        READANDCHECK(&size, 1);                                          \
        FAISS_THROW_IF_NOT(                                              \
                size < (faiss::get_deserialization_vector_byte_limit() / \
                        (4 * sizeof(*(vec).data()))));                   \
        size *= 4;                                                       \
        (vec).resize(size);                                              \
        READANDCHECK((vec).data(), size);                                \
    }
