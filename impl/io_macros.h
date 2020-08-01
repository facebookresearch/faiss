/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/*************************************************************
 * I/O macros
 *
 * we use macros so that we have a line number to report in abort
 * (). This makes debugging a lot easier. The IOReader or IOWriter is
 * always called f and thus is not passed in as a macro parameter.
 **************************************************************/


#define READANDCHECK(ptr, n) {                                  \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);              \
        FAISS_THROW_IF_NOT_FMT(ret == (n),                      \
            "read error in %s: %ld != %ld (%s)",                \
            f->name.c_str(), ret, size_t(n), strerror(errno));  \
    }

#define READ1(x)  READANDCHECK(&(x), 1)

// will fail if we write 256G of data at once...
#define READVECTOR(vec) {                       \
        long size;                            \
        READANDCHECK (&size, 1);                \
        FAISS_THROW_IF_NOT (size >= 0 && size < (1L << 40));  \
        (vec).resize (size);                    \
        READANDCHECK ((vec).data (), size);     \
    }


#define READSTRING(s) {                      \
        size_t size = (s).size ();            \
        WRITEANDCHECK (&size, 1);               \
        WRITEANDCHECK ((s).c_str(), size);      \
    }

#define WRITEANDCHECK(ptr, n) {                                 \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);              \
        FAISS_THROW_IF_NOT_FMT(ret == (n),                      \
            "write error in %s: %ld != %ld (%s)",               \
            f->name.c_str(), ret, size_t(n), strerror(errno));  \
    }

#define WRITE1(x) WRITEANDCHECK(&(x), 1)

#define WRITEVECTOR(vec) {                      \
        size_t size = (vec).size ();            \
        WRITEANDCHECK (&size, 1);               \
        WRITEANDCHECK ((vec).data (), size);    \
    }
