/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Python.h"
#include <faiss/impl/io.h>
#include <faiss/InvertedLists.h>

//  all callbacks have to acquire the GIL on input


/***********************************************************
 * Callbacks for IO reader and writer
 ***********************************************************/

struct PyCallbackIOWriter: faiss::IOWriter {

    PyObject * callback;
    size_t bs; // maximum write size

    /** Callback: Python function that takes a bytes object and
     *  returns the number of bytes successfully written.
     */
    explicit PyCallbackIOWriter(PyObject *callback,
                                size_t bs = 1024 * 1024);

    size_t operator()(const void *ptrv, size_t size, size_t nitems) override;

    ~PyCallbackIOWriter() override;

};


struct PyCallbackIOReader: faiss::IOReader {
    PyObject * callback;
    size_t bs; // maximum buffer size

    /** Callback: Python function that takes a size and returns a
     * bytes object with the resulting read */
    explicit PyCallbackIOReader(PyObject *callback,
                                size_t bs = 1024 * 1024);

    size_t operator()(void *ptrv, size_t size, size_t nitems) override;

    ~PyCallbackIOReader() override;

};
