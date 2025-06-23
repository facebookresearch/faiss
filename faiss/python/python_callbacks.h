/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IVFlib.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/io.h>
#include <faiss/invlists/InvertedLists.h>
#include "Python.h"

//  all callbacks have to acquire the GIL on input

/***********************************************************
 * Callbacks for IO reader and writer
 ***********************************************************/

struct PyCallbackIOWriter : faiss::IOWriter {
    PyObject* callback;
    size_t bs; // maximum write size

    /** Callback: Python function that takes a bytes object and
     *  returns the number of bytes successfully written.
     */
    explicit PyCallbackIOWriter(PyObject* callback, size_t bs = 1024 * 1024);

    size_t operator()(const void* ptrv, size_t size, size_t nitems) override;

    ~PyCallbackIOWriter() override;
};

struct PyCallbackIOReader : faiss::IOReader {
    PyObject* callback;
    size_t bs; // maximum buffer size

    /** Callback: Python function that takes a size and returns a
     * bytes object with the resulting read */
    explicit PyCallbackIOReader(PyObject* callback, size_t bs = 1024 * 1024);

    size_t operator()(void* ptrv, size_t size, size_t nitems) override;

    ~PyCallbackIOReader() override;
};

/***********************************************************
 * Callbacks for IDSelector
 ***********************************************************/

struct PyCallbackIDSelector : faiss::IDSelector {
    PyObject* callback;

    explicit PyCallbackIDSelector(PyObject* callback);

    bool is_member(faiss::idx_t id) const override;

    ~PyCallbackIDSelector() override;
};

/***********************************************************
 * Callbacks for IVF index sharding
 ***********************************************************/

struct PyCallbackShardingFunction : faiss::ivflib::ShardingFunction {
    PyObject* callback;

    explicit PyCallbackShardingFunction(PyObject* callback);

    int64_t operator()(int64_t i, int64_t shard_count) override;

    ~PyCallbackShardingFunction() override;

    PyCallbackShardingFunction(const PyCallbackShardingFunction&) = delete;
    PyCallbackShardingFunction(PyCallbackShardingFunction&&) noexcept = default;
    PyCallbackShardingFunction& operator=(const PyCallbackShardingFunction&) =
            default;
    PyCallbackShardingFunction& operator=(PyCallbackShardingFunction&&) =
            default;
};
