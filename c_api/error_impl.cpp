/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "error_impl.h"
#include <faiss/impl/FaissException.h>
#include <exception>
#include "error_c.h"

thread_local std::exception_ptr faiss_last_exception;

const char* faiss_get_last_error() {
    if (faiss_last_exception) {
        try {
            std::rethrow_exception(faiss_last_exception);
        } catch (std::exception& e) {
            return e.what();
        }
    }
    return nullptr;
}
