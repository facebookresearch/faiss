/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <exception>

/** global variable for holding the last exception thrown by
 * calls to Faiss functions through the C API
 */
extern thread_local std::exception_ptr faiss_last_exception;
