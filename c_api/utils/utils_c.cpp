/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "utils_c.h"
#include <faiss/Index.h>

const char* faiss_get_version() {
    return VERSION_STRING;
}
