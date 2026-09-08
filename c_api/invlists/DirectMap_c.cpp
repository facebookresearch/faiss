/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include "DirectMap_c.h"
#include <faiss/invlists/DirectMap.h>

// Keep the C enum in lock-step with faiss::DirectMap::Type. If the C++ enum is
// ever reordered or extended, these will fail to compile instead of silently
// mapping to the wrong type in the static_cast performed by callers.
static_assert(
        DIRECT_MAP_NO_MAP == static_cast<int>(faiss::DirectMap::NoMap),
        "FaissDirectMapType is out of sync with faiss::DirectMap::Type");
static_assert(
        DIRECT_MAP_ARRAY == static_cast<int>(faiss::DirectMap::Array),
        "FaissDirectMapType is out of sync with faiss::DirectMap::Type");
static_assert(
        DIRECT_MAP_HASHTABLE == static_cast<int>(faiss::DirectMap::Hashtable),
        "FaissDirectMapType is out of sync with faiss::DirectMap::Type");
