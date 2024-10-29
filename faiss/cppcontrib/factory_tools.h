/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>

namespace faiss {

std::string reverse_index_factory(const faiss::Index* index);

} // namespace faiss
