/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/ScalarQuantizer.h>
#include <map>

namespace faiss::perf_tests {

std::map<std::string, faiss::ScalarQuantizer::QuantizerType> sq_types();

} // namespace faiss::perf_tests
