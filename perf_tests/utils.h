// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include <faiss/impl/ScalarQuantizer.h>
#include <map>

namespace faiss::perf_tests {

std::map<std::string, faiss::ScalarQuantizer::QuantizerType> sq_types();

} // namespace faiss::perf_tests
