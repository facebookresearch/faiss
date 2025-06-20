/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>

namespace faiss {
namespace gpu {

/// A collection of various utility functions for index implementation

/// Returns the maximum k-selection value supported based on the CUDA SDK that
/// we were compiled with. .cu files can use DeviceDefs.cuh, but this is for
/// non-CUDA files
int getMaxKSelection(bool use_cuvs = false);

// Validate the k parameter for search
void validateKSelect(int k, bool use_cuvs = false);

// Validate the nprobe parameter for search
void validateNProbe(size_t nprobe, bool use_cuvs = false);

} // namespace gpu
} // namespace faiss
