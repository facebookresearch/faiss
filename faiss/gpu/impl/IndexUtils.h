/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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
int getMaxKSelection();

// Validate the k parameter for search
void validateKSelect(idx_t k);

// Validate the nprobe parameter for search
void validateNProbe(idx_t nprobe);

/// Validate the n (number of vectors) parameter for add, search, reconstruct
void validateNumVectors(idx_t n);

} // namespace gpu
} // namespace faiss
