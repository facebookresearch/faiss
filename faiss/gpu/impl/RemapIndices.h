/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <vector>

namespace faiss {
namespace gpu {

/// Utility function to translate (list id, offset) to a user index on
/// the CPU. In a cpp in order to use OpenMP.
void ivfOffsetToUserIndex(
        Index::idx_t* indices,
        int numLists,
        int queries,
        int k,
        const std::vector<std::vector<Index::idx_t>>& listOffsetToUserIndex);

} // namespace gpu
} // namespace faiss
