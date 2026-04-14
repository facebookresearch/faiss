/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace faiss {
namespace gpu {

template <typename IndexT, typename MultiIndexT>
__device__ inline MultiIndexT toMultiIndex(
        int& codebookSize,
        IndexT& index1,
        IndexT& index2) {
    return index1 + index2 * (MultiIndexT)codebookSize;
}

} // namespace gpu
} // namespace faiss
