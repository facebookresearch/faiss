/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

void runL2SelectMin(
        Tensor<float, 2, true>& productDistances,
        Tensor<float, 1, true>& centroidDistances,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        idx_t k,
        cudaStream_t stream);

}
} // namespace faiss
