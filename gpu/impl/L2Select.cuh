/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "../utils/Float16.cuh"
#include "../utils/Tensor.cuh"

namespace faiss { namespace gpu {

void runL2SelectMin(Tensor<float, 2, true>& productDistances,
                    Tensor<float, 1, true>& centroidDistances,
                    Tensor<float, 2, true>& outDistances,
                    Tensor<int, 2, true>& outIndices,
                    int k,
                    cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runL2SelectMin(Tensor<half, 2, true>& productDistances,
                    Tensor<half, 1, true>& centroidDistances,
                    Tensor<half, 2, true>& outDistances,
                    Tensor<int, 2, true>& outIndices,
                    int k,
                    cudaStream_t stream);
#endif

} } // namespace
