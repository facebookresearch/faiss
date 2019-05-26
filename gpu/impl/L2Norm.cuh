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

void runL2Norm(Tensor<float, 2, true>& input,
               bool inputRowMajor,
               Tensor<float, 1, true>& output,
               bool normSquared,
               cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runL2Norm(Tensor<half, 2, true>& input,
               bool inputRowMajor,
               Tensor<half, 1, true>& output,
               bool normSquared,
               cudaStream_t stream);
#endif

} } // namespace
