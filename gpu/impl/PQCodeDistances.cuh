/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "../utils/Tensor.cuh"
#include "../utils/NoTypeTensor.cuh"
#include <cublas_v2.h>

namespace faiss { namespace gpu {

class DeviceMemory;

/// pqCentroids is of the form (sub q)(sub dim)(code id)
/// Calculates the distance from the (query - centroid) residual to
/// each sub-code vector, for the given list of query results in
/// topQueryToCentroid
void runPQCodeDistances(Tensor<float, 3, true>& pqCentroids,
                        Tensor<float, 2, true>& queries,
                        Tensor<float, 2, true>& coarseCentroids,
                        Tensor<int, 2, true>& topQueryToCentroid,
                        NoTypeTensor<4, true>& outCodeDistances,
                        bool useFloat16Lookup,
                        cudaStream_t stream);

void runPQCodeDistancesMM(Tensor<float, 3, true>& pqCentroids,
                          Tensor<float, 2, true>& queries,
                          Tensor<float, 2, true>& coarseCentroids,
                          Tensor<int, 2, true>& topQueryToCentroid,
                          NoTypeTensor<4, true>& outCodeDistances,
                          bool useFloat16Lookup,
                          DeviceMemory& mem,
                          cublasHandle_t handle,
                          cudaStream_t stream);

} } // namespace
