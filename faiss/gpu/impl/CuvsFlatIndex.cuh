// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>

#pragma GCC visibility push(default)
namespace faiss {
namespace gpu {

class GpuResources;

/// Holder of GPU resources for a particular flat index
/// Can be in either float16 or float32 mode. If float32, we only store
/// the vectors in float32.
/// If float16, we store the vectors in both float16 and float32, where float32
/// data is possibly needed for certain residual operations
class CuvsFlatIndex : public FlatIndex {
   public:
    CuvsFlatIndex(
            GpuResources* res,
            int dim,
            bool useFloat16,
            MemorySpace space);

    void query(
            Tensor<float, 2, true>& vecs,
            int k,
            faiss::MetricType metric,
            float metricArg,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            bool exactDistance) override;

    void query(
            Tensor<half, 2, true>& vecs,
            int k,
            faiss::MetricType metric,
            float metricArg,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            bool exactDistance) override;
};

} // namespace gpu
} // namespace faiss
#pragma GCC visibility pop
