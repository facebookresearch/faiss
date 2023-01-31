/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

/// Holder of GPU resources for a particular flat index
/// Can be in either float16 or float32 mode. If float32, we only store
/// the vectors in float32.
/// If float16, we store the vectors in both float16 and float32, where float32
/// data is possibly needed for certain residual operations
class RaftFlatIndex : public FlatIndex {
   public:
    RaftFlatIndex(
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
            Tensor<int, 2, true>& outIndices,
            bool exactDistance) override;
};

} // namespace gpu
} // namespace faiss
