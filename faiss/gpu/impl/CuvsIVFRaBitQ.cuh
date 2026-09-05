// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <faiss/gpu/GpuIndexIVFRaBitQ.h>
#include <faiss/gpu/utils/Tensor.cuh>

#include <cuvs/neighbors/ivf_rabitq.hpp>

#include <memory>

namespace faiss {
namespace gpu {

class CuvsIVFRaBitQ {
   public:
    CuvsIVFRaBitQ(
            GpuResources* resources,
            int dim,
            idx_t nlist,
            faiss::MetricType metric,
            const GpuIndexIVFRaBitQConfig& config);

    void train(idx_t n, const float* x);

    void search(
            Tensor<float, 2, true>& queries,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            uint32_t nprobe,
            IVFRaBitQSearchMode searchMode);

    void reset();

   private:
    GpuResources* resources_;
    const int dim_;
    const idx_t nlist_;
    const faiss::MetricType metric_;
    const GpuIndexIVFRaBitQConfig config_;
    std::shared_ptr<cuvs::neighbors::ivf_rabitq::index<int64_t>> cuvs_index_;
};

} // namespace gpu
} // namespace faiss
