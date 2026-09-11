// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/common.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/impl/IDSelector.h>
#include <raft/core/bitset.hpp>

#include <optional>

#pragma GCC visibility push(default)
namespace faiss::gpu {
/// Convert a Faiss IDSelector to a RAFT bitset consumed by the cuVS C API.
/// @param res The GpuResources object to use for the conversion
/// @param selector The Faiss IDSelector to convert
/// @param bitset The bitset view to store the result
/// @param num_threads Number of threads to use for the conversion. If 0, the
/// number of threads is set to the number of available threads.
void convert_to_bitset(
        faiss::gpu::GpuResources* res,
        const faiss::IDSelector& selector,
        raft::core::bitset_view<uint32_t, int64_t> bitset,
        int num_threads = 0);

/// Owns the device bitset and DLPack metadata referenced by a cuvsFilter.
class CuvsFilter {
   public:
    CuvsFilter(
            GpuResources* resources,
            const IDSelector* selector,
            int64_t bitCount)
            : filter_{0, NO_FILTER} {
        if (!selector) {
            return;
        }

        auto& raftResources = resources->getRaftHandleCurrentDevice();
        bitset_.emplace(raftResources, bitCount, false);
        convert_to_bitset(resources, *selector, bitset_->view());
        tensor_.emplace(
                bitset_->data(), static_cast<int64_t>(bitset_->n_elements()));
        filter_ =
                cuvsFilter{reinterpret_cast<uintptr_t>(tensor_->get()), BITSET};
    }

    cuvsFilter get() const {
        return filter_;
    }

   private:
    std::optional<raft::core::bitset<uint32_t, int64_t>> bitset_;
    std::optional<CuvsTensor<uint32_t, 1>> tensor_;
    cuvsFilter filter_;
};
} // namespace faiss::gpu
