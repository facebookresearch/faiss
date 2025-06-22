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

#include <cuvs/core/bitset.hpp>
#include <faiss/gpu/GpuResources.h>
#include <faiss/impl/IDSelector.h>

#pragma GCC visibility push(default)
namespace faiss::gpu {
/// Convert a Faiss IDSelector to a cuvs::core::bitset_view
/// @param res The GpuResources object to use for the conversion
/// @param selector The Faiss IDSelector to convert
/// @param bitset The cuvs::core::bitset_view to store the result
/// @param num_threads Number of threads to use for the conversion. If 0, the
/// number of threads is set to the number of available threads.
void convert_to_bitset(
        faiss::gpu::GpuResources* res,
        const faiss::IDSelector& selector,
        cuvs::core::bitset_view<uint32_t, uint32_t> bitset,
        int num_threads = 0);
} // namespace faiss::gpu
