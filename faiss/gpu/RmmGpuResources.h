/**
* Copyright (c) Facebook, Inc. and its affiliates.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/
/*
* Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StackDeviceMemory.h>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

namespace faiss {
namespace gpu {

/// Standard implementation of the GpuResources object that provides for a
/// temporary memory manager
class RmmGpuResourcesImpl : public StandardGpuResourcesImpl {
  public:
   RmmGpuResourcesImpl();
   ~RmmGpuResourcesImpl() override;


   void initializeForDevice(int id);
   void* allocMemory(const faiss::gpu::AllocRequest&);
   void deallocMemory(int, void*);


  private:
   // cuda_memory_resource
   std::unique_ptr<rmm::mr::device_memory_resource> cmr;

   // managed_memory_resource
   std::unique_ptr<rmm::mr::device_memory_resource> mmr;

   // pinned_memory_resource
   std::unique_ptr<rmm::mr::host_memory_resource> pmr;
};

/// Default implementation of GpuResources that allocates a cuBLAS
/// stream and 2 streams for use, as well as temporary memory.
/// Internally, the Faiss GPU code uses the instance managed by getResources,
/// but this is the user-facing object that is internally reference counted.
class RmmGpuResources : public StandardGpuResources {
  public:
   RmmGpuResources();
   ~RmmGpuResources() override;

   std::shared_ptr<GpuResources> getResources();

   private:
   std::shared_ptr<RmmGpuResourcesImpl> res_;
};

} // namespace gpu
} // namespace faiss
