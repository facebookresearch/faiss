/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <faiss/gpu/GpuResources.h>
#include <memory>
#include <string>

namespace faiss {
namespace gpu {

/// Device memory manager that provides fixed size memory allocations
/// out of a region of memory, for a single device
class FixedDeviceMemory {
   public:
    FixedDeviceMemory(
            GpuResources* res,
            int device,
            size_t allocSize,
            MemorySpace space = MemorySpace::Device);

    virtual ~FixedDeviceMemory();

    int getDevice() const;

    void* allocMemory(size_t size);

    size_t getSizeAvailable() const;
    std::string toString() const;

   protected:
    /// Our GpuResources object
    GpuResources* res_;

    /// Device this allocation is on
    int device_;

    /// Where our temporary memory buffer is allocated;
    char* alloc_;

    /// Total size of our allocation
    size_t allocSize_;

    size_t offset_;
};

} // namespace gpu
} // namespace faiss
