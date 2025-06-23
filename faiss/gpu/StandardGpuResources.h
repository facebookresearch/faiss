// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#if defined USE_NVIDIA_CUVS
#include <raft/core/device_resources.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#endif

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StackDeviceMemory.h>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#pragma GCC visibility push(default)
namespace faiss {
namespace gpu {

/// Standard implementation of the GpuResources object that provides for a
/// temporary memory manager
class StandardGpuResourcesImpl : public GpuResources {
   public:
    StandardGpuResourcesImpl();

    ~StandardGpuResourcesImpl() override;

    /// Does the given GPU support bfloat16?
    bool supportsBFloat16(int device) override;

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call cudaMalloc / cudaFree at the point of use
    void noTempMemory();

    /// Specify that we wish to use a certain fixed size of memory on
    /// all devices as temporary memory. This is the upper bound for the GPU
    /// memory that we will reserve. We will never go above 1.5 GiB on any GPU;
    /// smaller GPUs (with <= 4 GiB or <= 8 GiB) will use less memory than that.
    /// To avoid any temporary memory allocation, pass 0.
    void setTempMemory(size_t size);

    /// Set amount of pinned memory to allocate, for async GPU <-> CPU
    /// transfers
    void setPinnedMemory(size_t size);

    /// Called to change the stream for work ordering. We do not own `stream`;
    /// i.e., it will not be destroyed when the GpuResources object gets cleaned
    /// up.
    /// We are guaranteed that all Faiss GPU work is ordered with respect to
    /// this stream upon exit from an index or other Faiss GPU call.
    void setDefaultStream(int device, cudaStream_t stream) override;

    /// Revert the default stream to the original stream managed by this
    /// resources object, in case someone called `setDefaultStream`.
    void revertDefaultStream(int device);

    /// Returns the stream for the given device on which all Faiss GPU work is
    /// ordered.
    /// We are guaranteed that all Faiss GPU work is ordered with respect to
    /// this stream upon exit from an index or other Faiss GPU call.
    cudaStream_t getDefaultStream(int device) override;

#if defined USE_NVIDIA_CUVS
    /// Returns the raft handle for the given device which can be used to
    /// make calls to other raft primitives.
    raft::device_resources& getRaftHandle(int device) override;
#endif

    /// Called to change the work ordering streams to the null stream
    /// for all devices
    void setDefaultNullStreamAllDevices();

    /// If enabled, will print every GPU memory allocation and deallocation to
    /// standard output
    void setLogMemoryAllocations(bool enable);

   public:
    /// Internal system calls

    /// Initialize resources for this device
    void initializeForDevice(int device) override;

    cublasHandle_t getBlasHandle(int device) override;

    std::vector<cudaStream_t> getAlternateStreams(int device) override;

    /// Allocate non-temporary GPU memory
    void* allocMemory(const AllocRequest& req) override;

    /// Returns a previous allocation
    void deallocMemory(int device, void* in) override;

    size_t getTempMemoryAvailable(int device) const override;

    /// Export a description of memory used for Python
    std::map<int, std::map<std::string, std::pair<int, size_t>>> getMemoryInfo()
            const;

    std::pair<void*, size_t> getPinnedMemory() override;

    cudaStream_t getAsyncCopyStream(int device) override;

   protected:
    /// Have GPU resources been initialized for this device yet?
    bool isInitialized(int device) const;

    /// Adjust the default temporary memory allocation based on the total GPU
    /// memory size
    static size_t getDefaultTempMemForGPU(int device, size_t requested);

   protected:
    /// Set of currently outstanding memory allocations per device
    /// device -> (alloc request, allocated ptr)
    std::unordered_map<int, std::unordered_map<void*, AllocRequest>> allocs_;

    /// Temporary memory provider, per each device
    std::unordered_map<int, std::unique_ptr<StackDeviceMemory>> tempMemory_;

    /// Our default stream that work is ordered on, one per each device
    std::unordered_map<int, cudaStream_t> defaultStreams_;

    /// This contains particular streams as set by the user for
    /// ordering, if any
    std::unordered_map<int, cudaStream_t> userDefaultStreams_;

    /// Other streams we can use, per each device
    std::unordered_map<int, std::vector<cudaStream_t>> alternateStreams_;

    /// Async copy stream to use for GPU <-> CPU pinned memory copies
    std::unordered_map<int, cudaStream_t> asyncCopyStreams_;

    /// cuBLAS handle for each device
    std::unordered_map<int, cublasHandle_t> blasHandles_;

#if defined USE_NVIDIA_CUVS
    /// raft handle for each device
    std::unordered_map<int, raft::device_resources> raftHandles_;

    /**
     * FIXME: Integrating these in a separate code path for now. Ultimately,
     * it would be nice if we use a simple memory resource abstraction
     * in Faiss so we could plug in whether to use RMM's memory resources
     * or the default.
     *
     * There's enough duplicated logic that it doesn't *seem* to make sense
     * to create a subclass only for the RMM memory resources.
     */

    // managed_memory_resource
    std::unique_ptr<rmm::mr::device_memory_resource> mmr_;

    // pinned_memory_resource
    std::unique_ptr<rmm::mr::host_memory_resource> pmr_;
#endif

    /// Pinned memory allocation for use with this GPU
    void* pinnedMemAlloc_;
    size_t pinnedMemAllocSize_;

    /// Another option is to use a specified amount of memory on all
    /// devices
    size_t tempMemSize_;

    /// Amount of pinned memory we should allocate
    size_t pinnedMemSize_;

    /// Whether or not we log every GPU memory allocation and deallocation
    bool allocLogging_;
};

/// Default implementation of GpuResources that allocates a cuBLAS
/// stream and 2 streams for use, as well as temporary memory.
/// Internally, the Faiss GPU code uses the instance managed by getResources,
/// but this is the user-facing object that is internally reference counted.
class StandardGpuResources : public GpuResourcesProvider {
   public:
    StandardGpuResources();
    ~StandardGpuResources() override;

    std::shared_ptr<GpuResources> getResources() override;

    /// Whether or not the given device supports native bfloat16 arithmetic
    bool supportsBFloat16(int device);

    /// Whether or not the current device supports native bfloat16 arithmetic
    bool supportsBFloat16CurrentDevice();

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call cudaMalloc / cudaFree at the point of use
    void noTempMemory();

    /// Specify that we wish to use a certain fixed size of memory on
    /// all devices as temporary memory. This is the upper bound for the GPU
    /// memory that we will reserve. We will never go above 1.5 GiB on any GPU;
    /// smaller GPUs (with <= 4 GiB or <= 8 GiB) will use less memory than that.
    /// To avoid any temporary memory allocation, pass 0.
    void setTempMemory(size_t size);

    /// Set amount of pinned memory to allocate, for async GPU <-> CPU
    /// transfers
    void setPinnedMemory(size_t size);

    /// Called to change the stream for work ordering. We do not own `stream`;
    /// i.e., it will not be destroyed when the GpuResources object gets cleaned
    /// up.
    /// We are guaranteed that all Faiss GPU work is ordered with respect to
    /// this stream upon exit from an index or other Faiss GPU call.
    void setDefaultStream(int device, cudaStream_t stream);

    /// Revert the default stream to the original stream managed by this
    /// resources object, in case someone called `setDefaultStream`.
    void revertDefaultStream(int device);

    /// Called to change the work ordering streams to the null stream
    /// for all devices
    void setDefaultNullStreamAllDevices();

    /// Export a description of memory used for Python
    std::map<int, std::map<std::string, std::pair<int, size_t>>> getMemoryInfo()
            const;
    /// Returns the current default stream
    cudaStream_t getDefaultStream(int device);

#if defined USE_NVIDIA_CUVS
    /// Returns the raft handle for the given device which can be used to
    /// make calls to other raft primitives.
    raft::device_resources& getRaftHandle(int device);
#endif

    /// Returns the current amount of temp memory available
    size_t getTempMemoryAvailable(int device) const;

    /// Synchronize our default stream with the CPU
    void syncDefaultStreamCurrentDevice();

    /// If enabled, will print every GPU memory allocation and deallocation to
    /// standard output
    void setLogMemoryAllocations(bool enable);

   private:
    std::shared_ptr<StandardGpuResourcesImpl> res_;
};

} // namespace gpu
} // namespace faiss
#pragma GCC visibility pop
