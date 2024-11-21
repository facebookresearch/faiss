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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <faiss/impl/FaissAssert.h>

#include <memory>
#include <utility>
#include <vector>

#if defined USE_NVIDIA_CUVS
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#endif

namespace faiss {
namespace gpu {

class GpuResources;

enum AllocType {
    /// Unknown allocation type or miscellaneous (not currently categorized)
    Other = 0,

    /// Primary data storage for GpuIndexFlat (the raw matrix of vectors and
    /// vector norms if needed)
    FlatData = 1,

    /// Primary data storage for GpuIndexIVF* (the storage for each individual
    /// IVF list)
    IVFLists = 2,

    /// Quantizer (PQ, SQ) dictionary information
    Quantizer = 3,

    /// For GpuIndexIVFPQ, "precomputed codes" for more efficient PQ lookup
    /// require the use of possibly large tables. These are marked separately
    /// from
    /// Quantizer as these can frequently be 100s - 1000s of MiB in size
    QuantizerPrecomputedCodes = 4,

    ///
    /// StandardGpuResources implementation specific types
    ///

    /// When using StandardGpuResources, temporary memory allocations
    /// (MemorySpace::Temporary) come out of a stack region of memory that is
    /// allocated up front for each gpu (e.g., 1.5 GiB upon initialization).
    /// This
    /// allocation by StandardGpuResources is marked with this AllocType.
    TemporaryMemoryBuffer = 10,

    /// When using StandardGpuResources, any MemorySpace::Temporary allocations
    /// that cannot be satisfied within the TemporaryMemoryBuffer region fall
    /// back
    /// to calling cudaMalloc which are sized to just the request at hand. These
    /// "overflow" temporary allocations are marked with this AllocType.
    TemporaryMemoryOverflow = 11,
};

/// Convert an AllocType to string
std::string allocTypeToString(AllocType t);

/// Memory regions accessible to the GPU
enum MemorySpace {
    /// Temporary device memory (guaranteed to no longer be used upon exit of a
    /// top-level index call, and where the streams using it have completed GPU
    /// work). Typically backed by Device memory (cudaMalloc/cudaFree).
    Temporary = 0,

    /// Managed using cudaMalloc/cudaFree (typical GPU device memory)
    Device = 1,

    /// Managed using cudaMallocManaged/cudaFree (typical Unified CPU/GPU
    /// memory)
    Unified = 2,
};

/// Convert a MemorySpace to string
std::string memorySpaceToString(MemorySpace s);

/// Information on what/where an allocation is
struct AllocInfo {
    inline AllocInfo() {}

    inline AllocInfo(AllocType at, int dev, MemorySpace sp, cudaStream_t st)
            : type(at), device(dev), space(sp), stream(st) {}

    /// Returns a string representation of this info
    std::string toString() const;

    /// The internal category of the allocation
    AllocType type = AllocType::Other;

    /// The device on which the allocation is happening
    int device = 0;

    /// The memory space of the allocation
    MemorySpace space = MemorySpace::Device;

    /// The stream on which new work on the memory will be ordered (e.g., if a
    /// piece of memory cached and to be returned for this call was last used on
    /// stream 3 and a new memory request is for stream 4, the memory manager
    /// will synchronize stream 4 to wait for the completion of stream 3 via
    /// events or other stream synchronization.
    ///
    /// The memory manager guarantees that the returned memory is free to use
    /// without data races on this stream specified.
    cudaStream_t stream = nullptr;
};

/// Create an AllocInfo for the current device with MemorySpace::Device
AllocInfo makeDevAlloc(AllocType at, cudaStream_t st);

/// Create an AllocInfo for the current device with MemorySpace::Temporary
AllocInfo makeTempAlloc(AllocType at, cudaStream_t st);

/// Create an AllocInfo for the current device
AllocInfo makeSpaceAlloc(AllocType at, MemorySpace sp, cudaStream_t st);

/// Information on what/where an allocation is, along with how big it should be
struct AllocRequest : public AllocInfo {
    inline AllocRequest() {}

    inline AllocRequest(const AllocInfo& info, size_t sz)
            : AllocInfo(info), size(sz) {}

    inline AllocRequest(
            AllocType at,
            int dev,
            MemorySpace sp,
            cudaStream_t st,
            size_t sz)
            : AllocInfo(at, dev, sp, st), size(sz) {}

    /// Returns a string representation of this request
    std::string toString() const;

    /// The size in bytes of the allocation
    size_t size = 0;

#if defined USE_NVIDIA_CUVS
    rmm::mr::device_memory_resource* mr = nullptr;
#endif
};

/// A RAII object that manages a temporary memory request
struct GpuMemoryReservation {
    GpuMemoryReservation();
    GpuMemoryReservation(
            GpuResources* r,
            int dev,
            cudaStream_t str,
            void* p,
            size_t sz);
    GpuMemoryReservation(GpuMemoryReservation&& m) noexcept;
    ~GpuMemoryReservation();

    GpuMemoryReservation& operator=(GpuMemoryReservation&& m);

    inline void* get() {
        return data;
    }

    void release();

    GpuResources* res;
    int device;
    cudaStream_t stream;
    void* data;
    size_t size;
};

/// Base class of GPU-side resource provider; hides provision of
/// cuBLAS handles, CUDA streams and all device memory allocation performed
class GpuResources {
   public:
    virtual ~GpuResources();

    /// Call to pre-allocate resources for a particular device. If this is
    /// not called, then resources will be allocated at the first time
    /// of demand
    virtual void initializeForDevice(int device) = 0;

    /// Does the given GPU support bfloat16?
    virtual bool supportsBFloat16(int device) = 0;

    /// Returns the cuBLAS handle that we use for the given device
    virtual cublasHandle_t getBlasHandle(int device) = 0;

    /// Returns the stream that we order all computation on for the
    /// given device
    virtual cudaStream_t getDefaultStream(int device) = 0;

#if defined USE_NVIDIA_CUVS
    /// Returns the raft handle for the given device which can be used to
    /// make calls to other raft primitives.
    virtual raft::device_resources& getRaftHandle(int device) = 0;
    raft::device_resources& getRaftHandleCurrentDevice();
#endif

    /// Overrides the default stream for a device to the user-supplied stream.
    /// The resources object does not own this stream (i.e., it will not destroy
    /// it).
    virtual void setDefaultStream(int device, cudaStream_t stream) = 0;

    /// Returns the set of alternative streams that we use for the given device
    virtual std::vector<cudaStream_t> getAlternateStreams(int device) = 0;

    /// Memory management
    /// Returns an allocation from the given memory space, ordered with respect
    /// to the given stream (i.e., the first user will be a kernel in this
    /// stream). All allocations are sized internally to be the next highest
    /// multiple of 16 bytes, and all allocations returned are guaranteed to be
    /// 16 byte aligned.
    virtual void* allocMemory(const AllocRequest& req) = 0;

    /// Returns a previous allocation
    virtual void deallocMemory(int device, void* in) = 0;

    /// For MemorySpace::Temporary, how much space is immediately available
    /// without cudaMalloc allocation?
    virtual size_t getTempMemoryAvailable(int device) const = 0;

    /// Returns the available CPU pinned memory buffer
    virtual std::pair<void*, size_t> getPinnedMemory() = 0;

    /// Returns the stream on which we perform async CPU <-> GPU copies
    virtual cudaStream_t getAsyncCopyStream(int device) = 0;

    ///
    /// Functions provided by default
    ///

    /// Does the current GPU support bfloat16?
    bool supportsBFloat16CurrentDevice();

    /// Calls getBlasHandle with the current device
    cublasHandle_t getBlasHandleCurrentDevice();

    /// Calls getDefaultStream with the current device
    cudaStream_t getDefaultStreamCurrentDevice();

    /// Calls getTempMemoryAvailable with the current device
    size_t getTempMemoryAvailableCurrentDevice() const;

    /// Returns a temporary memory allocation via a RAII object
    GpuMemoryReservation allocMemoryHandle(const AllocRequest& req);

    /// Synchronizes the CPU with respect to the default stream for the
    /// given device
    // equivalent to cudaDeviceSynchronize(getDefaultStream(device))
    void syncDefaultStream(int device);

    /// Calls syncDefaultStream for the current device
    void syncDefaultStreamCurrentDevice();

    /// Calls getAlternateStreams for the current device
    std::vector<cudaStream_t> getAlternateStreamsCurrentDevice();

    /// Calls getAsyncCopyStream for the current device
    cudaStream_t getAsyncCopyStreamCurrentDevice();
};

/// Interface for a provider of a shared resources object. This is to avoid
/// interfacing std::shared_ptr to Python
class GpuResourcesProvider {
   public:
    virtual ~GpuResourcesProvider();

    /// Returns the shared resources object
    virtual std::shared_ptr<GpuResources> getResources() = 0;
};

/// A simple wrapper for a GpuResources object to make a GpuResourcesProvider
/// out of it again
class GpuResourcesProviderFromInstance : public GpuResourcesProvider {
   public:
    explicit GpuResourcesProviderFromInstance(std::shared_ptr<GpuResources> p);
    ~GpuResourcesProviderFromInstance() override;

    std::shared_ptr<GpuResources> getResources() override;

   private:
    std::shared_ptr<GpuResources> res_;
};

} // namespace gpu
} // namespace faiss
