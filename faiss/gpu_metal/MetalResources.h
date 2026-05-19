// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * This header uses Objective-C types (Metal framework: id, nil, MTLDevice,
 * etc.). For correct IDE/linter behavior, associate this file with
 * "Objective-C++":
 *
 */

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>

namespace faiss {
namespace gpu_metal {

/// Allocation type for Metal buffers (mirrors faiss::gpu::AllocType roles).
enum MetalAllocType {
    Other = 0,
    FlatData = 1,
    IVFLists = 2,
    Quantizer = 3,
    QuantizerPrecomputedCodes = 4,
    TemporaryMemoryBuffer = 10,
    TemporaryMemoryOverflow = 11,
};

/// Owns Metal device, command queue, and provides buffer allocation.
/// Mirrors the roles of faiss::gpu::GpuResources for the Metal backend.
class MetalResources {
   public:
    MetalResources();
    ~MetalResources();

    MetalResources(const MetalResources&) = delete;
    MetalResources& operator=(const MetalResources&) = delete;

    /// Returns the Metal device (nil if no Metal-capable device is available).
    id<MTLDevice> getDevice() const {
        return device_;
    }

    /// Returns the command queue for the device (nil if device is nil).
    id<MTLCommandQueue> getCommandQueue() const {
        return commandQueue_;
    }

    /// Allocates a buffer of the given size (bytes). Caller owns the returned
    /// buffer and must call deallocBuffer when done, or the buffer will leak.
    /// Returns nil on failure (e.g. device nil or allocation failure).
    id<MTLBuffer> allocBuffer(size_t size, MetalAllocType type);

    /// Releases a buffer previously returned by allocBuffer. The caller must
    /// not use the buffer after this call.
    void deallocBuffer(id<MTLBuffer> buffer, MetalAllocType type);

    /// Blocks until all work submitted to the default command queue has
    /// completed.
    void synchronize();

    /// Returns true if the Metal device and queue are available.
    bool isAvailable() const {
        return device_ != nil && commandQueue_ != nil;
    }

   private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;
};

} // namespace gpu_metal
} // namespace faiss
