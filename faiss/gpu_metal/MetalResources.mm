// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalResources.h"

namespace faiss {
namespace gpu_metal {

MetalResources::MetalResources() : device_(nil), commandQueue_(nil) {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_) {
        commandQueue_ = [device_ newCommandQueue];
    }
}

MetalResources::~MetalResources() {
    commandQueue_ = nil;
    device_ = nil;
}

id<MTLBuffer> MetalResources::allocBuffer(
        size_t size,
        MetalAllocType /*type*/) {
    if (!device_) {
        return nil;
    }
    // Use shared storage so the buffer is accessible from both CPU and GPU
    // (Apple Silicon unified memory). Suitable for temporary and moderate-sized
    // allocations; switch to MTLResourceStorageModePrivate for large
    // GPU-only data when optimizing.
    return [device_ newBufferWithLength:size
                                options:MTLResourceStorageModeShared];
}

void MetalResources::deallocBuffer(
        id<MTLBuffer> buffer,
        MetalAllocType /*type*/) {
    // In Objective-C/ARC, the caller passes their last reference; we do not
    // retain it, so when this function returns the parameter is released.
    (void)buffer;
}

void MetalResources::synchronize() {
    if (!commandQueue_) {
        return;
    }
    id<MTLCommandBuffer> cmdBuf = [commandQueue_ commandBuffer];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

} // namespace gpu_metal
} // namespace faiss
