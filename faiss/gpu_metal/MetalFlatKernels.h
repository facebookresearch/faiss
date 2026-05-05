// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Objective-C++ header. Runs L2/IP distance + top-k via Metal compute.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>

namespace faiss {
namespace gpu_metal {

/// Runs GPU search: distance matrix (L2 or IP) then top-k. Uses shared buffers
/// (queries, vectors, outDistances, outIndices). outIndices are int32
/// (0..nb-1). Maximum k supported by the GPU top-k kernel (256).
int getMetalFlatSearchMaxK();

/// Returns true on success; false if pipeline creation failed.
bool runFlatSearchGPU(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries, // (nq * d) float, row-major
        id<MTLBuffer> vectors, // (nb * d) float, row-major
        int nq,
        int nb,
        int d,
        int k,
        bool isL2,                  // true = L2 squared, false = inner product
        id<MTLBuffer> outDistances, // (nq * k) float
        id<MTLBuffer> outIndices);  // (nq * k) int32

} // namespace gpu_metal
} // namespace faiss
