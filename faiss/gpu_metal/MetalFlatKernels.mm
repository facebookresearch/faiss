// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Flat search dispatch: distance matrix + top-k via MetalKernels.
 */

#import "MetalFlatKernels.h"
#include <faiss/MetricType.h>
#import "MetalKernels.h"

namespace faiss {
namespace gpu_metal {

static constexpr int kMaxK = MetalKernels::kMaxK;

bool runFlatSearchGPU(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        int nq,
        int nb,
        int d,
        int k,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices) {
    if (!device || !queue || !queries || !vectors || !outDistances ||
        !outIndices) {
        return false;
    }
    if (k <= 0 || k > kMaxK) {
        return false;
    }

    MetalKernels& kernels = getMetalKernels(device);
    if (!kernels.isValid()) {
        return false;
    }

    id<MTLBuffer> distMatrix =
            [device newBufferWithLength:(size_t)nq * (size_t)nb * sizeof(float)
                                options:MTLResourceStorageModeShared];
    if (!distMatrix) {
        return false;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    MetricType metric = isL2 ? METRIC_L2 : METRIC_INNER_PRODUCT;
    kernels.encodeDistanceMatrix(
            enc, queries, vectors, distMatrix, nq, nb, d, metric);

    kernels.encodeTopKThreadgroup(
            enc, distMatrix, outDistances, outIndices, nq, nb, k, isL2);

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return true;
}

int getMetalFlatSearchMaxK() {
    return kMaxK;
}

} // namespace gpu_metal
} // namespace faiss
