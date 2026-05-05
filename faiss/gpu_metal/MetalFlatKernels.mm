// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MSL kernels: L2 squared / IP distance matrix, then top-k reduction.
 */

#import "MetalFlatKernels.h"
#include <cstring>

namespace faiss {
namespace gpu_metal {

namespace {

static const char* kMSLSource = R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void l2_squared_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // nq, nb, d
    uint2 gid [[thread_position_in_grid]]
) {
    uint nq = params[0], nb = params[1], d = params[2];
    uint i = gid.y;
    uint j = gid.x;
    if (i >= nq || j >= nb) return;
    float sum = 0.0f;
    for (uint t = 0; t < d; t++) {
        float a = queries[i * d + t];
        float b = vectors[j * d + t];
        sum += (a - b) * (a - b);
    }
    distances[i * nb + j] = sum;
}

kernel void ip_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // nq, nb, d
    uint2 gid [[thread_position_in_grid]]
) {
    uint nq = params[0], nb = params[1], d = params[2];
    uint i = gid.y;
    uint j = gid.x;
    if (i >= nq || j >= nb) return;
    float sum = 0.0f;
    for (uint t = 0; t < d; t++)
        sum += queries[i * d + t] * vectors[j * d + t];
    distances[i * nb + j] = sum;
}

// want_min: 1 = k smallest (L2), 0 = k largest (IP). k <= 256.
kernel void topk(
    device const float* distances [[buffer(0)]],
    device float* outDistances [[buffer(1)]],
    device int* outIndices [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // nq, nb, k, want_min
    uint qi [[thread_position_in_grid]]
) {
    uint nq = params[0], nb = params[1], k = params[2], want_min = params[3];
    if (qi >= nq || k == 0) return;
    const device float* row = distances + qi * nb;
    float bestDist[256];
    int bestIdx[256];
    uint kk = min(k, (uint)256);
    uint n = min(kk, nb);
    for (uint i = 0; i < n; i++) {
        bestDist[i] = row[i];
        bestIdx[i] = (int)i;
    }
    // Sort first n by distance (ascending for L2/smallest, descending for IP/largest)
    for (uint i = 0; i < n; i++) {
        for (uint j = i + 1; j < n; j++) {
            bool swap = want_min ? (bestDist[j] < bestDist[i]) : (bestDist[j] > bestDist[i]);
            if (swap) {
                float td = bestDist[i]; bestDist[i] = bestDist[j]; bestDist[j] = td;
                int ti = bestIdx[i]; bestIdx[i] = bestIdx[j]; bestIdx[j] = ti;
            }
        }
    }
    for (uint i = n; i < kk; i++) {
        bestDist[i] = want_min ? 1e38f : -1e38f;
        bestIdx[i] = -1;
    }
    for (uint j = n; j < nb; j++) {
        float v = row[j];
        bool insert = want_min ? (v < bestDist[kk-1]) : (v > bestDist[kk-1]);
        if (!insert) continue;
        uint pos = kk - 1;
        if (want_min) {
            while (pos > 0 && v < bestDist[pos-1]) {
                bestDist[pos] = bestDist[pos-1];
                bestIdx[pos] = bestIdx[pos-1];
                pos--;
            }
        } else {
            while (pos > 0 && v > bestDist[pos-1]) {
                bestDist[pos] = bestDist[pos-1];
                bestIdx[pos] = bestIdx[pos-1];
                pos--;
            }
        }
        bestDist[pos] = v;
        bestIdx[pos] = (int)j;
    }
    for (uint i = 0; i < kk; i++) {
        outDistances[qi * k + i] = bestDist[i];
        outIndices[qi * k + i] = bestIdx[i];
    }
    for (uint i = kk; i < k; i++) {
        outDistances[qi * k + i] = want_min ? 1e38f : -1e38f;
        outIndices[qi * k + i] = -1;
    }
}
)msl";

static constexpr int kMaxK = 256;

} // namespace

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

    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:@(kMSLSource)
                                              options:nil
                                                error:&err];
    if (!lib) {
        return false;
    }

    id<MTLFunction> fnDist = [lib
            newFunctionWithName:isL2 ? @"l2_squared_matrix" : @"ip_matrix"];
    id<MTLFunction> fnTopK = [lib newFunctionWithName:@"topk"];
    if (!fnDist || !fnTopK) {
        return false;
    }

    id<MTLComputePipelineState> psDist =
            [device newComputePipelineStateWithFunction:fnDist error:&err];
    id<MTLComputePipelineState> psTopK =
            [device newComputePipelineStateWithFunction:fnTopK error:&err];
    if (!psDist || !psTopK) {
        return false;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    // Distance matrix: (nb x nq) threadgroups of (1,1) or tile for better
    // occupancy
    const NSUInteger w = 16;
    const NSUInteger h = 16;
    [enc setComputePipelineState:psDist];
    [enc setBuffer:queries offset:0 atIndex:0];
    [enc setBuffer:vectors offset:0 atIndex:1];
    id<MTLBuffer> distMatrix =
            [device newBufferWithLength:(size_t)nq * (size_t)nb * sizeof(float)
                                options:MTLResourceStorageModeShared];
    if (!distMatrix) {
        [enc endEncoding];
        return false;
    }
    [enc setBuffer:distMatrix offset:0 atIndex:2];
    uint32_t distArgs[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
    [enc setBytes:distArgs length:sizeof(distArgs) atIndex:3];
    MTLSize tgSize = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake((nb + w - 1) / w, (nq + h - 1) / h, 1);
    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    // Top-k: nq threads
    [enc setComputePipelineState:psTopK];
    [enc setBuffer:distMatrix offset:0 atIndex:0];
    [enc setBuffer:outDistances offset:0 atIndex:1];
    [enc setBuffer:outIndices offset:0 atIndex:2];
    uint32_t topkArgs[4] = {
            (uint32_t)nq, (uint32_t)nb, (uint32_t)k, isL2 ? 1u : 0u};
    [enc setBytes:topkArgs
             length:sizeof(topkArgs)
            atIndex:3]; // nq, nb, k, want_min
    MTLSize gridTopK = MTLSizeMake(nq, 1, 1);
    [enc dispatchThreadgroups:gridTopK
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

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
