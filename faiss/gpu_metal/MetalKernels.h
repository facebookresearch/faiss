// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MetalKernels: typed wrapper around Metal compute kernels.
 * Owns library compilation, pipeline caching, and dispatch encoding.
 */

#pragma once

#import <Metal/Metal.h>
#include <faiss/MetricType.h>
#include <string>
#include <unordered_map>

namespace faiss {
namespace gpu_metal {

class MetalKernels {
   public:
    explicit MetalKernels(id<MTLDevice> device);
    ~MetalKernels();

    bool isValid() const;
    static constexpr int kMaxK = 256;

    void encodeDistanceMatrix(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> distances,
            int nq,
            int nb,
            int d,
            MetricType metric);

    void encodeTopKThreadgroup(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> distances,
            id<MTLBuffer> outDist,
            id<MTLBuffer> outIdx,
            int nq,
            int nb,
            int k,
            bool wantMin);

    static int selectTopKVariantIndex(int k);

   private:
    id<MTLComputePipelineState> pipeline(const char* name);

    id<MTLDevice> device_;
    id<MTLLibrary> library_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> cache_;

    static constexpr int kTopKVariantSizes[] = {32, 64, 128, 256};
    static constexpr int kNumTopKVariants = 4;
};

MetalKernels& getMetalKernels(id<MTLDevice> device);

} // namespace gpu_metal
} // namespace faiss
