// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MetalKernels implementation: library loading, pipeline caching, and
 * encode methods for Metal compute kernels used by the Faiss Metal backend.
 */

#import "MetalKernels.h"
#include <mutex>

#ifndef FAISS_METALLIB_PATH
#error "FAISS_METALLIB_PATH must be defined by CMake"
#endif

namespace faiss {
namespace gpu_metal {

namespace {

static const char* kThreadgroupNames[] = {
        "topk_threadgroup_32",
        "topk_threadgroup_64",
        "topk_threadgroup_128",
        "topk_threadgroup_256"};

} // namespace

constexpr int MetalKernels::kTopKVariantSizes[];

int MetalKernels::selectTopKVariantIndex(int k) {
    for (int i = 0; i < kNumTopKVariants; i++) {
        if (k <= kTopKVariantSizes[i])
            return i;
    }
    return kNumTopKVariants - 1;
}

MetalKernels::MetalKernels(id<MTLDevice> device)
        : device_(device), library_(nil) {
    NSString* path = @(FAISS_METALLIB_PATH);
    NSURL* url = [NSURL fileURLWithPath:path];
    NSError* err = nil;
    library_ = [device_ newLibraryWithURL:url error:&err];
}

MetalKernels::~MetalKernels() = default;

bool MetalKernels::isValid() const {
    return library_ != nil;
}

id<MTLComputePipelineState> MetalKernels::pipeline(const char* name) {
    std::string key(name);
    auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second;
    id<MTLFunction> fn = [library_ newFunctionWithName:@(name)];
    if (!fn)
        return nil;
    NSError* err = nil;
    id<MTLComputePipelineState> ps =
            [device_ newComputePipelineStateWithFunction:fn error:&err];
    if (!ps)
        return nil;
    cache_[key] = ps;
    return ps;
}

void MetalKernels::encodeDistanceMatrix(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> distances,
        int nq,
        int nb,
        int d,
        MetricType metric) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "Metal distance matrix only supports L2 and inner product");
    const char* kernel =
            metric == METRIC_L2 ? "l2_squared_matrix" : "ip_matrix";
    [enc setComputePipelineState:pipeline(kernel)];
    [enc setBuffer:queries offset:0 atIndex:0];
    [enc setBuffer:vectors offset:0 atIndex:1];
    [enc setBuffer:distances offset:0 atIndex:2];
    uint32_t args[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    const NSUInteger tileN = 32;
    const NSUInteger tileM = 32;
    MTLSize grid = MTLSizeMake(
            ((NSUInteger)nb + tileN - 1) / tileN,
            ((NSUInteger)nq + tileM - 1) / tileM,
            1);
    [enc dispatchThreadgroups:grid
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

void MetalKernels::encodeTopKThreadgroup(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> distances,
        id<MTLBuffer> outDist,
        id<MTLBuffer> outIdx,
        int nq,
        int nb,
        int k,
        bool wantMin) {
    int vi = selectTopKVariantIndex(k);
    [enc setComputePipelineState:pipeline(kThreadgroupNames[vi])];
    [enc setBuffer:distances offset:0 atIndex:0];
    [enc setBuffer:outDist offset:0 atIndex:1];
    [enc setBuffer:outIdx offset:0 atIndex:2];
    uint32_t args[4] = {
            (uint32_t)nq, (uint32_t)nb, (uint32_t)k, wantMin ? 1u : 0u};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

MetalKernels& getMetalKernels(id<MTLDevice> device) {
    static std::mutex mu;
    static std::unordered_map<uintptr_t, std::unique_ptr<MetalKernels>> map;
    uintptr_t key = (uintptr_t)(__bridge void*)device;
    std::lock_guard<std::mutex> lock(mu);
    auto& ptr = map[key];
    if (!ptr)
        ptr = std::make_unique<MetalKernels>(device);
    return *ptr;
}

} // namespace gpu_metal
} // namespace faiss
