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
#include <dlfcn.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <mutex>

#ifndef FAISS_METALLIB_BUILD_PATH
#error "FAISS_METALLIB_BUILD_PATH must be defined by CMake"
#endif

namespace faiss {
namespace gpu_metal {

namespace {

static const char* kThreadgroupNames[] = {
        "topk_threadgroup_32",
        "topk_threadgroup_64",
        "topk_threadgroup_128",
        "topk_threadgroup_256",
        "topk_threadgroup_512",
        "topk_threadgroup_1024",
        "topk_threadgroup_2048"};

} // namespace

constexpr int MetalKernels::kTopKVariantSizes[];

int MetalKernels::selectTopKVariantIndex(int k) {
    for (int i = 0; i < kNumTopKVariants; i++) {
        if (k <= kTopKVariantSizes[i])
            return i;
    }
    return kNumTopKVariants - 1;
}

// Locate MetalDistance.metallib at runtime.
// Search order:
//   1. FAISS_METALLIB_PATH env var (explicit override)
//   2. Compiled-in build-tree path (development builds)
//   3. Next to the library/binary containing this code (dladdr;
//      works for pip/conda where metallib is in the Python package)
static NSString* findMetalLibPath() {
    NSFileManager* fm = [NSFileManager defaultManager];

    const char* envPath = getenv("FAISS_METALLIB_PATH");
    if (envPath) {
        NSString* ep = @(envPath);
        if ([fm fileExistsAtPath:ep])
            return ep;
    }

    NSString* buildPath = @(FAISS_METALLIB_BUILD_PATH);
    if ([fm fileExistsAtPath:buildPath])
        return buildPath;

    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&faiss::gpu_metal::getMetalKernels),
               &info) &&
        info.dli_fname) {
        NSString* libDir =
                [@(info.dli_fname) stringByDeletingLastPathComponent];
        NSString* installed = [libDir
                stringByAppendingPathComponent:@"MetalDistance.metallib"];
        if ([fm fileExistsAtPath:installed])
            return installed;
    }

    return nil;
}

MetalKernels::MetalKernels(id<MTLDevice> device)
        : device_(device), library_(nil) {
    NSString* path = findMetalLibPath();
    FAISS_THROW_IF_NOT_MSG(
            path,
            "MetalDistance.metallib not found. Set FAISS_METALLIB_PATH or "
            "reinstall faiss.");
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

void MetalKernels::encodeL2WithNorms(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> distances,
        id<MTLBuffer> vecNorms,
        int nq,
        int nb,
        int d) {
    [enc setComputePipelineState:pipeline("l2_with_norms")];
    [enc setBuffer:queries offset:0 atIndex:0];
    [enc setBuffer:vectors offset:0 atIndex:1];
    [enc setBuffer:distances offset:0 atIndex:2];
    uint32_t args[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    [enc setBuffer:vecNorms offset:0 atIndex:4];
    const NSUInteger w = 16, h = 16;
    MTLSize grid = MTLSizeMake(
            ((NSUInteger)nb + w - 1) / w, ((NSUInteger)nq + h - 1) / h, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
}

void MetalKernels::encodeComputeNorms(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> vectors,
        id<MTLBuffer> norms,
        int nb,
        int d) {
    auto ps = pipeline("compute_norms");
    [enc setComputePipelineState:ps];
    [enc setBuffer:vectors offset:0 atIndex:0];
    [enc setBuffer:norms offset:0 atIndex:1];
    uint32_t args[2] = {(uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:2];
    NSUInteger tgSize =
            std::min((NSUInteger)256, ps.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = ((NSUInteger)nb + tgSize - 1) / tgSize;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
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

void MetalKernels::encodeIVFScanList(
        id<MTLComputeCommandEncoder> enc,
        IVFScanVariant variant,
        id<MTLBuffer> queries,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        id<MTLBuffer> perListDist,
        id<MTLBuffer> perListIdx,
        id<MTLBuffer> paramsBuf,
        int nq,
        int nprobe,
        id<MTLBuffer> ilCodesOffset) {
    const char* name;
    NSUInteger tgSize;
    switch (variant) {
        case IVFScanVariant::Small:
            name = "ivf_scan_list_small";
            tgSize = 32;
            break;
        case IVFScanVariant::Interleaved:
            name = "ivf_scan_list_interleaved";
            tgSize = 256;
            break;
        default:
            name = "ivf_scan_list";
            tgSize = 256;
            break;
    }
    [enc setComputePipelineState:pipeline(name)];
    [enc setBuffer:queries offset:0 atIndex:0];
    [enc setBuffer:codes offset:0 atIndex:1];
    [enc setBuffer:ids offset:0 atIndex:2];
    [enc setBuffer:listOffset offset:0 atIndex:3];
    [enc setBuffer:listLength offset:0 atIndex:4];
    [enc setBuffer:coarseAssign offset:0 atIndex:5];
    [enc setBuffer:perListDist offset:0 atIndex:6];
    [enc setBuffer:perListIdx offset:0 atIndex:7];
    [enc setBuffer:paramsBuf offset:0 atIndex:8];
    if (variant == IVFScanVariant::Interleaved && ilCodesOffset) {
        [enc setBuffer:ilCodesOffset offset:0 atIndex:9];
    }
    NSUInteger totalTGs = (NSUInteger)nq * (NSUInteger)nprobe;
    [enc dispatchThreadgroups:MTLSizeMake(totalTGs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
}

void MetalKernels::encodeIVFMergeLists(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> perListDist,
        id<MTLBuffer> perListIdx,
        id<MTLBuffer> outDist,
        id<MTLBuffer> outIdx,
        id<MTLBuffer> paramsBuf,
        int nq) {
    [enc setComputePipelineState:pipeline("ivf_merge_lists")];
    [enc setBuffer:perListDist offset:0 atIndex:0];
    [enc setBuffer:perListIdx offset:0 atIndex:1];
    [enc setBuffer:outDist offset:0 atIndex:2];
    [enc setBuffer:outIdx offset:0 atIndex:3];
    [enc setBuffer:paramsBuf offset:0 atIndex:4];
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

extern std::string& ref_gpu_compile_options();

struct InitMetalCompileOptions {
    InitMetalCompileOptions() {
        ref_gpu_compile_options() = "MAC_METAL ";
    }
};

InitMetalCompileOptions InitMetalCompileOptions_instance;

} // namespace faiss
