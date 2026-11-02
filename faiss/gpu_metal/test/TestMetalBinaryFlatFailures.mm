// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu_metal/MetalIndexBinaryFlat.h>
#import <objc/runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace faiss::gpu_metal;

namespace {
IMP original;

typedef id Lookup(id, SEL, NSString*) __attribute__((ns_returns_retained));
id rejectFunction(id self, SEL selector, NSString* name)
        __attribute__((ns_returns_retained));
id rejectFunction(id self, SEL selector, NSString* name) {
    if ([name isEqualToString:@"hamming_distance_topk"]) {
        return nil;
    }
    return reinterpret_cast<Lookup*>(original)(self, selector, name);
}

typedef id<MTLComputePipelineState> CreatePipeline(
        id,
        SEL,
        id<MTLFunction>,
        NSError**) __attribute__((ns_returns_retained));
id<MTLComputePipelineState> rejectPipeline(
        id self,
        SEL selector,
        id<MTLFunction> function,
        NSError** error) __attribute__((ns_returns_retained));
id<MTLComputePipelineState> rejectPipeline(
        id self,
        SEL selector,
        id<MTLFunction> function,
        NSError** error) {
    if ([function.name isEqualToString:@"hamming_distance_topk"]) {
        return nil;
    }
    return reinterpret_cast<CreatePipeline*>(original)(
            self, selector, function, error);
}
size_t allocationCount = 0;
size_t allocatedBytes = 0;
bool rejectAllocation = false;
size_t rejectAt = 0;
typedef id<MTLBuffer> Allocate(id, SEL, NSUInteger, MTLResourceOptions)
        __attribute__((ns_returns_retained));
id<MTLBuffer> trackAllocation(
        id self,
        SEL selector,
        NSUInteger size,
        MTLResourceOptions options) __attribute__((ns_returns_retained));
id<MTLBuffer> trackAllocation(
        id self,
        SEL selector,
        NSUInteger size,
        MTLResourceOptions options) {
    ++allocationCount;
    allocatedBytes += size;
    if (rejectAllocation || (rejectAt && allocationCount == rejectAt)) {
        return nil;
    }
    return reinterpret_cast<Allocate*>(original)(self, selector, size, options);
}

bool checkGrowth(std::shared_ptr<MetalResources> resources) {
    auto device = resources->getDevice();
    Method method = class_getInstanceMethod(
            [device class], @selector(newBufferWithLength:options:));
    original = method_setImplementation(
            method, reinterpret_cast<IMP>(trackAllocation));
    MetalIndexBinaryFlat index(resources, 24);
    constexpr int n = 1025;
    for (int i = 0; i < n; ++i) {
        uint8_t code[] = {uint8_t(i), uint8_t(i >> 8), 0};
        index.add(1, code);
    }
    // Bound work rather than a particular initial capacity or growth factor.
    bool valid = allocationCount <= 16 && allocatedBytes <= 4 * n * 3;
    rejectAllocation = true;
    std::vector<uint8_t> extra(1024 * 3);
    bool caught = false;
    try {
        index.add(1024, extra.data()); // exceeds the geometrically grown buffer
    } catch (const std::exception&) {
        caught = true;
    }
    valid = valid && caught && index.ntotal == n;
    for (int i = 0; i < n; ++i) {
        uint8_t actual[3];
        index.reconstruct(i, actual);
        valid = valid && actual[0] == uint8_t(i) &&
                actual[1] == uint8_t(i >> 8) && actual[2] == 0;
    }
    // An append within spare capacity must succeed even if GPU allocations
    // are rejected. It must not recopy the existing prefix.
    size_t before = allocationCount;
    uint8_t code[] = {1, 4, 0};
    index.add(1, code);
    valid = valid && allocationCount == before && index.ntotal == n + 1;
    rejectAllocation = false;
    method_setImplementation(method, original);
    int32_t distance = -1;
    faiss::idx_t label = -1;
    index.search(1, code, 1, &distance, &label);
    return valid && distance == 0 && label == n;
}

bool checkScratch(std::shared_ptr<MetalResources> resources) {
    MetalIndexBinaryFlat index(resources, 24);
    std::vector<uint8_t> codes(257 * 3, 0);
    index.add(257, codes.data());
    constexpr int nq = 7, k = 64;
    std::vector<uint8_t> queries(513 * 3, 0);
    std::vector<int32_t> distances(513 * k, -123);
    std::vector<faiss::idx_t> labels(513 * k, -123);
    index.search(nq, queries.data(), k, distances.data(), labels.data());
    auto device = resources->getDevice();
    Method method = class_getInstanceMethod(
            [device class], @selector(newBufferWithLength:options:));
    original = method_setImplementation(
            method, reinterpret_cast<IMP>(trackAllocation));
    bool valid = true;
    // Every individual allocation in a larger search may fail. None may
    // publish a partial replacement or write any caller output.
    for (size_t failure : {1u, 2u, 3u}) {
        std::fill(distances.begin(), distances.end(), -123);
        std::fill(labels.begin(), labels.end(), -123);
        allocationCount = 0;
        rejectAt = failure;
        bool caught = false;
        try {
            index.search(
                    513, queries.data(), k, distances.data(), labels.data());
        } catch (const std::exception&) {
            caught = true;
        }
        valid = valid && caught && allocationCount == failure &&
                std::all_of(
                        distances.begin(),
                        distances.end(),
                        [](int32_t value) { return value == -123; }) &&
                std::all_of(
                        labels.begin(), labels.end(), [](faiss::idx_t value) {
                            return value == -123;
                        });
        rejectAt = 0;
        rejectAllocation = true;
        allocationCount = 0;
        index.search(nq, queries.data(), k, distances.data(), labels.data());
        valid = valid && allocationCount == 0;
        for (int q = 0; q < nq; ++q) {
            for (int j = 0; j < k; ++j) {
                valid = valid && distances[q * k + j] == 0 &&
                        labels[q * k + j] == j;
            }
        }
        rejectAllocation = false;
    }
    method_setImplementation(method, original);
    index.reset();
    index.add(257, codes.data());
    index.search(nq, queries.data(), k, distances.data(), labels.data());
    return valid && index.ntotal == 257;
}

} // namespace

// Run each fault in a fresh process: pipeline caches must start empty and Metal
// validation must be enabled before device creation. Returning successfully
// after draining the autorelease pool proves exception cleanup does not abort.
int main(int argc, char** argv) {
    bool caught = false;
    @autoreleasepool {
        auto resources = std::make_shared<MetalResources>();
        if (!resources->isAvailable()) {
            return 77;
        }
        if (argc == 2 && std::strcmp(argv[1], "growth") == 0) {
            return checkGrowth(resources) ? 0 : 1;
        }
        if (argc == 2 && std::strcmp(argv[1], "scratch") == 0) {
            return checkScratch(resources) ? 0 : 1;
        }
        auto device = resources->getDevice();
        Method method;
        IMP replacement;
        if (argc == 2 && std::strcmp(argv[1], "function") == 0) {
            NSError* error = nil;
            auto library = [device
                    newLibraryWithURL:
                            [NSURL fileURLWithPath:@(FAISS_TEST_METALLIB_PATH)]
                                error:&error];
            if (!library) {
                return 1;
            }
            method = class_getInstanceMethod(
                    [library class], @selector(newFunctionWithName:));
            replacement = reinterpret_cast<IMP>(rejectFunction);
        } else if (argc == 2 && std::strcmp(argv[1], "pipeline") == 0) {
            method = class_getInstanceMethod(
                    [device class],
                    @selector(newComputePipelineStateWithFunction:error:));
            replacement = reinterpret_cast<IMP>(rejectPipeline);
        } else {
            return 1;
        }
        if (!method) {
            return 1;
        }
        original = method_setImplementation(method, replacement);
        {
            MetalIndexBinaryFlat index(resources, 8);
            uint8_t code = 42;
            int32_t distance = -123;
            faiss::idx_t label = -123;
            index.add(1, &code);
            try {
                index.search(1, &code, 1, &distance, &label);
            } catch (const std::exception& e) {
                caught = std::strstr(
                                 e.what(),
                                 "Metal Hamming pipeline unavailable") &&
                        distance == -123 && label == -123;
            }
        }
        method_setImplementation(method, original);
    }
    if (!caught) {
        std::cerr
                << "Expected a catchable pipeline failure and unchanged outputs\n";
        return 1;
    }
    return 0;
}
