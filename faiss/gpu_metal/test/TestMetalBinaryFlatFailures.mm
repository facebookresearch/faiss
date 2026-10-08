// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu_metal/MetalIndexBinaryFlat.h>
#import <objc/runtime.h>
#include <cstring>
#include <iostream>

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
