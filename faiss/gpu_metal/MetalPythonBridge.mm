// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalPythonBridge.h"
#include <faiss/impl/FaissAssert.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#import "MetalCloner.h"
#import "MetalResources.h"
#import "StandardMetalResources.h"

namespace faiss {
namespace gpu_metal {

namespace {
std::atomic<int> gMetalProfilerDepth{0};
} // namespace

StandardMetalResourcesHolder::StandardMetalResourcesHolder() {
    impl = new StandardMetalResources();
}

StandardMetalResourcesHolder::~StandardMetalResourcesHolder() {
    delete static_cast<StandardMetalResources*>(impl);
    impl = nullptr;
}

void gpu_profiler_start() {
    const int depth = gMetalProfilerDepth.fetch_add(1) + 1;
    if (std::getenv("FAISS_METAL_PROFILE_LOG")) {
        std::fprintf(
                stderr, "[faiss_metal] gpu_profiler_start depth=%d\n", depth);
    }
}

void gpu_profiler_stop() {
    int depth = gMetalProfilerDepth.load();
    while (depth > 0 &&
           !gMetalProfilerDepth.compare_exchange_weak(depth, depth - 1)) {
    }
    if (std::getenv("FAISS_METAL_PROFILE_LOG")) {
        std::fprintf(
                stderr,
                "[faiss_metal] gpu_profiler_stop depth=%d\n",
                gMetalProfilerDepth.load());
    }
}

void gpu_sync_all_devices() {
    auto res = std::make_shared<MetalResources>();
    if (res && res->isAvailable()) {
        res->synchronize();
    }
}

faiss::Index* index_cpu_to_gpu(
        StandardMetalResourcesHolder* res,
        int device,
        const faiss::Index* index) {
    if (!res || !res->impl) {
        return nullptr;
    }
    return index_cpu_to_metal_gpu(
            static_cast<StandardMetalResources*>(res->impl), device, index);
}

faiss::Index* index_gpu_to_cpu(const faiss::Index* index) {
    return index_metal_gpu_to_cpu(index);
}

} // namespace gpu_metal
} // namespace faiss
