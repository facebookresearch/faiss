// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/MetalIndexIVFScalarQuantizer.h>
#include <faiss/gpu_metal/MetalKernels.h>
#include <faiss/gpu_metal/MetalResources.h>
#import <objc/runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>
using namespace faiss;
using namespace faiss::gpu_metal;
namespace {
IMP original;
size_t allocations = 0, rejectAt = 0, allocatedBytes = 0;
typedef id<MTLBuffer> Allocate(id, SEL, NSUInteger, MTLResourceOptions)
        __attribute__((ns_returns_retained));
id<MTLBuffer> allocate(
        id self,
        SEL sel,
        NSUInteger size,
        MTLResourceOptions options) __attribute__((ns_returns_retained));
id<MTLBuffer> allocate(
        id self,
        SEL sel,
        NSUInteger size,
        MTLResourceOptions options) {
    allocatedBytes += size;
    if (++allocations == rejectAt)
        return nil;
    return reinterpret_cast<Allocate*>(original)(self, sel, size, options);
}
typedef id Lookup(id, SEL, NSString*) __attribute__((ns_returns_retained));
id lookup(id self, SEL sel, NSString* name)
        __attribute__((ns_returns_retained));
id lookup(id self, SEL sel, NSString* name) {
    if ([name isEqualToString:@"ivfsq_scan"])
        return nil;
    return reinterpret_cast<Lookup*>(original)(self, sel, name);
}
struct Restore {
    Method method;
    ~Restore() {
        method_setImplementation(method, original);
    }
};
template <class F>
bool throws(F action) {
    try {
        action();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}
} // namespace
int main(int argc, char** argv) {
    auto res = std::make_shared<MetalResources>();
    if (!res->isAvailable())
        return 77;
    auto device = res->getDevice();
    std::string mode = argc > 1 ? argv[1] : "scratch";
    if (mode == "codec-cache") {
        constexpr size_t dim = 128, lists = 1024, count = 10;
        IndexFlatL2 coarse(dim);
        std::vector<float> centroids(lists * dim);
        for (size_t l = 0; l < lists; ++l)
            centroids[l * dim] = float(l * 4);
        coarse.add(lists, centroids.data());
        IndexIVFScalarQuantizer cpu(
                &coarse, dim, lists, ScalarQuantizer::QT_8bit);
        cpu.sq.trained.assign(2 * dim, 0.f);
        std::fill(cpu.sq.trained.begin() + dim, cpu.sq.trained.end(), 1.f);
        cpu.is_trained = true;
        std::vector<float> x(17 * dim, 0.25f);
        cpu.add(17, x.data());
        MetalIndexIVFScalarQuantizer gpu(res, &cpu);
        gpu.add(1, x.data()); // establish spare per-list capacity
        Method method = class_getInstanceMethod(
                [device class], @selector(newBufferWithLength:options:));
        original = method_setImplementation(
                method, reinterpret_cast<IMP>(allocate));
        Restore restore{method};
        allocations = allocatedBytes = 0;
        for (size_t j = 0; j < count; ++j)
            gpu.add(1, x.data());
        // Only offsets and lengths may be replaced. Neither code storage nor
        // the 512 KiB centroid table needs growth or reconstruction.
        if (allocations != 2 * count ||
            allocatedBytes != 2 * count * lists * sizeof(uint32_t)) {
            std::cerr << "unexpected append allocations: " << allocations
                      << " buffers, " << allocatedBytes << " bytes\n";
            return 14;
        }
        std::unique_ptr<IndexIVFScalarQuantizer> back(gpu.copyTo());
        if (back->ntotal != 28)
            return 15;
        return 0;
    }
    IndexFlatL2 coarse(7);
    IndexIVFScalarQuantizer cpu(&coarse, 7, 1, ScalarQuantizer::QT_fp16);
    std::vector<float> x(1024 * 7);
    for (size_t j = 0; j < x.size(); ++j)
        x[j] = float(j % 61) / 8;
    cpu.train(1024, x.data());
    cpu.add(17, x.data());
    MetalIndexIVFScalarQuantizer gpu(res, &cpu);
    std::vector<float> d(1024 * 13, -123);
    std::vector<idx_t> i(1024 * 13, -123);
    if (mode == "pipeline") {
        NSError* error = nil;
        auto library = [device newLibraryWithFile:@FAISS_TEST_METALLIB_PATH
                                            error:&error];
        if (!library)
            return 2;
        Method method = class_getInstanceMethod(
                [library class], @selector(newFunctionWithName:));
        original =
                method_setImplementation(method, reinterpret_cast<IMP>(lookup));
        {
            Restore restore{method};
            if (!throws([&] {
                    gpu.search(1, x.data(), 13, d.data(), i.data());
                }))
                return 3;
        }
        gpu.search(1, x.data(), 13, d.data(), i.data());
        return gpu.gpuSearchCount() == 1 ? 0 : 4;
    }
    gpu.search(1, x.data(), 13, d.data(), i.data());
    Method method = class_getInstanceMethod(
            [device class], @selector(newBufferWithLength:options:));
    original =
            method_setImplementation(method, reinterpret_cast<IMP>(allocate));
    Restore restore{method};
    if (mode == "scratch") {
        for (size_t fail : {1, 2, 3, 4, 5}) {
            allocations = 0;
            rejectAt = fail;
            std::fill(d.begin(), d.end(), -123);
            std::fill(i.begin(), i.end(), -123);
            if (!throws([&] {
                    gpu.search(1024, x.data(), 13, d.data(), i.data());
                }))
                return 5;
            if (!std::all_of(
                        d.begin(),
                        d.end(),
                        [](float v) { return v == -123; }) ||
                !std::all_of(
                        i.begin(), i.end(), [](idx_t v) { return v == -123; }))
                return 6;
            rejectAt = 0;
            gpu.search(1, x.data(), 13, d.data(), i.data());
        }
        gpu.search(1024, x.data(), 13, d.data(), i.data());
    } else if (mode == "growth") {
        allocatedBytes = 0;
        for (int j = 0; j < 257; ++j)
            gpu.add(1, x.data() + j * 7);
        // Bound total allocated bytes; rebuilding the full prefix on every
        // one-vector append would exceed this by more than an order of
        // magnitude.
        if (allocatedBytes > 16 * 274 * (cpu.code_size + sizeof(idx_t)))
            return 13;
        gpu.search(1, x.data(), 13, d.data(), i.data());
    } else if (mode == "upload") {
        for (size_t fail : {1, 2, 3, 4}) {
            rejectAt = 0;
            gpu.copyFrom(&cpu);
            allocations = 0;
            rejectAt = fail;
            if (!throws([&] { gpu.add(1000, x.data()); }))
                return 7;
            if (!throws([&] {
                    gpu.search(1, x.data(), 1, d.data(), i.data());
                }))
                return 8;
            std::unique_ptr<IndexIVFScalarQuantizer> recovery(gpu.copyTo());
            if (recovery->ntotal != 1017)
                return 9;
            rejectAt = 0;
            gpu.reset();
            gpu.add(17, x.data());
            gpu.search(1, x.data(), 1, d.data(), i.data());
        }
    } else if (mode == "copy") {
        for (size_t fail : {1, 2, 3, 4, 5, 6}) {
            allocations = 0;
            rejectAt = fail;
            if (!throws([&] { gpu.copyFrom(&cpu); }))
                return 10;
            rejectAt = 0;
            gpu.search(1, x.data(), 13, d.data(), i.data());
            if (gpu.ntotal != 17)
                return 11;
        }
    } else
        return 12;
    return 0;
}
