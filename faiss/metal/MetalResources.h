#pragma once

#include <Metal/Metal.h>
#include <faiss/impl/FaissAssert.h>
#include <memory>
#include <vector>

namespace faiss {
namespace metal {

struct DistanceLabel {
    float distance;
    int label;
};

class MetalResources {
   public:
    virtual ~MetalResources();

    virtual void initializeForDevice(int device) = 0;

    virtual id<MTLDevice> getDevice(int device) = 0;

    virtual id<MTLCommandQueue> getCommandQueue(int device) = 0;

    virtual void* allocMemory(size_t size) = 0;

    virtual void deallocMemory(void* ptr) = 0;
};

class MetalResourcesProvider {
   public:
    virtual ~MetalResourcesProvider();

    virtual std::shared_ptr<MetalResources> getResources() = 0;
};

class MetalKernels {
    public:
    MetalKernels(std::shared_ptr<MetalResources> resources);

    void addVectors(id<MTLBuffer> inA, id<MTLBuffer> inB, id<MTLBuffer> out, int n);

    void l2Distance(
            id<MTLBuffer> query,
            id<MTLBuffer> data,
            id<MTLBuffer> dist_labels,
            int d,
            int n);

    void bitonicSort(id<MTLBuffer> data, int size);

    private:
    std::shared_ptr<MetalResources> resources_;
    id<MTLLibrary> library_;
};

} // namespace metal
} // namespace faiss
