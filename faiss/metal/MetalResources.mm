#include "MetalResources.h"

namespace faiss {
namespace metal {

MetalResources::~MetalResources() = default;
MetalResourcesProvider::~MetalResourcesProvider() = default;

class StandardMetalResources : public MetalResources {
   public:
    StandardMetalResources() {
        device_ = MTLCreateSystemDefaultDevice();
        FAISS_THROW_IF_NOT_MSG(device_, "Metal is not supported on this device");
        commandQueue_ = [device_ newCommandQueue];
    }

    void initializeForDevice(int device) override {
        // For now, we only support the default device
    }

    id<MTLDevice> getDevice(int device) override {
        return device_;
    }

    id<MTLCommandQueue> getCommandQueue(int device) override {
        return commandQueue_;
    }

    void* allocMemory(size_t size) override {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
        return (__bridge_retained void*)buffer;
    }

    void deallocMemory(void* ptr) override {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)ptr;
        // ARC will handle the release
    }

   private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> commandQueue_ = nil;
};

std::shared_ptr<MetalResources> get_default_metal_resources() {
    static std::shared_ptr<MetalResources> resources = std::make_shared<StandardMetalResources>();
    return resources;
}

MetalKernels::MetalKernels(std::shared_ptr<MetalResources> resources)
    : resources_(resources) {
    NSError* error = nil;
    id<MTLDevice> device = resources->getDevice(0);
    library_ = [device newDefaultLibrary];
    FAISS_THROW_IF_NOT_MSG(library_, "Could not create default Metal library");
}

void MetalKernels::addVectors(id<MTLBuffer> inA, id<MTLBuffer> inB, id<MTLBuffer> out, int n) {
    NSError* error = nil;
    id<MTLFunction> addFunction = [library_ newFunctionWithName:@"add_vectors"];
    FAISS_THROW_IF_NOT_MSG(addFunction, "Could not create Metal function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:addFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:inA offset:0 atIndex:0];
    [commandEncoder setBuffer:inB offset:0 atIndex:1];
    [commandEncoder setBuffer:out offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > n) {
        threadGroupSize = n;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernels::l2Distance(
        id<MTLBuffer> query,
        id<MTLBuffer> data,
        id<MTLBuffer> distances,
        int d,
        int n) {
    NSError* error = nil;
    id<MTLFunction> l2Function = [library_ newFunctionWithName:@"l2_distance"];
    FAISS_THROW_IF_NOT_MSG(l2Function, "Could not create Metal function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:l2Function error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:query offset:0 atIndex:0];
    [commandEncoder setBuffer:data offset:0 atIndex:1];
    [commandEncoder setBuffer:distances offset:0 atIndex:2];
    [commandEncoder setBytes:&d length:sizeof(int) atIndex:3];

    MTLSize gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > n) {
        threadGroupSize = n;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

} // namespace metal
} // namespace faiss
