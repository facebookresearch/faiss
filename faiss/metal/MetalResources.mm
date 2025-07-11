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
        id<MTLBuffer> dist_labels,
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
    [commandEncoder setBuffer:dist_labels offset:0 atIndex:2];
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

void MetalKernels::bitonicSort(id<MTLBuffer> data, int size) {
    NSError* error = nil;
    id<MTLFunction> sortFunction = [library_ newFunctionWithName:@"bitonic_sort"];
    FAISS_THROW_IF_NOT_MSG(sortFunction, "Could not create Metal function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:sortFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:data offset:0 atIndex:0];

    uint num_stages = 0;
    for (uint temp = size; temp > 1; temp >>= 1) {
        num_stages++;
    }

    for (uint stage = 0; stage < num_stages; stage++) {
        for (uint pass_of_stage = 0; pass_of_stage < stage + 1; pass_of_stage++) {
            uint direction = (pass_of_stage == stage) ? 1 : 0;
            [commandEncoder setBytes:&size length:sizeof(uint) atIndex:1];
            [commandEncoder setBytes:&stage length:sizeof(uint) atIndex:2];
            [commandEncoder setBytes:&pass_of_stage length:sizeof(uint) atIndex:3];
            [commandEncoder setBytes:&direction length:sizeof(uint) atIndex:4];

            MTLSize gridSize = MTLSizeMake(size / 2, 1, 1);
            NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
            if (threadGroupSize > size/2) {
                threadGroupSize = size/2;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        }
    }

    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernels::ivfflat_scan_per_query(
        id<MTLBuffer> queries,
        id<MTLBuffer> db_vectors,
        id<MTLBuffer> db_ids,
        id<MTLBuffer> list_offsets,
        id<MTLBuffer> coarse_assign,
        int nprobe,
        int d,
        int k,
        id<MTLBuffer> out_distances,
        id<MTLBuffer> out_labels,
        int nq) {
    NSError* error = nil;
    id<MTLFunction> scanFunction = [library_ newFunctionWithName:@"ivfflat_scan_per_query"];
    FAISS_THROW_IF_NOT_MSG(scanFunction, "Could not create Metal function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:scanFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:queries offset:0 atIndex:0];
    [commandEncoder setBuffer:db_vectors offset:0 atIndex:1];
    [commandEncoder setBuffer:db_ids offset:0 atIndex:2];
    [commandEncoder setBuffer:list_offsets offset:0 atIndex:3];
    [commandEncoder setBuffer:coarse_assign offset:0 atIndex:4];
    [commandEncoder setBytes:&nprobe length:sizeof(int) atIndex:5];
    [commandEncoder setBytes:&d length:sizeof(int) atIndex:6];
    [commandEncoder setBytes:&k length:sizeof(int) atIndex:7];
    [commandEncoder setBuffer:out_distances offset:0 atIndex:8];
    [commandEncoder setBuffer:out_labels offset:0 atIndex:9];

    MTLSize gridSize = MTLSizeMake(nq, 1, 1);
    NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > nq) {
        threadGroupSize = nq;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernels::ivfpq_scan_per_query(
        id<MTLBuffer> queries,
        id<MTLBuffer> db_codes,
        id<MTLBuffer> db_ids,
        id<MTLBuffer> list_offsets,
        id<MTLBuffer> coarse_assign,
        id<MTLBuffer> dist_tables,
        int nprobe,
        int d,
        int k,
        int M,
        int ksub,
        id<MTLBuffer> out_distances,
        id<MTLBuffer> out_labels,
        int nq) {
    NSError* error = nil;
    id<MTLFunction> scanFunction = [library_ newFunctionWithName:@"ivfpq_scan_per_query"];
    FAISS_THROW_IF_NOT_MSG(scanFunction, "Could not create Metal function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:scanFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:queries offset:0 atIndex:0];
    [commandEncoder setBuffer:db_codes offset:0 atIndex:1];
    [commandEncoder setBuffer:db_ids offset:0 atIndex:2];
    [commandEncoder setBuffer:list_offsets offset:0 atIndex:3];
    [commandEncoder setBuffer:coarse_assign offset:0 atIndex:4];
    [commandEncoder setBuffer:dist_tables offset:0 atIndex:5];
    [commandEncoder setBytes:&nprobe length:sizeof(int) atIndex:6];
    [commandEncoder setBytes:&d length:sizeof(int) atIndex:7];
    [commandEncoder setBytes:&k length:sizeof(int) atIndex:8];
    [commandEncoder setBytes:&M length:sizeof(int) atIndex:9];
    [commandEncoder setBytes:&ksub length:sizeof(int) atIndex:10];
    [commandEncoder setBuffer:out_distances offset:0 atIndex:11];
    [commandEncoder setBuffer:out_labels offset:0 atIndex:12];

    MTLSize gridSize = MTLSizeMake(nq, 1, 1);
    NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > nq) {
        threadGroupSize = nq;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

} // namespace metal
} // namespace faiss
