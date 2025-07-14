#include "MetalResources.h"
#include "MetalDataTypes.h"
#include <algorithm>

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
    
    // Try to load the kernels metallib file
    NSString* libraryPath = @"build/faiss/metal/kernels.metallib";
    NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
    library_ = [device newLibraryWithURL:libraryURL error:&error];
    
    if (!library_) {
        // Try relative path
        libraryPath = @"faiss/metal/kernels.metallib";
        libraryURL = [NSURL fileURLWithPath:libraryPath];
        library_ = [device newLibraryWithURL:libraryURL error:&error];
    }
    
    if (!library_) {
        // Try loading from default library as fallback
        library_ = [device newDefaultLibrary];
    }
    
    FAISS_THROW_IF_NOT_FMT(library_, "Could not create Metal library, error: %s", 
                           error ? [[error description] UTF8String] : "Unknown error");
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

id<MTLCommandBuffer> MetalKernels::l2DistanceAsync(
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
    [commandEncoder setBytes:&n length:sizeof(int) atIndex:4];

    MTLSize gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > n) {
        threadGroupSize = n;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    
    return commandBuffer;
}

void MetalKernels::l2Distance(
        id<MTLBuffer> query,
        id<MTLBuffer> data,
        id<MTLBuffer> dist_labels,
        int d,
        int n) {
    id<MTLCommandBuffer> commandBuffer = l2DistanceAsync(query, data, dist_labels, d, n);
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

void MetalKernels::hnsw_search(
        id<MTLBuffer> queries,
        id<MTLBuffer> db_vectors,
        id<MTLBuffer> levels,
        id<MTLBuffer> graph_offsets,
        id<MTLBuffer> graph_neighbors,
        int d,
        int k,
        int ef,
        int M,
        int nb,
        id<MTLBuffer> out_distances,
        id<MTLBuffer> out_labels,
        int entry_point) {
    NSError* error = nil;
    id<MTLFunction> searchFunction = [library_ newFunctionWithName:@"hnsw_search"];
    FAISS_THROW_IF_NOT_MSG(searchFunction, "Could not create Metal function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:searchFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:queries offset:0 atIndex:0];
    [commandEncoder setBuffer:db_vectors offset:0 atIndex:1];
    [commandEncoder setBuffer:levels offset:0 atIndex:2];
    [commandEncoder setBuffer:graph_offsets offset:0 atIndex:3];
    [commandEncoder setBuffer:graph_neighbors offset:0 atIndex:4];
    [commandEncoder setBytes:&d length:sizeof(int) atIndex:5];
    [commandEncoder setBytes:&k length:sizeof(int) atIndex:6];
    [commandEncoder setBytes:&ef length:sizeof(int) atIndex:7];
    [commandEncoder setBytes:&M length:sizeof(int) atIndex:8];
    [commandEncoder setBytes:&nb length:sizeof(int) atIndex:9];
    [commandEncoder setBuffer:out_distances offset:0 atIndex:10];
    [commandEncoder setBuffer:out_labels offset:0 atIndex:11];
    [commandEncoder setBytes:&entry_point length:sizeof(int) atIndex:12];

    MTLSize gridSize = MTLSizeMake(1, 1, 1);  // Process one query at a time
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

id<MTLCommandBuffer> MetalKernels::innerProductDistanceAsync(
        id<MTLBuffer> query,
        id<MTLBuffer> data,
        id<MTLBuffer> dist_labels,
        int d,
        int n) {
    NSError* error = nil;
    id<MTLFunction> ipFunction = [library_ newFunctionWithName:@"inner_product_distance"];
    FAISS_THROW_IF_NOT_MSG(ipFunction, "Could not create inner_product_distance function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:ipFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:query offset:0 atIndex:0];
    [commandEncoder setBuffer:data offset:0 atIndex:1];
    [commandEncoder setBuffer:dist_labels offset:0 atIndex:2];
    [commandEncoder setBytes:&d length:sizeof(int) atIndex:3];
    [commandEncoder setBytes:&n length:sizeof(int) atIndex:4];

    MTLSize gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > n) {
        threadGroupSize = n;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    
    return commandBuffer;
}

void MetalKernels::innerProductDistance(
        id<MTLBuffer> query,
        id<MTLBuffer> data,
        id<MTLBuffer> dist_labels,
        int d,
        int n) {
    id<MTLCommandBuffer> commandBuffer = innerProductDistanceAsync(query, data, dist_labels, d, n);
    [commandBuffer waitUntilCompleted];
}

id<MTLCommandBuffer> MetalKernels::selectTopKAsync(
        id<MTLBuffer> distances,
        id<MTLBuffer> out_distances,
        id<MTLBuffer> out_labels,
        int n,
        int k,
        int query_id) {
    NSError* error = nil;
    id<MTLFunction> topKFunction = [library_ newFunctionWithName:@"select_top_k"];
    FAISS_THROW_IF_NOT_MSG(topKFunction, "Could not create select_top_k function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:topKFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:distances offset:0 atIndex:0];
    [commandEncoder setBuffer:out_distances offset:0 atIndex:1];
    [commandEncoder setBuffer:out_labels offset:0 atIndex:2];
    [commandEncoder setBytes:&n length:sizeof(int) atIndex:3];
    [commandEncoder setBytes:&k length:sizeof(int) atIndex:4];
    [commandEncoder setBytes:&query_id length:sizeof(int) atIndex:5];

    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    
    return commandBuffer;
}

void MetalKernels::selectTopK(
        id<MTLBuffer> distances,
        id<MTLBuffer> out_distances,
        id<MTLBuffer> out_labels,
        int n,
        int k,
        int query_id) {
    id<MTLCommandBuffer> commandBuffer = selectTopKAsync(distances, out_distances, out_labels, n, k, query_id);
    [commandBuffer waitUntilCompleted];
}

void MetalKernels::l2DistanceBatch(
        id<MTLBuffer> queries,
        id<MTLBuffer> data,
        id<MTLBuffer> distances,
        int d,
        int n,
        int nq) {
    NSError* error = nil;
    id<MTLFunction> batchFunction = [library_ newFunctionWithName:@"l2_distance_batch"];
    FAISS_THROW_IF_NOT_MSG(batchFunction, "Could not create l2_distance_batch function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:batchFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:queries offset:0 atIndex:0];
    [commandEncoder setBuffer:data offset:0 atIndex:1];
    [commandEncoder setBuffer:distances offset:0 atIndex:2];
    [commandEncoder setBytes:&d length:sizeof(int) atIndex:3];
    [commandEncoder setBytes:&n length:sizeof(int) atIndex:4];
    [commandEncoder setBytes:&nq length:sizeof(int) atIndex:5];

    // 2D grid for queries x database vectors
    MTLSize gridSize = MTLSizeMake(nq, n, 1);
    NSUInteger maxThreads = [pipelineState maxTotalThreadsPerThreadgroup];
    
    // Find optimal threadgroup size
    NSUInteger threadsX = 1;
    NSUInteger threadsY = maxThreads;
    
    // Try to balance the threadgroup dimensions
    if (nq < maxThreads && n < maxThreads) {
        threadsX = nq;
        threadsY = std::min((NSUInteger)n, maxThreads / threadsX);
    } else if (nq >= maxThreads) {
        threadsX = maxThreads;
        threadsY = 1;
    }
    
    MTLSize threadgroupSize = MTLSizeMake(threadsX, threadsY, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernels::innerProductDistanceBatch(
        id<MTLBuffer> queries,
        id<MTLBuffer> data,
        id<MTLBuffer> distances,
        int d,
        int n,
        int nq) {
    NSError* error = nil;
    id<MTLFunction> batchFunction = [library_ newFunctionWithName:@"inner_product_batch"];
    FAISS_THROW_IF_NOT_MSG(batchFunction, "Could not create inner_product_batch function");

    id<MTLComputePipelineState> pipelineState = [resources_->getDevice(0) newComputePipelineStateWithFunction:batchFunction error:&error];
    FAISS_THROW_IF_NOT_FMT(pipelineState, "Failed to create compute pipeline state, error %s", [[error description] UTF8String]);

    id<MTLCommandQueue> commandQueue = resources_->getCommandQueue(0);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:queries offset:0 atIndex:0];
    [commandEncoder setBuffer:data offset:0 atIndex:1];
    [commandEncoder setBuffer:distances offset:0 atIndex:2];
    [commandEncoder setBytes:&d length:sizeof(int) atIndex:3];
    [commandEncoder setBytes:&n length:sizeof(int) atIndex:4];
    [commandEncoder setBytes:&nq length:sizeof(int) atIndex:5];

    // 2D grid for queries x database vectors
    MTLSize gridSize = MTLSizeMake(nq, n, 1);
    NSUInteger maxThreads = [pipelineState maxTotalThreadsPerThreadgroup];
    
    // Find optimal threadgroup size
    NSUInteger threadsX = 1;
    NSUInteger threadsY = maxThreads;
    
    // Try to balance the threadgroup dimensions
    if (nq < maxThreads && n < maxThreads) {
        threadsX = nq;
        threadsY = std::min((NSUInteger)n, maxThreads / threadsX);
    } else if (nq >= maxThreads) {
        threadsX = maxThreads;
        threadsY = 1;
    }
    
    MTLSize threadgroupSize = MTLSizeMake(threadsX, threadsY, 1);

    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

} // namespace metal
} // namespace faiss
