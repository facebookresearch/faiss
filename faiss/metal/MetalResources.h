#pragma once

#include <Metal/Metal.h>
#include <faiss/impl/FaissAssert.h>
#include <memory>
#include <vector>

namespace faiss {
namespace metal {

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
    
    // Synchronous operations (wait for completion)
    void addVectors(id<MTLBuffer> inA, id<MTLBuffer> inB, id<MTLBuffer> out, int n);
    
    // Asynchronous operations (return command buffer for manual synchronization)
    id<MTLCommandBuffer> addVectorsAsync(id<MTLBuffer> inA, id<MTLBuffer> inB, id<MTLBuffer> out, int n);

    void l2Distance(
            id<MTLBuffer> query,
            id<MTLBuffer> data,
            id<MTLBuffer> dist_labels,
            int d,
            int n);
    
    void l2DistanceBatch(
            id<MTLBuffer> queries,
            id<MTLBuffer> data,
            id<MTLBuffer> distances,
            int d,
            int n,
            int nq);
    
    void innerProductDistanceBatch(
            id<MTLBuffer> queries,
            id<MTLBuffer> data,
            id<MTLBuffer> distances,
            int d,
            int n,
            int nq);
    
    id<MTLCommandBuffer> l2DistanceAsync(
            id<MTLBuffer> query,
            id<MTLBuffer> data,
            id<MTLBuffer> dist_labels,
            int d,
            int n);
    
    void innerProductDistance(
            id<MTLBuffer> query,
            id<MTLBuffer> data,
            id<MTLBuffer> dist_labels,
            int d,
            int n);
    
    id<MTLCommandBuffer> innerProductDistanceAsync(
            id<MTLBuffer> query,
            id<MTLBuffer> data,
            id<MTLBuffer> dist_labels,
            int d,
            int n);

    void selectTopK(
            id<MTLBuffer> distances,
            id<MTLBuffer> out_distances,
            id<MTLBuffer> out_labels,
            int n,
            int k,
            int query_id);
    
    id<MTLCommandBuffer> selectTopKAsync(
            id<MTLBuffer> distances,
            id<MTLBuffer> out_distances,
            id<MTLBuffer> out_labels,
            int n,
            int k,
            int query_id);

    void bitonicSort(id<MTLBuffer> data, int size);

    void ivfflat_scan_per_query(
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
            int nq);

    void ivfpq_scan_per_query(
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
            int nq);

    void hnsw_search(
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
            int entry_point);

    private:
    std::shared_ptr<MetalResources> resources_;
    id<MTLLibrary> library_;
};

// Get the default Metal resources singleton
std::shared_ptr<MetalResources> get_default_metal_resources();

} // namespace metal
} // namespace faiss