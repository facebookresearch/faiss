#include "MetalIndexHNSW.h"
#include <faiss/IndexFlat.h>
#include <faiss/impl/HNSW.h>
#include <vector>
#include <iostream>

namespace faiss {
namespace metal {

MetalIndexHNSW::MetalIndexHNSW(
        std::shared_ptr<MetalResources> resources,
        int d,
        int M,
        faiss::MetricType metric)
    : faiss::IndexHNSW(
          metric == faiss::METRIC_L2 
              ? static_cast<faiss::Index*>(new faiss::IndexFlatL2(d))
              : static_cast<faiss::Index*>(new faiss::IndexFlatIP(d)), 
          M),
      resources_(resources),
      is_gpu_sync_(false) {
    own_fields = true;
    this->metric_type = metric;
}

void MetalIndexHNSW::sync_with_gpu() {
    if (is_gpu_sync_) {
        return;
    }

    id<MTLDevice> device = resources_->getDevice(0);

    // Copy vectors from storage
    const IndexFlat* flat_storage = dynamic_cast<const IndexFlat*>(storage);
    if (flat_storage) {
        vectors_ = [device newBufferWithBytes:flat_storage->get_xb()
                                       length:ntotal * d * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    }

    // Copy levels
    levels_ = [device newBufferWithBytes:hnsw.levels.data()
                                 length:hnsw.levels.size() * sizeof(int)
                                options:MTLResourceStorageModeShared];

    // Prepare graph data in a more GPU-friendly format
    // Create offsets array for neighbor lists
    std::vector<int> graph_offsets;
    std::vector<int> graph_neighbors;
    
    graph_offsets.reserve(ntotal + 1);
    int offset = 0;
    
    for (size_t i = 0; i < ntotal; i++) {
        graph_offsets.push_back(offset);
        
        // Add neighbors for all levels
        for (int level = 0; level <= hnsw.levels[i]; level++) {
            size_t begin, end;
            hnsw.neighbor_range(i, level, &begin, &end);
            
            for (size_t j = begin; j < end; j++) {
                if (hnsw.neighbors[j] >= 0) {
                    graph_neighbors.push_back(hnsw.neighbors[j]);
                    offset++;
                }
            }
        }
    }
    graph_offsets.push_back(offset);
    
    // Create Metal buffers
    graph_offsets_ = [device newBufferWithBytes:graph_offsets.data()
                                         length:graph_offsets.size() * sizeof(int)
                                        options:MTLResourceStorageModeShared];
    
    graph_neighbors_ = [device newBufferWithBytes:graph_neighbors.data()
                                           length:graph_neighbors.size() * sizeof(int)
                                          options:MTLResourceStorageModeShared];

    is_gpu_sync_ = true;
}

void MetalIndexHNSW::add(idx_t n, const float* x) {
    // For now, we use the CPU-based implementation for graph construction.
    std::cout << "MetalIndexHNSW::add called with n=" << n << std::endl;
    faiss::IndexHNSW::add(n, x);
    is_gpu_sync_ = false;
    std::cout << "MetalIndexHNSW::add completed" << std::endl;
}

void MetalIndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (ntotal == 0) {
        return;
    }
    
    const_cast<MetalIndexHNSW*>(this)->sync_with_gpu();

    MetalKernels kernels(resources_);
    id<MTLDevice> device = resources_->getDevice(0);
    
    // Get search parameters
    int ef = hnsw.efSearch;
    if (params) {
        const SearchParametersHNSW* params_hnsw = 
            dynamic_cast<const SearchParametersHNSW*>(params);
        if (params_hnsw) {
            ef = params_hnsw->efSearch;
        }
    }
    
    // Process queries one at a time for now (GPU kernel limitation)
    for (idx_t q = 0; q < n; q++) {
        id<MTLBuffer> query_buffer = [device newBufferWithBytes:x + q * d
                                                      length:d * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

        id<MTLBuffer> out_distances_buffer = [device newBufferWithLength:k * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_labels_buffer = [device newBufferWithLength:k * sizeof(idx_t)
                                                              options:MTLResourceStorageModeShared];

        // Get max neighbors per level (2*M for level 0, M for higher levels)
        int max_neighbors = hnsw.nb_neighbors(0);
        
        kernels.hnsw_search(
                query_buffer,
                vectors_,
                levels_,
                graph_offsets_,
                graph_neighbors_,
                d,
                k,
                ef,
                max_neighbors,
                ntotal,
                out_distances_buffer,
                out_labels_buffer,
                hnsw.entry_point);

        memcpy(distances + q * k, [out_distances_buffer contents], k * sizeof(float));
        memcpy(labels + q * k, [out_labels_buffer contents], k * sizeof(idx_t));
    }
}

} // namespace metal
} // namespace faiss
