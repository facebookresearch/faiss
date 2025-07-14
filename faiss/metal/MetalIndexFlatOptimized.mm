#include "MetalIndexFlat.h"
#include <faiss/impl/FaissAssert.h>
#include <vector>

namespace faiss {
namespace metal {

// Optimized search with batched async operations
void MetalIndexFlat::searchOptimized(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (n == 0 || ntotal_ == 0) {
        return;
    }

    MetalKernels kernels(resources_);
    id<MTLDevice> device = resources_->getDevice(0);
    
    // Batch size for parallel processing
    const int batch_size = 32;
    const int num_batches = (n + batch_size - 1) / batch_size;
    
    // Process queries in batches
    for (int batch = 0; batch < num_batches; ++batch) {
        int batch_start = batch * batch_size;
        int batch_end = std::min(batch_start + batch_size, (int)n);
        int batch_queries = batch_end - batch_start;
        
        // Create command buffers for parallel execution
        std::vector<id<MTLCommandBuffer>> distanceCommands;
        std::vector<id<MTLCommandBuffer>> topKCommands;
        std::vector<id<MTLBuffer>> queryBuffers;
        std::vector<id<MTLBuffer>> distLabelsBuffers;
        std::vector<id<MTLBuffer>> outDistancesBuffers;
        std::vector<id<MTLBuffer>> outLabelsBuffers;
        
        // Reserve space
        distanceCommands.reserve(batch_queries);
        topKCommands.reserve(batch_queries);
        queryBuffers.reserve(batch_queries);
        distLabelsBuffers.reserve(batch_queries);
        outDistancesBuffers.reserve(batch_queries);
        outLabelsBuffers.reserve(batch_queries);
        
        // Launch distance computations asynchronously
        for (int i = 0; i < batch_queries; ++i) {
            idx_t q = batch_start + i;
            
            // Create buffers
            id<MTLBuffer> query_buffer = [device newBufferWithBytes:x + q * d
                                                          length:d * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            
            id<MTLBuffer> dist_labels_buffer = [device newBufferWithLength:ntotal_ * sizeof(DistanceLabel)
                                                                    options:MTLResourceStorageModeShared];
            
            queryBuffers.push_back(query_buffer);
            distLabelsBuffers.push_back(dist_labels_buffer);
            
            // Launch async distance computation
            id<MTLCommandBuffer> cmd;
            if (metric_type == METRIC_L2) {
                cmd = kernels.l2DistanceAsync(query_buffer, vectors_, dist_labels_buffer, d, ntotal_);
            } else if (metric_type == METRIC_INNER_PRODUCT) {
                cmd = kernels.innerProductDistanceAsync(query_buffer, vectors_, dist_labels_buffer, d, ntotal_);
            } else {
                FAISS_THROW_MSG("Metric not supported");
            }
            
            distanceCommands.push_back(cmd);
        }
        
        // Wait for distance computations to complete
        for (auto& cmd : distanceCommands) {
            [cmd waitUntilCompleted];
        }
        
        // Launch top-k selections asynchronously
        for (int i = 0; i < batch_queries; ++i) {
            idx_t q = batch_start + i;
            
            id<MTLBuffer> out_distances_buffer = [device newBufferWithLength:k * sizeof(float)
                                                                      options:MTLResourceStorageModeShared];
            
            id<MTLBuffer> out_labels_buffer = [device newBufferWithLength:k * sizeof(idx_t)
                                                                  options:MTLResourceStorageModeShared];
            
            outDistancesBuffers.push_back(out_distances_buffer);
            outLabelsBuffers.push_back(out_labels_buffer);
            
            // Launch async top-k selection
            id<MTLCommandBuffer> cmd = kernels.selectTopKAsync(
                    distLabelsBuffers[i], out_distances_buffer, out_labels_buffer, 
                    ntotal_, k, q);
            
            topKCommands.push_back(cmd);
        }
        
        // Wait for all top-k selections to complete
        for (auto& cmd : topKCommands) {
            [cmd waitUntilCompleted];
        }
        
        // Copy results back
        for (int i = 0; i < batch_queries; ++i) {
            idx_t q = batch_start + i;
            memcpy(distances + q * k, [outDistancesBuffers[i] contents], k * sizeof(float));
            memcpy(labels + q * k, [outLabelsBuffers[i] contents], k * sizeof(idx_t));
        }
    }
}

} // namespace metal
} // namespace faiss