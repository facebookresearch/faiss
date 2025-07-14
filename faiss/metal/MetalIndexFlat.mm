#include "MetalIndexFlat.h"
#include "MetalResources.h"
#include "MetalDataTypes.h"
#include <algorithm>
#include <vector>

namespace faiss {
namespace metal {

MetalIndexFlat::MetalIndexFlat(
        std::shared_ptr<MetalResources> resources,
        int d,
        faiss::MetricType metric)
    : faiss::IndexFlat(d, metric),
      resources_(resources),
      vectors_(nil),
      ntotal_(0) {}

void MetalIndexFlat::add(idx_t n, const float* x) {
    if (n == 0) {
        return;
    }

    size_t size = n * d * sizeof(float);
    id<MTLDevice> device = resources_->getDevice(0);

    if (vectors_ == nil) {
        vectors_ = [device newBufferWithBytes:x
                                       length:size
                                      options:MTLResourceStorageModeShared];
    } else {
        id<MTLBuffer> new_vectors = [device newBufferWithLength:[vectors_ length] + size
                                                         options:MTLResourceStorageModeShared];
        memcpy([new_vectors contents], [vectors_ contents], [vectors_ length]);
        memcpy((uint8_t*)[new_vectors contents] + [vectors_ length], x, size);
        vectors_ = new_vectors;
    }

    ntotal_ += n;
}

void MetalIndexFlat::search(
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
    
    // Use batched search for better performance
    const int batch_size = 32;  // Process queries in batches
    
    for (idx_t batch_start = 0; batch_start < n; batch_start += batch_size) {
        idx_t batch_end = (batch_start + batch_size < n) ? batch_start + batch_size : n;
        idx_t batch_queries = batch_end - batch_start;
        
        // Allocate buffers for batch
        id<MTLBuffer> queries_buffer = [device newBufferWithBytes:x + batch_start * d
                                                          length:batch_queries * d * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> all_distances_buffer = [device newBufferWithLength:batch_queries * ntotal_ * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        
        // Compute all distances for this batch
        if (metric_type == METRIC_L2) {
            kernels.l2DistanceBatch(queries_buffer, vectors_, all_distances_buffer, d, ntotal_, batch_queries);
        } else {
            kernels.innerProductDistanceBatch(queries_buffer, vectors_, all_distances_buffer, d, ntotal_, batch_queries);
        }
        
        // Select top-k for each query in the batch
        float* all_distances = (float*)[all_distances_buffer contents];
        
        for (idx_t i = 0; i < batch_queries; ++i) {
            idx_t q = batch_start + i;
            float* query_distances = all_distances + i * ntotal_;
            
            // Create distance-label pairs
            std::vector<std::pair<float, idx_t>> dist_idx_pairs;
            dist_idx_pairs.reserve(ntotal_);
            
            for (idx_t j = 0; j < ntotal_; ++j) {
                dist_idx_pairs.emplace_back(query_distances[j], j);
            }
            
            // Partial sort to get top k
            std::partial_sort(dist_idx_pairs.begin(), 
                            dist_idx_pairs.begin() + k, 
                            dist_idx_pairs.end());
            
            // Copy results
            for (idx_t j = 0; j < k; ++j) {
                distances[q * k + j] = dist_idx_pairs[j].first;
                labels[q * k + j] = dist_idx_pairs[j].second;
            }
        }
    }
    
    return;
    
    // Old per-query implementation below (kept for reference)
    for (idx_t q = 0; q < n; ++q) {
        // Create buffers
        id<MTLBuffer> query_buffer = [device newBufferWithBytes:x + q * d
                                                      length:d * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> dist_labels_buffer = [device newBufferWithLength:ntotal_ * sizeof(DistanceLabel)
                                                                options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> out_distances_buffer = [device newBufferWithLength:k * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> out_labels_buffer = [device newBufferWithLength:k * sizeof(idx_t)
                                                              options:MTLResourceStorageModeShared];
        
        // Compute distances
        if (metric_type == METRIC_L2) {
            kernels.l2Distance(query_buffer, vectors_, dist_labels_buffer, d, ntotal_);
        } else if (metric_type == METRIC_INNER_PRODUCT) {
            kernels.innerProductDistance(query_buffer, vectors_, dist_labels_buffer, d, ntotal_);
        }
        
        // Select top-k
        kernels.selectTopK(dist_labels_buffer, out_distances_buffer, out_labels_buffer, 
                          ntotal_, k, 0);
        
        // Copy results back
        memcpy(distances + q * k, [out_distances_buffer contents], k * sizeof(float));
        memcpy(labels + q * k, [out_labels_buffer contents], k * sizeof(idx_t));
    }
}

} // namespace metal
} // namespace faiss