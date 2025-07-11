#include "MetalIndexFlat.h"
#include "MetalResources.h"
#include "MetalKernels.h"

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
        idx_t* labels) const {
    if (n == 0) {
        return;
    }

    MetalKernels kernels(resources_);

    id<MTLDevice> device = resources_->getDevice(0);
    id<MTLBuffer> query_buffer = [device newBufferWithBytes:x
                                                  length:n * d * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    id<MTLBuffer> distances_buffer = [device newBufferWithLength:n * ntotal_ * sizeof(float)
                                                         options:MTLResourceStorageModeShared];

    for (int i = 0; i < n; ++i) {
        kernels.l2Distance(
                query_buffer,
                vectors_,
                distances_buffer,
                d,
                ntotal_);
    }

    // For now, we just copy the distances back to the host and do the rest of the
    // work on the CPU. This is inefficient, but it's a start.
    float* all_distances = (float*)[distances_buffer contents];

    // Find the k-nearest neighbors on the CPU
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            float min_dist = -1.0f;
            idx_t min_idx = -1;
            for (int l = 0; l < ntotal_; ++l) {
                float dist = all_distances[i * ntotal_ + l];
                if (min_dist < 0 || dist < min_dist) {
                    min_dist = dist;
                    min_idx = l;
                }
            }
            distances[i * k + j] = min_dist;
            labels[i * k + j] = min_idx;
        }
    }
}

} // namespace metal
} // namespace faiss