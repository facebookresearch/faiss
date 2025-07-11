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

    id<MTLBuffer> dist_labels_buffer = [device newBufferWithLength:n * ntotal_ * sizeof(DistanceLabel)
                                                            options:MTLResourceStorageModeShared];

    for (int i = 0; i < n; ++i) {
        kernels.l2Distance(
                query_buffer,
                vectors_,
                dist_labels_buffer,
                d,
                ntotal_);
    }

    // Now we have the distances and labels, we need to find the top-k for each query.
    // We can do this by sorting the distances and taking the first k.
    kernels.bitonicSort(dist_labels_buffer, n * ntotal_);

    // Copy the results back to the host
    DistanceLabel* dist_labels = (DistanceLabel*)[dist_labels_buffer contents];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            distances[i * k + j] = dist_labels[i * ntotal_ + j].distance;
            labels[i * k + j] = dist_labels[i * ntotal_ + j].label;
        }
    }
}

} // namespace metal
} // namespace faiss