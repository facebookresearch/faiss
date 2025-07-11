#include "MetalIndexIVFFlat.h"

namespace faiss {
namespace metal {

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        faiss::Index* quantizer,
        size_t d,
        size_t nlist,
        faiss::MetricType metric)
    : faiss::IndexIVFFlat(quantizer, d, nlist, metric),
      resources_(resources) {
    offsets_ = [resources_->getDevice(0) newBufferWithLength:(nlist + 1) * sizeof(idx_t) options:MTLResourceStorageModeShared];
    memset([offsets_ contents], 0, (nlist + 1) * sizeof(idx_t));
}

void MetalIndexIVFFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    idx_t* coarse_assign = new idx_t[n];
    quantizer->search(n, x, 1, nullptr, coarse_assign);

    // This is a CPU-based implementation for now. A more efficient implementation
    // would do this on the GPU.

    // First, we need to figure out how many vectors are in each list
    std::vector<idx_t> list_counts(nlist, 0);
    for (idx_t i = 0; i < n; ++i) {
        list_counts[coarse_assign[i]]++;
    }

    // Then, we need to update the offsets
    idx_t* offsets = (idx_t*)[offsets_ contents];
    for (int i = 0; i < nlist; ++i) {
        offsets[i + 1] = offsets[i] + list_counts[i];
    }

    // Now we can allocate the new buffers
    id<MTLDevice> device = resources_->getDevice(0);
    id<MTLBuffer> new_vectors = [device newBufferWithLength:offsets[nlist] * d * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> new_ids = [device newBufferWithLength:offsets[nlist] * sizeof(idx_t) options:MTLResourceStorageModeShared];

    // And copy the data over
    float* new_vectors_ptr = (float*)[new_vectors contents];
    idx_t* new_ids_ptr = (idx_t*)[new_ids contents];

    std::vector<idx_t> current_offsets = std::vector<idx_t>(offsets, offsets + nlist + 1);

    for (idx_t i = 0; i < n; ++i) {
        idx_t list_no = coarse_assign[i];
        idx_t offset = current_offsets[list_no];
        memcpy(new_vectors_ptr + offset * d, x + i * d, d * sizeof(float));
        new_ids_ptr[offset] = xids ? xids[i] : ntotal + i;
        current_offsets[list_no]++;
    }

    // Finally, we can replace the old buffers with the new ones
    vectors_ = new_vectors;
    ids_ = new_ids;

    ntotal += n;
    delete[] coarse_assign;
}

void MetalIndexIVFFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    idx_t* coarse_assign = new idx_t[n * nprobe];
    float* coarse_dist = new float[n * nprobe];
    quantizer->search(n, x, nprobe, coarse_dist, coarse_assign);

    MetalKernels kernels(resources_);

    id<MTLDevice> device = resources_->getDevice(0);
    id<MTLBuffer> query_buffer = [device newBufferWithBytes:x
                                                  length:n * d * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    id<MTLBuffer> coarse_assign_buffer = [device newBufferWithBytes:coarse_assign
                                                            length:n * nprobe * sizeof(idx_t)
                                                           options:MTLResourceStorageModeShared];

    id<MTLBuffer> out_distances_buffer = [device newBufferWithLength:n * k * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_labels_buffer = [device newBufferWithLength:n * k * sizeof(idx_t)
                                                          options:MTLResourceStorageModeShared];

    kernels.ivfflat_scan_per_query(
            query_buffer,
            vectors_,
            ids_,
            offsets_,
            coarse_assign_buffer,
            nprobe,
            d,
            k,
            out_distances_buffer,
            out_labels_buffer,
            n);

    memcpy(distances, [out_distances_buffer contents], n * k * sizeof(float));
    memcpy(labels, [out_labels_buffer contents], n * k * sizeof(idx_t));

    delete[] coarse_assign;
    delete[] coarse_dist;
}

} // namespace metal
} // namespace faiss
