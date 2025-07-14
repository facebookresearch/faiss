#include "MetalIndexIVFFlat.h"

namespace faiss {
namespace metal {

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        faiss::Index* quantizer,
        size_t d,
        size_t nlist,
        faiss::MetricType metric)
    : faiss::IndexIVFFlat(quantizer, d, nlist, metric, false),  // false = don't own invlists
      resources_(resources) {
    // Initialize Metal buffers
    id<MTLDevice> device = resources_->getDevice(0);
    offsets_ = [device newBufferWithLength:(nlist + 1) * sizeof(idx_t) options:MTLResourceStorageModeShared];
    memset([offsets_ contents], 0, (nlist + 1) * sizeof(idx_t));
    
    // Initialize empty buffers
    vectors_ = nil;
    ids_ = nil;
}

void MetalIndexIVFFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    // First train the quantizer if needed
    if (!is_trained) {
        train(n, x);
    }
    
    idx_t* coarse_assign = new idx_t[n];
    quantizer->search(n, x, 1, nullptr, coarse_assign);

    // This is a CPU-based implementation for now. A more efficient implementation
    // would do this on the GPU.

    // First, we need to figure out how many vectors are in each list
    std::vector<idx_t> list_counts(nlist, 0);
    for (idx_t i = 0; i < n; ++i) {
        FAISS_THROW_IF_NOT_MSG(coarse_assign[i] >= 0 && coarse_assign[i] < nlist,
                               "Invalid coarse assignment");
        list_counts[coarse_assign[i]]++;
    }

    // Get current offsets
    idx_t* offsets = (idx_t*)[offsets_ contents];
    std::vector<idx_t> old_offsets(nlist + 1);
    memcpy(old_offsets.data(), offsets, (nlist + 1) * sizeof(idx_t));
    
    // Calculate new offsets
    std::vector<idx_t> new_offsets(nlist + 1);
    new_offsets[0] = 0;
    for (int i = 0; i < nlist; ++i) {
        idx_t old_count = old_offsets[i + 1] - old_offsets[i];
        new_offsets[i + 1] = new_offsets[i] + old_count + list_counts[i];
    }

    // Now we can allocate the new buffers
    id<MTLDevice> device = resources_->getDevice(0);
    size_t total_vectors = new_offsets[nlist];
    
    id<MTLBuffer> new_vectors = nil;
    id<MTLBuffer> new_ids = nil;
    
    if (total_vectors > 0) {
        new_vectors = [device newBufferWithLength:total_vectors * d * sizeof(float) 
                                         options:MTLResourceStorageModeShared];
        new_ids = [device newBufferWithLength:total_vectors * sizeof(idx_t) 
                                      options:MTLResourceStorageModeShared];
        
        FAISS_THROW_IF_NOT_MSG(new_vectors, "Failed to allocate vectors buffer");
        FAISS_THROW_IF_NOT_MSG(new_ids, "Failed to allocate ids buffer");
    }

    // Copy existing data to new buffers
    float* new_vectors_ptr = (float*)[new_vectors contents];
    idx_t* new_ids_ptr = (idx_t*)[new_ids contents];
    
    if (vectors_) {
        float* old_vectors_ptr = (float*)[vectors_ contents];
        idx_t* old_ids_ptr = (idx_t*)[ids_ contents];
        
        for (int i = 0; i < nlist; ++i) {
            idx_t old_start = old_offsets[i];
            idx_t old_count = old_offsets[i + 1] - old_offsets[i];
            idx_t new_start = new_offsets[i];
            
            if (old_count > 0) {
                memcpy(new_vectors_ptr + new_start * d, 
                       old_vectors_ptr + old_start * d, 
                       old_count * d * sizeof(float));
                memcpy(new_ids_ptr + new_start, 
                       old_ids_ptr + old_start, 
                       old_count * sizeof(idx_t));
            }
        }
    }

    // Add new vectors
    std::vector<idx_t> current_offsets = new_offsets;
    for (int i = 0; i < nlist; ++i) {
        idx_t old_count = old_offsets[i + 1] - old_offsets[i];
        current_offsets[i] += old_count;
    }

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
    
    // Update the offsets buffer
    memcpy(offsets, new_offsets.data(), (nlist + 1) * sizeof(idx_t));

    ntotal += n;
    delete[] coarse_assign;
}

void MetalIndexIVFFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // Early exit if empty
    if (ntotal == 0 || n == 0) {
        return;
    }
    
    idx_t* coarse_assign = new idx_t[n * nprobe];
    float* coarse_dist = new float[n * nprobe];
    quantizer->search(n, x, nprobe, coarse_dist, coarse_assign);

    // Convert idx_t to int for Metal kernel
    int* coarse_assign_int = new int[n * nprobe];
    for (idx_t i = 0; i < n * nprobe; ++i) {
        coarse_assign_int[i] = static_cast<int>(coarse_assign[i]);
    }
    
    // Convert offsets to int
    idx_t* offsets_ptr = (idx_t*)[offsets_ contents];
    int* offsets_int = new int[nlist + 1];
    for (idx_t i = 0; i <= nlist; ++i) {
        offsets_int[i] = static_cast<int>(offsets_ptr[i]);
    }
    
    // Convert ids to int
    int total_ids = offsets_ptr[nlist];
    int* ids_int = nullptr;
    if (total_ids > 0 && ids_) {
        idx_t* ids_ptr = (idx_t*)[ids_ contents];
        ids_int = new int[total_ids];
        for (int i = 0; i < total_ids; ++i) {
            ids_int[i] = static_cast<int>(ids_ptr[i]);
        }
    } else {
        // Create empty buffer
        ids_int = new int[1];
        ids_int[0] = 0;
        total_ids = 1;
    }

    MetalKernels kernels(resources_);

    id<MTLDevice> device = resources_->getDevice(0);
    id<MTLBuffer> query_buffer = [device newBufferWithBytes:x
                                                  length:n * d * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    id<MTLBuffer> coarse_assign_buffer = [device newBufferWithBytes:coarse_assign_int
                                                            length:n * nprobe * sizeof(int)
                                                           options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> offsets_buffer = [device newBufferWithBytes:offsets_int
                                                       length:(nlist + 1) * sizeof(int)
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> ids_buffer = [device newBufferWithBytes:ids_int
                                                   length:total_ids * sizeof(int)
                                                  options:MTLResourceStorageModeShared];

    id<MTLBuffer> out_distances_buffer = [device newBufferWithLength:n * k * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_labels_buffer = [device newBufferWithLength:n * k * sizeof(int)
                                                          options:MTLResourceStorageModeShared];

    // Check if we have vectors to search
    if (!vectors_ || total_ids == 0) {
        // Fill with invalid results
        for (idx_t i = 0; i < n * k; ++i) {
            distances[i] = HUGE_VALF;
            labels[i] = -1;
        }
        delete[] coarse_assign;
        delete[] coarse_dist;
        delete[] coarse_assign_int;
        delete[] offsets_int;
        delete[] ids_int;
        return;
    }
    
    kernels.ivfflat_scan_per_query(
            query_buffer,
            vectors_,
            ids_buffer,
            offsets_buffer,
            coarse_assign_buffer,
            nprobe,
            d,
            k,
            out_distances_buffer,
            out_labels_buffer,
            n);

    memcpy(distances, [out_distances_buffer contents], n * k * sizeof(float));
    
    // Convert int labels back to idx_t
    int* out_labels_int = (int*)[out_labels_buffer contents];
    for (idx_t i = 0; i < n * k; ++i) {
        labels[i] = static_cast<idx_t>(out_labels_int[i]);
    }

    delete[] coarse_assign;
    delete[] coarse_dist;
    delete[] coarse_assign_int;
    delete[] offsets_int;
    delete[] ids_int;
}

} // namespace metal
} // namespace faiss
