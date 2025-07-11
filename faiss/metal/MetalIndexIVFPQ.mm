#include "MetalIndexIVFPQ.h"

namespace faiss {
namespace metal {

MetalIndexIVFPQ::MetalIndexIVFPQ(
        std::shared_ptr<MetalResources> resources,
        faiss::Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        faiss::MetricType metric)
    : faiss::IndexIVFPQ(quantizer, d, nlist, M, nbits_per_idx, metric),
      resources_(resources) {
    offsets_ = [resources_->getDevice(0) newBufferWithLength:(nlist + 1) * sizeof(idx_t) options:MTLResourceStorageModeShared];
    memset([offsets_ contents], 0, (nlist + 1) * sizeof(idx_t));
}

void MetalIndexIVFPQ::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
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
    id<MTLBuffer> new_codes = [device newBufferWithLength:offsets[nlist] * code_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> new_ids = [device newBufferWithLength:offsets[nlist] * sizeof(idx_t) options:MTLResourceStorageModeShared];

    // And copy the data over
    uint8_t* new_codes_ptr = (uint8_t*)[new_codes contents];
    idx_t* new_ids_ptr = (idx_t*)[new_ids contents];

    std::vector<idx_t> current_offsets = std::vector<idx_t>(offsets, offsets + nlist + 1);

    for (idx_t i = 0; i < n; ++i) {
        idx_t list_no = coarse_assign[i];
        idx_t offset = current_offsets[list_no];
        pq.compute_code(x + i * d, new_codes_ptr + offset * code_size);
        new_ids_ptr[offset] = xids ? xids[i] : ntotal + i;
        current_offsets[list_no]++;
    }

    // Finally, we can replace the old buffers with the new ones
    codes_ = new_codes;
    ids_ = new_ids;

    ntotal += n;
    delete[] coarse_assign;
}

void MetalIndexIVFPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    idx_t* coarse_assign = new idx_t[n * nprobe];
    float* coarse_dist = new float[n * nprobe];
    quantizer->search(n, x, nprobe, coarse_dist, coarse_assign);

    idx_t* offsets = (idx_t*)[offsets_ contents];
    uint8_t* codes = (uint8_t*)[codes_ contents];
    idx_t* ids = (idx_t*)[ids_ contents];

    for (idx_t i = 0; i < n; ++i) {
        std::vector<float> all_distances;
        std::vector<idx_t> all_labels;

        float* dist_table = new float[pq.ksub * pq.M];
        pq.compute_distance_tables(1, x + i * d, dist_table);

        for (int j = 0; j < nprobe; ++j) {
            idx_t list_no = coarse_assign[i * nprobe + j];
            idx_t start = offsets[list_no];
            idx_t end = offsets[list_no + 1];

            for (idx_t l = start; l < end; ++l) {
                float dist = pq.compute_distance_from_tables(
                        codes + l * code_size, dist_table);
                all_distances.push_back(dist);
                all_labels.push_back(ids[l]);
            }
        }

        delete[] dist_table;

        // This is a naive top-k selection. A more efficient implementation
        // would use a heap.
        for (int j = 0; j < k; ++j) {
            float min_dist = -1.0f;
            idx_t min_idx = -1;
            int min_pos = -1;

            for (size_t l = 0; l < all_distances.size(); ++l) {
                if (min_dist < 0 || all_distances[l] < min_dist) {
                    min_dist = all_distances[l];
                    min_idx = all_labels[l];
                    min_pos = l;
                }
            }
            distances[i * k + j] = min_dist;
            labels[i * k + j] = min_idx;
            if (min_pos != -1) {
                all_distances[min_pos] = -1.0f;
            }
        }
    }

    delete[] coarse_assign;
    delete[] coarse_dist;
}

} // namespace metal
} // namespace faiss
