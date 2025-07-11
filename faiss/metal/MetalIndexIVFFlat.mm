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
    inverted_lists_.resize(nlist);
    inverted_lists_vectors_.resize(nlist);
}

void MetalIndexIVFFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    idx_t* coarse_assign = new idx_t[n];
    quantizer->search(n, x, 1, nullptr, coarse_assign);

    for (idx_t i = 0; i < n; ++i) {
        idx_t list_no = coarse_assign[i];
        inverted_lists_[list_no].push_back(xids ? xids[i] : ntotal + i);
        for (int j = 0; j < d; ++j) {
            inverted_lists_vectors_[list_no].push_back(x[i * d + j]);
        }
    }

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

    for (idx_t i = 0; i < n; ++i) {
        std::vector<float> all_distances;
        std::vector<idx_t> all_labels;

        for (int j = 0; j < nprobe; ++j) {
            idx_t list_no = coarse_assign[i * nprobe + j];
            const std::vector<float>& vectors = inverted_lists_vectors_[list_no];
            const std::vector<idx_t>& ids = inverted_lists_[list_no];

            for (size_t l = 0; l < ids.size(); ++l) {
                float dist = 0.0f;
                for (int m = 0; m < d; ++m) {
                    float diff = x[i * d + m] - vectors[l * d + m];
                    dist += diff * diff;
                }
                all_distances.push_back(dist);
                all_labels.push_back(ids[l]);
            }
        }

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
