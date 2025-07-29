/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryHNSWCagra.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>

#include <algorithm>
#include <random>

namespace faiss {

IndexBinaryHNSWCagra::IndexBinaryHNSWCagra() : IndexBinaryHNSW() {storage = nullptr;}

IndexBinaryHNSWCagra::IndexBinaryHNSWCagra(int d, int M) 
        : IndexBinaryHNSW(d, M) {
    storage = new IndexBinaryFlat(d);
    own_fields = true;
    is_trained = true;
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexBinaryHNSWCagra::add(idx_t n, const uint8_t* x) {
    FAISS_THROW_IF_NOT_MSG(
            !base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexBinaryHNSW::add(n, x);
    
    // After adding, ensure fixed degree if keep_max_size_level0 is set
    if (keep_max_size_level0) {
        ensure_fixed_degree();
    }
}

void IndexBinaryHNSWCagra::add(idx_t n, const void* x, NumericType numeric_type) {
    IndexBinary::add(n, x, numeric_type);
}

void IndexBinaryHNSWCagra::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    
    if (!base_level_only) {
        IndexBinaryHNSW::search(n, x, k, distances, labels, params);
    } else {
        // Search only the base level with random entry points
        std::vector<storage_idx_t> nearest(n);
        std::vector<int32_t> nearest_d(n);
        
        IndexBinaryFlat* flat_storage = dynamic_cast<IndexBinaryFlat*>(storage);
        FAISS_ASSERT(flat_storage != nullptr);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            // Create distance computer for this query
            const uint8_t* q = x + i * code_size;
            HammingComputerDefault hc(q, code_size);
            
            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<int32_t>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            // Select best entry point from random candidates
            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                
                // Compute distance to this candidate
                const uint8_t* yi = flat_storage->xb.data() + idx * code_size;
                int distance = hc.hamming(yi);
                
                if (distance < nearest_d[i]) {
                    nearest[i] = idx;
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

        // Now search from these entry points on level 0 only
        SearchParametersHNSW hnsw_params;
        if (params) {
            hnsw_params = *dynamic_cast<const SearchParametersHNSW*>(params);
        }
        hnsw_params.efSearch = std::max(hnsw_params.efSearch, (int)k);
        
        // Search level 0
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            HammingComputerDefault hc(x + i * code_size, code_size);
            VisitedTable vt(ntotal);
            
            // Priority queue for results
            HNSW::MinimaxHeap candidates(k);
            candidates.push(nearest[i], nearest_d[i]);
            vt.set(nearest[i]);
            
            // Priority queue for exploration
            std::priority_queue<std::pair<int32_t, idx_t>> w;
            w.push(std::make_pair(-nearest_d[i], nearest[i]));
            
            while (!w.empty()) {
                int32_t d0 = -w.top().first;
                idx_t v0 = w.top().second;
                w.pop();
                
                if (d0 > candidates.dis[0]) {
                    break;
                }
                
                // Explore neighbors at level 0
                size_t begin, end;
                hnsw.neighbor_range(v0, 0, &begin, &end);
                
                for (size_t j = begin; j < end; j++) {
                    idx_t v1 = hnsw.neighbors[j];
                    if (v1 < 0) break;
                    if (vt.get(v1)) continue;
                    vt.set(v1);
                    
                    const uint8_t* y1 = flat_storage->xb.data() + v1 * code_size;
                    int32_t d1 = hc.hamming(y1);
                    
                    candidates.push(v1, d1);
                    if (d1 < candidates.dis[0]) {
                        w.push(std::make_pair(-d1, v1));
                    }
                }
            }
            
            // Copy results
            for (size_t j = 0; j < k; j++) {
                labels[i * k + j] = candidates.ids[j];
                distances[i * k + j] = candidates.dis[j];
            }
        }
    }
}

void IndexBinaryHNSWCagra::search(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IndexBinary::search(n, x, numeric_type, k, distances, labels, params);
}

void IndexBinaryHNSWCagra::ensure_fixed_degree() {
    size_t expected_neighbors = hnsw.nb_neighbors(0);
    
    // Check all nodes at level 0
    for (idx_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        
        // Count actual neighbors
        size_t actual_neighbors = 0;
        for (size_t j = begin; j < end; j++) {
            if (hnsw.neighbors[j] >= 0) {
                actual_neighbors++;
            }
        }
        
        // If not enough neighbors, pad with self-loops or nearest valid neighbors
        if (actual_neighbors < expected_neighbors) {
            // For now, we'll pad with self-loops
            // In a production implementation, you might want to find
            // the nearest valid neighbors instead
            size_t j = begin + actual_neighbors;
            while (j < end) {
                hnsw.neighbors[j] = i;  // Self-loop
                j++;
            }
        }
    }
}

bool IndexBinaryHNSWCagra::has_fixed_degree() const {
    if (ntotal == 0) return true;
    
    size_t expected_neighbors = hnsw.nb_neighbors(0);
    
    for (idx_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        
        size_t actual_neighbors = 0;
        for (size_t j = begin; j < end; j++) {
            if (hnsw.neighbors[j] >= 0) {
                actual_neighbors++;
            }
        }
        
        if (actual_neighbors != expected_neighbors) {
            return false;
        }
    }
    
    return true;
}

size_t IndexBinaryHNSWCagra::get_fixed_degree() const {
    return hnsw.nb_neighbors(0);
}

} // namespace faiss 