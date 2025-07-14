#include <metal_stdlib>
using namespace metal;

struct DistanceLabel {
    float distance;
    int32_t label;
};

// A simple kernel to add two vectors
kernel void add_vectors(const device float* inA,
                        const device float* inB,
                        device float* out,
                        uint index [[thread_position_in_grid]]) {
    out[index] = inA[index] + inB[index];
}

kernel void l2_distance(const device float* query [[buffer(0)]],
                      const device float* data [[buffer(1)]],
                      device DistanceLabel* dist_labels [[buffer(2)]],
                      constant uint& d [[buffer(3)]],
                      constant uint& n [[buffer(4)]],
                      uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    
    float dist = 0.0f;
    const device float* vector = data + index * d;
    
    // Unroll loop for better performance
    uint d4 = d / 4;
    uint remainder = d % 4;
    
    for (uint i = 0; i < d4; ++i) {
        float4 q = ((const device float4*)query)[i];
        float4 v = ((const device float4*)vector)[i];
        float4 diff = q - v;
        dist += dot(diff, diff);
    }
    
    // Handle remaining elements
    for (uint i = d4 * 4; i < d; ++i) {
        float diff = query[i] - vector[i];
        dist += diff * diff;
    }
    
    dist_labels[index].distance = dist;
    dist_labels[index].label = index;
}

kernel void inner_product_distance(const device float* query [[buffer(0)]],
                                 const device float* data [[buffer(1)]],
                                 device DistanceLabel* dist_labels [[buffer(2)]],
                                 constant uint& d [[buffer(3)]],
                                 constant uint& n [[buffer(4)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    
    float dist = 0.0f;
    const device float* vector = data + index * d;
    
    // Unroll loop for better performance
    uint d4 = d / 4;
    
    for (uint i = 0; i < d4; ++i) {
        float4 q = ((const device float4*)query)[i];
        float4 v = ((const device float4*)vector)[i];
        dist += dot(q, v);
    }
    
    // Handle remaining elements
    for (uint i = d4 * 4; i < d; ++i) {
        dist += query[i] * vector[i];
    }
    
    // Negative because we want to maximize inner product
    dist_labels[index].distance = -dist;
    dist_labels[index].label = index;
}

// Parallel k-selection using sorting network
kernel void select_top_k(device DistanceLabel* distances [[buffer(0)]],
                        device float* out_distances [[buffer(1)]],
                        device int64_t* out_labels [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        constant uint& k [[buffer(4)]],
                        constant uint& query_id [[buffer(5)]],
                        uint tid [[thread_position_in_grid]]) {
    
    // Each thread handles one query's results
    if (tid != 0) return;
    
    // Simple selection sort for top-k
    // For production, use a more efficient parallel algorithm
    for (uint i = 0; i < k && i < n; ++i) {
        uint min_idx = i;
        float min_dist = distances[i].distance;
        
        for (uint j = i + 1; j < n; ++j) {
            if (distances[j].distance < min_dist) {
                min_dist = distances[j].distance;
                min_idx = j;
            }
        }
        
        // Swap
        if (min_idx != i) {
            DistanceLabel temp = distances[i];
            distances[i] = distances[min_idx];
            distances[min_idx] = temp;
        }
        
        // Write to output
        out_distances[query_id * k + i] = distances[i].distance;
        out_labels[query_id * k + i] = distances[i].label;
    }
}

void heap_sift_down(thread float* distances, thread int* labels, uint size, uint i) {
    uint left = 2 * i + 1;
    uint right = 2 * i + 2;
    uint largest = i;

    if (left < size && distances[left] > distances[largest]) {
        largest = left;
    }
    if (right < size && distances[right] > distances[largest]) {
        largest = right;
    }

    if (largest != i) {
        float temp_dist = distances[i];
        int temp_label = labels[i];
        distances[i] = distances[largest];
        labels[i] = labels[largest];
        distances[largest] = temp_dist;
        labels[largest] = temp_label;
        heap_sift_down(distances, labels, size, largest);
    }
}

void heap_push(thread float* distances, thread int* labels, thread uint& size, uint k, float dist, int label) {
    if (size < k) {
        distances[size] = dist;
        labels[size] = label;
        size++;
        if (size == k) {
            for (int i = k / 2 - 1; i >= 0; i--) {
                heap_sift_down(distances, labels, k, i);
            }
        }
    } else if (dist < distances[0]) {
        distances[0] = dist;
        labels[0] = label;
        heap_sift_down(distances, labels, k, 0);
    }
}

kernel void ivfpq_scan_per_query(
    const device float* queries,
    const device uint8_t* db_codes,
    const device int* db_ids,
    const device int* list_offsets,
    const device int* coarse_assign,
    const device float* dist_tables,
    constant uint& nprobe,
    constant uint& d,
    constant uint& k,
    constant uint& M,
    constant uint& ksub,
    device float* out_distances,
    device int* out_labels,
    uint query_id [[threadgroup_position_in_grid]]) {

    const device float* query = queries + query_id * d;
    const device float* query_dist_table = dist_tables + query_id * M * ksub;

    // A simple max-heap for top-k selection
    float heap_distances[1024];
    int heap_labels[1024];
    uint heap_size = 0;

    for (uint i = 0; i < nprobe; ++i) {
        uint list_no = coarse_assign[query_id * nprobe + i];
        uint start = list_offsets[list_no];
        uint end = list_offsets[list_no + 1];

        for (uint j = start; j < end; ++j) {
            float dist = 0.0f;
            for (uint m = 0; m < M; ++m) {
                dist += query_dist_table[m * ksub + db_codes[j * M + m]];
            }

            heap_push(heap_distances, heap_labels, heap_size, k, dist, db_ids[j]);
        }
    }

    // For now, just copy the heap to the output
    for (uint i = 0; i < heap_size; ++i) {
        out_distances[query_id * k + i] = heap_distances[i];
        out_labels[query_id * k + i] = heap_labels[i];
    }
}

kernel void ivfflat_scan_per_query(
    const device float* queries,
    const device float* db_vectors,
    const device int* db_ids,
    const device int* list_offsets,
    const device int* coarse_assign,
    constant uint& nprobe,
    constant uint& d,
    constant uint& k,
    device float* out_distances,
    device int* out_labels,
    uint query_id [[threadgroup_position_in_grid]]) {

    const device float* query = queries + query_id * d;

    // A simple max-heap for top-k selection
    float heap_distances[1024];
    int heap_labels[1024];
    uint heap_size = 0;

    for (uint i = 0; i < nprobe; ++i) {
        uint list_no = coarse_assign[query_id * nprobe + i];
        uint start = list_offsets[list_no];
        uint end = list_offsets[list_no + 1];

        for (uint j = start; j < end; ++j) {
            float dist = 0.0f;
            for (uint l = 0; l < d; ++l) {
                float diff = query[l] - db_vectors[j * d + l];
                dist += diff * diff;
            }

            heap_push(heap_distances, heap_labels, heap_size, k, dist, db_ids[j]);
        }
    }

    // For now, just copy the heap to the output
    for (uint i = 0; i < heap_size; ++i) {
        out_distances[query_id * k + i] = heap_distances[i];
        out_labels[query_id * k + i] = heap_labels[i];
    }
}

// HNSW search kernel - simplified beam search
kernel void hnsw_search(
    const device float* queries [[buffer(0)]],
    const device float* db_vectors [[buffer(1)]],
    const device int* levels [[buffer(2)]],
    const device int* graph_offsets [[buffer(3)]],  // Start offset for each node's neighbors
    const device int* graph_neighbors [[buffer(4)]], // Flattened neighbor lists
    constant uint& d [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& ef [[buffer(7)]],  // Search parameter
    constant uint& M [[buffer(8)]],   // Max neighbors per node
    constant uint& nb [[buffer(9)]],  // Total number of vectors
    device float* out_distances [[buffer(10)]],
    device int64_t* out_labels [[buffer(11)]],
    constant int& entry_point [[buffer(12)]],
    uint query_id [[thread_position_in_grid]]) {
    
    if (query_id >= 1) return; // Process one query at a time for now
    
    const device float* query = queries + query_id * d;
    
    // Working memory for beam search
    float candidates[256];  // distances
    int candidate_ids[256]; // node ids
    int candidate_count = 0;
    
    float w_distances[256];  // working set distances
    int w_ids[256];         // working set ids
    int w_count = 0;
    
    // Visited tracking (using a simple array for small graphs)
    bool visited[4096];
    for (int i = 0; i < 4096 && i < nb; i++) {
        visited[i] = false;
    }
    
    // Start from entry point
    float entry_dist = 0.0f;
    const device float* entry_vec = db_vectors + entry_point * d;
    for (uint i = 0; i < d; i++) {
        float diff = query[i] - entry_vec[i];
        entry_dist += diff * diff;
    }
    
    candidates[0] = entry_dist;
    candidate_ids[0] = entry_point;
    candidate_count = 1;
    
    w_distances[0] = entry_dist;
    w_ids[0] = entry_point;
    w_count = 1;
    
    visited[entry_point] = true;
    
    // Main search loop
    while (candidate_count > 0) {
        // Get nearest unvisited candidate
        int best_idx = -1;
        float best_dist = INFINITY;
        for (int i = 0; i < candidate_count; i++) {
            if (candidates[i] < best_dist) {
                best_dist = candidates[i];
                best_idx = i;
            }
        }
        
        if (best_idx < 0) break;
        
        int current = candidate_ids[best_idx];
        float current_dist = candidates[best_idx];
        
        // Remove from candidates
        candidates[best_idx] = candidates[candidate_count - 1];
        candidate_ids[best_idx] = candidate_ids[candidate_count - 1];
        candidate_count--;
        
        // If this point is further than our furthest w point, we can stop
        float furthest_w = 0.0f;
        for (int i = 0; i < w_count; i++) {
            if (w_distances[i] > furthest_w) {
                furthest_w = w_distances[i];
            }
        }
        if (current_dist > furthest_w && w_count >= ef) {
            break;
        }
        
        // Check neighbors
        int neighbors_start = graph_offsets[current];
        int neighbors_end = (current + 1 < nb) ? graph_offsets[current + 1] : neighbors_start + M;
        
        for (int idx = neighbors_start; idx < neighbors_end; idx++) {
            int neighbor = graph_neighbors[idx];
            if (neighbor < 0 || neighbor >= nb) continue;
            if (visited[neighbor]) continue;
            
            visited[neighbor] = true;
            
            // Compute distance
            float dist = 0.0f;
            const device float* neighbor_vec = db_vectors + neighbor * d;
            for (uint i = 0; i < d; i++) {
                float diff = query[i] - neighbor_vec[i];
                dist += diff * diff;
            }
            
            // Add to candidates and w if good enough
            if (dist < furthest_w || w_count < ef) {
                // Add to candidates
                if (candidate_count < 256) {
                    candidates[candidate_count] = dist;
                    candidate_ids[candidate_count] = neighbor;
                    candidate_count++;
                }
                
                // Add to w
                if (w_count < ef) {
                    w_distances[w_count] = dist;
                    w_ids[w_count] = neighbor;
                    w_count++;
                } else {
                    // Replace furthest
                    int furthest_idx = 0;
                    float furthest_dist = w_distances[0];
                    for (int i = 1; i < w_count; i++) {
                        if (w_distances[i] > furthest_dist) {
                            furthest_dist = w_distances[i];
                            furthest_idx = i;
                        }
                    }
                    if (dist < furthest_dist) {
                        w_distances[furthest_idx] = dist;
                        w_ids[furthest_idx] = neighbor;
                    }
                }
            }
        }
    }
    
    // Sort w and output top k
    for (int i = 0; i < k && i < w_count; i++) {
        int min_idx = i;
        for (int j = i + 1; j < w_count; j++) {
            if (w_distances[j] < w_distances[min_idx]) {
                min_idx = j;
            }
        }
        // Swap
        if (min_idx != i) {
            float tmp_dist = w_distances[i];
            int tmp_id = w_ids[i];
            w_distances[i] = w_distances[min_idx];
            w_ids[i] = w_ids[min_idx];
            w_distances[min_idx] = tmp_dist;
            w_ids[min_idx] = tmp_id;
        }
        
        out_distances[query_id * k + i] = w_distances[i];
        out_labels[query_id * k + i] = w_ids[i];
    }
    
    // Fill remaining slots if needed
    for (int i = w_count; i < k; i++) {
        out_distances[query_id * k + i] = INFINITY;
        out_labels[query_id * k + i] = -1;
    }
}

// Batch version that processes multiple queries at once
kernel void l2_distance_batch(const device float* queries [[buffer(0)]],
                             const device float* data [[buffer(1)]],
                             device float* distances [[buffer(2)]],
                             constant uint& d [[buffer(3)]],
                             constant uint& n [[buffer(4)]],
                             constant uint& nq [[buffer(5)]],
                             uint2 index [[thread_position_in_grid]]) {
    uint query_id = index.x;
    uint vector_id = index.y;
    
    if (query_id >= nq || vector_id >= n) return;
    
    const device float* query = queries + query_id * d;
    const device float* vector = data + vector_id * d;
    
    float dist = 0.0f;
    
    // Vectorized computation
    uint d4 = d / 4;
    for (uint i = 0; i < d4; ++i) {
        float4 q = ((const device float4*)query)[i];
        float4 v = ((const device float4*)vector)[i];
        float4 diff = q - v;
        dist += dot(diff, diff);
    }
    
    // Handle remaining elements
    uint remainder = d % 4;
    if (remainder > 0) {
        uint base = d4 * 4;
        for (uint i = 0; i < remainder; ++i) {
            float diff = query[base + i] - vector[base + i];
            dist += diff * diff;
        }
    }
    
    // Store result
    distances[query_id * n + vector_id] = dist;
}

// Inner product batch version
kernel void inner_product_batch(const device float* queries [[buffer(0)]],
                               const device float* data [[buffer(1)]],
                               device float* distances [[buffer(2)]],
                               constant uint& d [[buffer(3)]],
                               constant uint& n [[buffer(4)]],
                               constant uint& nq [[buffer(5)]],
                               uint2 index [[thread_position_in_grid]]) {
    uint query_id = index.x;
    uint vector_id = index.y;
    
    if (query_id >= nq || vector_id >= n) return;
    
    const device float* query = queries + query_id * d;
    const device float* vector = data + vector_id * d;
    
    float dist = 0.0f;
    
    // Vectorized computation
    uint d4 = d / 4;
    for (uint i = 0; i < d4; ++i) {
        float4 q = ((const device float4*)query)[i];
        float4 v = ((const device float4*)vector)[i];
        dist += dot(q, v);
    }
    
    // Handle remaining elements
    uint remainder = d % 4;
    if (remainder > 0) {
        uint base = d4 * 4;
        for (uint i = 0; i < remainder; ++i) {
            dist += query[base + i] * vector[base + i];
        }
    }
    
    // Store negative for inner product (since we want maximum)
    distances[query_id * n + vector_id] = -dist;
}