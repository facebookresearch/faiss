#include <metal_stdlib>
using namespace metal;

struct DistanceLabel {
    float distance;
    int label;
};

// A simple kernel to add two vectors
kernel void add_vectors(const device float* inA,
                        const device float* inB,
                        device float* out,
                        uint index [[thread_position_in_grid]]) {
    out[index] = inA[index] + inB[index];
}

kernel void l2_distance(const device float* query,
                      const device float* data,
                      device DistanceLabel* dist_labels,
                      uint d,
                      uint index [[thread_position_in_grid]]) {
    float dist = 0.0f;
    for (uint i = 0; i < d; ++i) {
        float diff = query[i] - data[index * d + i];
        dist += diff * diff;
    }
    dist_labels[index].distance = dist;
    dist_labels[index].label = index;
}

// A bitonic sort kernel
kernel void bitonic_sort(device DistanceLabel* data,
                         uint size,
                         uint stage,
                         uint pass_of_stage,
                         uint direction,
                         uint index [[thread_position_in_grid]]) {
    uint sort_ascending = direction;
    uint thread_id = index;

    uint pair_distance = 1 << (pass_of_stage);
    uint block_width = 2 * pair_distance;

    uint left_id = (thread_id % pair_distance) + (thread_id / pair_distance) * block_width;
    uint right_id = left_id + pair_distance;

    DistanceLabel left_val = data[left_id];
    DistanceLabel right_val = data[right_id];

    uint same_direction = ((left_id / (1 << stage)) % 2) == 0;

    if (same_direction == sort_ascending) {
        if (left_val.distance > right_val.distance) {
            data[left_id] = right_val;
            data[right_id] = left_val;
        }
    } else {
        if (left_val.distance < right_val.distance) {
            data[left_id] = right_val;
            data[right_id] = left_val;
        }
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
    uint nprobe,
    uint d,
    uint k,
    uint M,
    uint ksub,
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
    uint nprobe,
    uint d,
    uint k,
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
