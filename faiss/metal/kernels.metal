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
