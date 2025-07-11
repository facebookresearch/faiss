#include <metal_stdlib>
using namespace metal;

// A simple kernel to add two vectors
kernel void add_vectors(const device float* inA,
                        const device float* inB,
                        device float* out,
                        uint index [[thread_position_in_grid]]) {
    out[index] = inA[index] + inB[index];
}

kernel void l2_distance(const device float* query,
                      const device float* data,
                      device float* distances,
                      uint d,
                      uint index [[thread_position_in_grid]]) {
    float dist = 0.0f;
    for (uint i = 0; i < d; ++i) {
        float diff = query[i] - data[index * d + i];
        dist += diff * diff;
    }
    distances[index] = dist;
}
