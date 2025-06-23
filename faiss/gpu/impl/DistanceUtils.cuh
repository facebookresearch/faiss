/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include <faiss/gpu/utils/Tensor.cuh>

//
// Shared utilities for brute-force distance calculations
//

namespace faiss {
namespace gpu {

struct IPDistance {
    __host__ __device__ IPDistance() : dist(0) {}

    static constexpr bool kDirection = true; // maximize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = -std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        dist += a * b;
    }

    __host__ __device__ float reduce() {
        return dist;
    }

    __host__ __device__ void combine(const IPDistance& v) {
        dist += v.dist;
    }

    __host__ __device__ IPDistance zero() const {
        return IPDistance();
    }

    float dist;
};

struct L1Distance {
    __host__ __device__ L1Distance() : dist(0) {}

    static constexpr bool kDirection = false; // minimize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        dist += fabsf(a - b);
    }

    __host__ __device__ float reduce() {
        return dist;
    }

    __host__ __device__ void combine(const L1Distance& v) {
        dist += v.dist;
    }

    __host__ __device__ L1Distance zero() const {
        return L1Distance();
    }

    float dist;
};

struct L2Distance {
    __host__ __device__ L2Distance() : dist(0) {}

    static constexpr bool kDirection = false; // minimize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        float v = a - b;
        dist += v * v;
    }

    __host__ __device__ float reduce() {
        return dist;
    }

    __host__ __device__ void combine(const L2Distance& v) {
        dist += v.dist;
    }

    __host__ __device__ L2Distance zero() const {
        return L2Distance();
    }

    float dist;
};

struct LpDistance {
    __host__ __device__ LpDistance() : p(2), dist(0) {}

    __host__ __device__ LpDistance(float arg) : p(arg), dist(0) {}

    __host__ __device__ LpDistance(const LpDistance& v)
            : p(v.p), dist(v.dist) {}

    __host__ __device__ LpDistance& operator=(const LpDistance& v) {
        p = v.p;
        dist = v.dist;
        return *this;
    }

    static constexpr bool kDirection = false; // minimize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        dist += powf(fabsf(a - b), p);
    }

    __host__ __device__ float reduce() {
        return dist;
    }

    __host__ __device__ void combine(const LpDistance& v) {
        dist += v.dist;
    }

    __host__ __device__ LpDistance zero() const {
        return LpDistance(p);
    }

    float p;
    float dist;
};

struct LinfDistance {
    __host__ __device__ LinfDistance() : dist(0) {}

    static constexpr bool kDirection = false; // minimize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        dist = fmaxf(dist, fabsf(a - b));
    }

    __host__ __device__ float reduce() {
        return dist;
    }

    __host__ __device__ void combine(const LinfDistance& v) {
        dist = fmaxf(dist, v.dist);
    }

    __host__ __device__ LinfDistance zero() const {
        return LinfDistance();
    }

    float dist;
};

struct CanberraDistance {
    __host__ __device__ CanberraDistance() : dist(0) {}

    static constexpr bool kDirection = false; // minimize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        float denom = fabsf(a) + fabsf(b);
        dist += fabsf(a - b) / denom;
    }

    __host__ __device__ float reduce() {
        return dist;
    }

    __host__ __device__ void combine(const CanberraDistance& v) {
        dist += v.dist;
    }

    __host__ __device__ CanberraDistance zero() const {
        return CanberraDistance();
    }

    float dist;
};

struct BrayCurtisDistance {
    __host__ __device__ BrayCurtisDistance() : numerator(0), denominator(0) {}

    static constexpr bool kDirection = false; // minimize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        numerator += fabsf(a - b);
        denominator += fabsf(a + b);
    }

    __host__ __device__ float reduce() {
        return (numerator / denominator);
    }

    __host__ __device__ void combine(const BrayCurtisDistance& v) {
        numerator += v.numerator;
        denominator += v.denominator;
    }

    __host__ __device__ BrayCurtisDistance zero() const {
        return BrayCurtisDistance();
    }

    float numerator;
    float denominator;
};

struct JensenShannonDistance {
    __host__ __device__ JensenShannonDistance() : dist(0) {}

    static constexpr bool kDirection = false; // minimize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        float m = 0.5f * (a + b);

        float x = m / a;
        float y = m / b;

        float kl1 = -a * log(x);
        float kl2 = -b * log(y);

        dist += kl1 + kl2;
    }

    __host__ __device__ float reduce() {
        return 0.5 * dist;
    }

    __host__ __device__ void combine(const JensenShannonDistance& v) {
        dist += v.dist;
    }

    __host__ __device__ JensenShannonDistance zero() const {
        return JensenShannonDistance();
    }

    float dist;
};

struct JaccardSimilarity {
    __host__ __device__ JaccardSimilarity() : numerator(0), denominator(0) {}

    static constexpr bool kDirection = true; // maximize
    static constexpr float kIdentityData = 0;
    static constexpr float kMaxDistance = -std::numeric_limits<float>::max();

    __host__ __device__ void handle(float a, float b) {
        numerator += fmin(a, b);
        denominator += fmax(a, b);
    }

    __host__ __device__ float reduce() {
        return numerator / denominator;
    }

    __host__ __device__ void combine(const JaccardSimilarity& v) {
        numerator += v.numerator;
        denominator += v.denominator;
    }

    __host__ __device__ JaccardSimilarity zero() const {
        return JaccardSimilarity();
    }

    float numerator;
    float denominator;
};

template <typename T, bool InnerContig>
Tensor<T, 2, InnerContig> sliceCentroids(
        Tensor<T, 2, InnerContig>& centroids,
        bool centroidsRowMajor,
        idx_t startCentroid,
        idx_t num) {
    // Row major is (num, dim)
    // Col major is (dim, num)
    if (startCentroid == 0 &&
        num == centroids.getSize(centroidsRowMajor ? 0 : 1)) {
        return centroids;
    }

    return centroids.narrow(centroidsRowMajor ? 0 : 1, startCentroid, num);
}

// For each chunk of k indices, increment the index by chunk * increment
template <typename T>
__global__ void incrementIndex(
        Tensor<T, 2, true> indices,
        int k,
        idx_t increment) {
    for (idx_t i = blockIdx.y; i < indices.getSize(0); i += gridDim.y) {
        for (auto j = threadIdx.x; j < k; j += blockDim.x) {
            indices[i][idx_t(blockIdx.x) * k + j] += blockIdx.x * increment;
        }
    }
}

// Used to update result indices in distance computation where the number of
// centroids is high, and is tiled
template <typename T>
void runIncrementIndex(
        Tensor<T, 2, true>& indices,
        int k,
        idx_t increment,
        cudaStream_t stream) {
    // Input should be an even divisor of k
    FAISS_ASSERT(indices.getSize(1) % k == 0);

    dim3 grid(indices.getSize(1) / k, indices.getSize(0));
    auto block = std::min(k, getMaxThreadsCurrentDevice());

    incrementIndex<<<grid, block, 0, stream>>>(indices, k, increment);
    CUDA_TEST_ERROR();
}

// If the inner size (dim) of the vectors is small, we want a larger query tile
// size, like 1024
inline void chooseTileSize(
        idx_t numQueries,
        idx_t numCentroids,
        int dim,
        idx_t elementSize,
        size_t tempMemAvailable,
        idx_t& tileRows,
        idx_t& tileCols) {
    // The matrix multiplication should be large enough to be efficient, but if
    // it is too large, we seem to lose efficiency as opposed to
    // double-streaming. Each tile size here defines 1/2 of the memory use due
    // to double streaming. We ignore available temporary memory, as that is
    // adjusted independently by the user and can thus meet these requirements
    // (or not). For <= 4 GB GPUs, prefer 512 MB of usage. For <= 8 GB GPUs,
    // prefer 768 MB of usage. Otherwise, prefer 1 GB of usage.
    auto totalMem = getCurrentDeviceProperties().totalGlobalMem;

    idx_t targetUsage = 0;

    if (totalMem <= ((size_t)4) * 1024 * 1024 * 1024) {
        targetUsage = 512 * 1024 * 1024;
    } else if (totalMem <= ((size_t)8) * 1024 * 1024 * 1024) {
        targetUsage = 768 * 1024 * 1024;
    } else {
        targetUsage = 1024 * 1024 * 1024;
    }

    targetUsage /= 2 * elementSize;

    // 512 seems to be a batch size sweetspot for float32.
    // If we are on float16, increase to 512.
    // If the k size (vec dim) of the matrix multiplication is small (<= 32),
    // increase to 1024.
    idx_t preferredTileRows = 512;
    if (dim <= 32) {
        preferredTileRows = 1024;
    }

    tileRows = std::min(preferredTileRows, numQueries);

    // tileCols is the remainder size
    tileCols = std::min(targetUsage / preferredTileRows, numCentroids);
}

} // namespace gpu
} // namespace faiss
