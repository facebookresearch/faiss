/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>

#pragma GCC visibility push(default)
namespace faiss {
namespace gpu {

class GpuResourcesProvider;

// Scalar type of the vector data
enum class DistanceDataType {
    F32 = 1,
    F16,
    BF16,
};

// Scalar type of the indices data
enum class IndicesDataType {
    I64 = 1,
    I32,
};

/// Arguments to brute-force GPU k-nearest neighbor searching
struct GpuDistanceParams {
    //
    // Search parameters
    //

    /// Search parameter: distance metric
    faiss::MetricType metric = METRIC_L2;

    /// Search parameter: distance metric argument (if applicable)
    /// For metric == METRIC_Lp, this is the p-value
    float metricArg = 0;

    /// Search parameter: return k nearest neighbors
    /// If the value provided is -1, then we report all pairwise distances
    /// without top-k filtering
    int k = 0;

    /// Vector dimensionality
    int dims = 0;

    //
    // Vectors being queried
    //

    /// If vectorsRowMajor is true, this is
    /// numVectors x dims, with dims innermost; otherwise,
    /// dims x numVectors, with numVectors innermost
    const void* vectors = nullptr;
    DistanceDataType vectorType = DistanceDataType::F32;
    bool vectorsRowMajor = true;
    idx_t numVectors = 0;

    /// Precomputed L2 norms for each vector in `vectors`, which can be
    /// optionally provided in advance to speed computation for METRIC_L2
    const float* vectorNorms = nullptr;

    //
    // The query vectors (i.e., find k-nearest neighbors in `vectors` for each
    // of the `queries`
    //

    /// If queriesRowMajor is true, this is
    /// numQueries x dims, with dims innermost; otherwise,
    /// dims x numQueries, with numQueries innermost
    const void* queries = nullptr;
    DistanceDataType queryType = DistanceDataType::F32;
    bool queriesRowMajor = true;
    idx_t numQueries = 0;

    //
    // Output results
    //

    /// A region of memory size numQueries x k, with k
    /// innermost (row major) if k > 0, or if k == -1, a region of memory of
    /// size numQueries x numVectors
    float* outDistances = nullptr;

    /// Do we only care about the indices reported, rather than the output
    /// distances? Not used if k == -1 (all pairwise distances)
    bool ignoreOutDistances = false;

    /// A region of memory size numQueries x k, with k
    /// innermost (row major). Not used if k == -1 (all pairwise distances)
    IndicesDataType outIndicesType = IndicesDataType::I64;
    void* outIndices = nullptr;

    //
    // Execution information
    //

    /// On which GPU device should the search run?
    /// -1 indicates that the current CUDA thread-local device
    /// (via cudaGetDevice/cudaSetDevice) is used
    /// Otherwise, an integer 0 <= device < numDevices indicates the device for
    /// execution
    int device = -1;

    /// Should the index dispatch down to cuVS?
#if defined USE_NVIDIA_CUVS
    bool use_cuvs = true;
#else
    bool use_cuvs = false;
#endif
};

/// A function that determines whether cuVS should be used based on various
/// conditions (such as unsupported architecture)
bool should_use_cuvs(GpuDistanceParams args);

/// A wrapper for gpu/impl/Distance.cuh to expose direct brute-force k-nearest
/// neighbor searches on an externally-provided region of memory (e.g., from a
/// pytorch tensor).
/// The data (vectors, queries, outDistances, outIndices) can be resident on the
/// GPU or the CPU, but all calculations are performed on the GPU. If the result
/// buffers are on the CPU, results will be copied back when done.
///
/// All GPU computation is performed on the current CUDA device, and ordered
/// with respect to resources->getDefaultStreamCurrentDevice().
///
/// For each vector in `queries`, searches all of `vectors` to find its k
/// nearest neighbors with respect to the given metric
void bfKnn(GpuResourcesProvider* resources, const GpuDistanceParams& args);

// bfKnn which takes two extra parameters to control the maximum GPU
// memory allowed for vectors and queries, the latter including the
// memory required for the results.
// If 0, the corresponding input must fit into GPU memory.
// If greater than 0, the function will use at most this much GPU
// memory (in bytes) for vectors and queries respectively.
// Vectors are broken up into chunks of size vectorsMemoryLimit,
// and queries are broken up into chunks of size queriesMemoryLimit.
// The tiles resulting from the product of the query and vector
// chunks are processed sequentially on the GPU.
// Only supported for row major matrices and k > 0. The input that
// needs sharding must reside on the CPU.
void bfKnn_tiling(
        GpuResourcesProvider* resources,
        const GpuDistanceParams& args,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit);

/// Deprecated legacy implementation
void bruteForceKnn(
        GpuResourcesProvider* resources,
        faiss::MetricType metric,
        // If vectorsRowMajor is true, this is
        // numVectors x dims, with dims innermost; otherwise,
        // dims x numVectors, with numVectors innermost
        const float* vectors,
        bool vectorsRowMajor,
        idx_t numVectors,
        // If queriesRowMajor is true, this is
        // numQueries x dims, with dims innermost; otherwise,
        // dims x numQueries, with numQueries innermost
        const float* queries,
        bool queriesRowMajor,
        idx_t numQueries,
        int dims,
        int k,
        // A region of memory size numQueries x k, with k
        // innermost (row major)
        float* outDistances,
        // A region of memory size numQueries x k, with k
        // innermost (row major)
        idx_t* outIndices);

} // namespace gpu
} // namespace faiss
#pragma GCC visibility pop
