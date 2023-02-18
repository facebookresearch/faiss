/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>

#if defined USE_NVIDIA_RAFT
#include <faiss/gpu/impl/RaftUtils.h>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/neighbors/brute_force.cuh>
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#define RAFT_NAME "raft"
#endif

namespace faiss {
namespace gpu {

#if defined USE_NVIDIA_RAFT
using namespace raft::distance;
using namespace raft::neighbors;
#endif

template <typename T>
void bfKnnConvert(GpuResourcesProvider* prov, const GpuDistanceParams& args) {
    // Validate the input data
    FAISS_THROW_IF_NOT_MSG(
            args.k > 0 || args.k == -1,
            "bfKnn: k must be > 0 for top-k reduction, "
            "or -1 for all pairwise distances");
    FAISS_THROW_IF_NOT_MSG(args.dims > 0, "bfKnn: dims must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.numVectors > 0, "bfKnn: numVectors must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.vectors, "bfKnn: vectors must be provided (passed null)");
    FAISS_THROW_IF_NOT_MSG(
            args.numQueries > 0, "bfKnn: numQueries must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.queries, "bfKnn: queries must be provided (passed null)");
    FAISS_THROW_IF_NOT_MSG(
            args.outDistances,
            "bfKnn: outDistances must be provided (passed null)");
    FAISS_THROW_IF_NOT_MSG(
            args.outIndices || args.k == -1,
            "bfKnn: outIndices must be provided (passed null)");

    // If the user specified a device, then ensure that it is currently set
    int device = -1;
    if (args.device == -1) {
        // Original behavior if no device is specified, use the current CUDA
        // thread local device
        device = getCurrentDevice();
    } else {
        // Otherwise, use the device specified in `args`
        device = args.device;

        FAISS_THROW_IF_NOT_FMT(
                device >= 0 && device < getNumDevices(),
                "bfKnn: device specified must be -1 (current CUDA thread local device) "
                "or within the range [0, %d)",
                getNumDevices());
    }

    DeviceScope scope(device);

    // Don't let the resources go out of scope
    auto resImpl = prov->getResources();
    auto res = resImpl.get();
    auto stream = res->getDefaultStreamCurrentDevice();

    auto tVectors = toDeviceTemporary<T, 2>(
            res,
            device,
            const_cast<T*>(reinterpret_cast<const T*>(args.vectors)),
            stream,
            {args.vectorsRowMajor ? args.numVectors : args.dims,
             args.vectorsRowMajor ? args.dims : args.numVectors});
    auto tQueries = toDeviceTemporary<T, 2>(
            res,
            device,
            const_cast<T*>(reinterpret_cast<const T*>(args.queries)),
            stream,
            {args.queriesRowMajor ? args.numQueries : args.dims,
             args.queriesRowMajor ? args.dims : args.numQueries});

    DeviceTensor<float, 1, true> tVectorNorms;
    if (args.vectorNorms) {
        tVectorNorms = toDeviceTemporary<float, 1>(
                res,
                device,
                const_cast<float*>(args.vectorNorms),
                stream,
                {args.numVectors});
    }

    auto tOutDistances = toDeviceTemporary<float, 2>(
            res,
            device,
            args.outDistances,
            stream,
            {args.numQueries, args.k == -1 ? args.numVectors : args.k});

    if (args.k == -1) {
        // Reporting all pairwise distances
        allPairwiseDistanceOnDevice<T>(
                res,
                device,
                stream,
                tVectors,
                args.vectorsRowMajor,
                args.vectorNorms ? &tVectorNorms : nullptr,
                tQueries,
                args.queriesRowMajor,
                args.metric,
                args.metricArg,
                tOutDistances);
    } else if (args.outIndicesType == IndicesDataType::I64) {
        auto tOutIndices = toDeviceTemporary<idx_t, 2>(
                res,
                device,
                (idx_t*)args.outIndices,
                stream,
                {args.numQueries, args.k});

        // Since we've guaranteed that all arguments are on device, call the
        // implementation
        bfKnnOnDevice<T>(
                res,
                device,
                stream,
                tVectors,
                args.vectorsRowMajor,
                args.vectorNorms ? &tVectorNorms : nullptr,
                tQueries,
                args.queriesRowMajor,
                args.k,
                args.metric,
                args.metricArg,
                tOutDistances,
                tOutIndices,
                args.ignoreOutDistances);

        fromDevice<idx_t, 2>(tOutIndices, (idx_t*)args.outIndices, stream);

    } else if (args.outIndicesType == IndicesDataType::I32) {
        // The brute-force API supports i64 indices, but our output buffer is
        // i32 so we need to temporarily allocate and then convert back to i32
        // FIXME: convert to int32_t everywhere?
        static_assert(sizeof(int) == 4, "");
        DeviceTensor<idx_t, 2, true> tIntIndices(
                res,
                makeTempAlloc(AllocType::Other, stream),
                {args.numQueries, args.k});

        // Since we've guaranteed that all arguments are on device, call the
        // implementation
        bfKnnOnDevice<T>(
                res,
                device,
                stream,
                tVectors,
                args.vectorsRowMajor,
                args.vectorNorms ? &tVectorNorms : nullptr,
                tQueries,
                args.queriesRowMajor,
                args.k,
                args.metric,
                args.metricArg,
                tOutDistances,
                tIntIndices,
                args.ignoreOutDistances);
        // Convert and copy int indices out
        auto tOutIntIndices = toDeviceTemporary<int, 2>(
                res,
                device,
                (int*)args.outIndices,
                stream,
                {args.numQueries, args.k});

        convertTensor<idx_t, int, 2>(stream, tIntIndices, tOutIntIndices);

        // Copy back if necessary
        fromDevice<int, 2>(tOutIntIndices, (int*)args.outIndices, stream);
    } else {
        FAISS_THROW_MSG("unknown outIndicesType");
    }

    // Copy distances back if necessary
    fromDevice<float, 2>(tOutDistances, args.outDistances, stream);
}

void bfKnn(GpuResourcesProvider* res, const GpuDistanceParams& args) {
    // For now, both vectors and queries must be of the same data type
    FAISS_THROW_IF_NOT_MSG(
            args.vectorType == args.queryType,
            "limitation: both vectorType and queryType must currently "
            "be the same (F32 or F16");

#if defined USE_NVIDIA_RAFT
    // Note: For now, RAFT bfknn requires queries and vectors to be same layout
    if(args.should_use_raft() && args.queriesRowMajor == args.vectorsRowMajor) {

        DistanceType distance = faiss_to_raft(args.metric, false);

        auto resImpl = res->getResources();
        auto res_impl = resImpl.get();
        raft::handle_t& handle = res_impl->getRaftHandleCurrentDevice();

        idx_t dims = args.dims;
        idx_t num_vectors = args.numVectors;
        const float *vectors = reinterpret_cast<const float*>(args.vectors);
        const float *queries = reinterpret_cast<const float*>(args.queries);
        idx_t num_queries = args.numQueries;
        int k = args.k;
        float metric_arg = args.metricArg;
        idx_t *out_indices = reinterpret_cast<idx_t*>(args.outIndices);
        float *out_distances = reinterpret_cast<float*>(args.outDistances);

        //args.queriesRowMajor ? raft::row_major : raft::col_major;


        auto inds = raft::make_device_matrix_view<idx_t, idx_t>(
                out_indices, num_queries, k);
        auto dists = raft::make_device_matrix_view<float, idx_t>(
                out_distances,
                num_queries,
                k);


        if(args.queriesRowMajor) {
            auto index = raft::make_device_matrix_view<const float, idx_t, raft::row_major>(vectors, num_vectors, dims);
            auto search = raft::make_device_matrix_view<const float, idx_t, raft::row_major>(queries, num_queries, dims);

            std::vector<raft::device_matrix_view<const float, idx_t, raft::row_major>> index_vec = {index};
            // For now, use RAFT's fused KNN when k <= 64 and L2 metric is used
            if (args.k <= 64 && args.metric == MetricType::METRIC_L2 && args.numVectors > 0) {
                RAFT_LOG_INFO("Invoking flat fused_l2_knn");
                brute_force::fused_l2_knn(handle, index, search, inds, dists, distance);
            } else {
                RAFT_LOG_INFO("Invoking flat bfknn");
                brute_force::knn(handle, index_vec, search, inds, dists, k, distance, metric_arg);
            }
        } else {
            auto index = raft::make_device_matrix_view<const float, idx_t, raft::col_major>(vectors, num_vectors, dims);
            auto search = raft::make_device_matrix_view<const float, idx_t, raft::col_major>(queries, num_queries, dims);

            std::vector<raft::device_matrix_view<const float, idx_t, raft::col_major>> index_vec = {index};
            // For now, use RAFT's fused KNN when k <= 64 and L2 metric is used
            if (args.k <= 64 && args.metric == MetricType::METRIC_L2 && args.numVectors > 0) {
                RAFT_LOG_INFO("Invoking flat fused_l2_knn");
                brute_force::fused_l2_knn(handle, index, search, inds, dists, distance);
            } else {
                RAFT_LOG_INFO("Invoking flat bfknn");
                brute_force::knn(handle, index_vec, search, inds, dists, k, distance, metric_arg);
            }
        }
    } else {
#else
        if (args.vectorType == DistanceDataType::F32) {
            bfKnnConvert<float>(res, args);
        } else if (args.vectorType == DistanceDataType::F16) {
            bfKnnConvert<half>(res, args);
        } else {
            FAISS_THROW_MSG("unknown vectorType");
        }
#endif
#if defined USE_NVIDIA_RAFT
    }
#endif
}

// legacy version
void bruteForceKnn(
        GpuResourcesProvider* res,
        faiss::MetricType metric,
        // A region of memory size numVectors x dims, with dims
        // innermost
        const float* vectors,
        bool vectorsRowMajor,
        idx_t numVectors,
        // A region of memory size numQueries x dims, with dims
        // innermost
        const float* queries,
        bool queriesRowMajor,
        idx_t numQueries,
        int dims,
        int k,
        // A region of memory size numQueries x k, with k
        // innermost
        float* outDistances,
        // A region of memory size numQueries x k, with k
        // innermost
        idx_t* outIndices) {
    std::cerr << "bruteForceKnn is deprecated; call bfKnn instead" << std::endl;

    GpuDistanceParams args;
    args.metric = metric;
    args.k = k;
    args.dims = dims;
    args.vectors = vectors;
    args.vectorsRowMajor = vectorsRowMajor;
    args.numVectors = numVectors;
    args.queries = queries;
    args.queriesRowMajor = queriesRowMajor;
    args.numQueries = numQueries;
    args.outDistances = outDistances;
    args.outIndices = outIndices;

    bfKnn(res, args);
}

} // namespace gpu
} // namespace faiss
