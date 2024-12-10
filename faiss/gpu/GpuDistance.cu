// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <optional>

#if defined USE_NVIDIA_CUVS
#include <cuvs/neighbors/brute_force.hpp>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/temporary_device_buffer.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/brute_force.cuh>
#endif

namespace faiss {
namespace gpu {

bool should_use_cuvs(GpuDistanceParams args) {
    int dev = args.device >= 0 ? args.device : getCurrentDevice();
    auto prop = getDeviceProperties(dev);

    if (prop.major < 7)
        return false;

    return args.use_cuvs;
}

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

void bfKnn(GpuResourcesProvider* prov, const GpuDistanceParams& args) {
    // For now, both vectors and queries must be of the same data type
    FAISS_THROW_IF_NOT_MSG(
            args.vectorType == args.queryType,
            "limitation: both vectorType and queryType must currently "
            "be the same (F32 / F16 / BF16");

#if defined USE_NVIDIA_CUVS
    // Note: For now, cuVS bfknn requires queries and vectors to be same layout
    if (should_use_cuvs(args) && args.queriesRowMajor == args.vectorsRowMajor &&
        args.outIndicesType == IndicesDataType::I64 &&
        args.vectorType == DistanceDataType::F32 && args.k > 0) {
        cuvsDistanceType distance = metricFaissToCuvs(args.metric, false);

        auto resImpl = prov->getResources();
        auto res = resImpl.get();
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
        raft::device_resources& handle = res->getRaftHandleCurrentDevice();
        auto stream = res->getDefaultStreamCurrentDevice();

        int64_t dims = args.dims;
        int64_t num_vectors = args.numVectors;
        int64_t num_queries = args.numQueries;
        int k = args.k;
        float metric_arg = args.metricArg;

        auto inds =
                raft::make_writeback_temporary_device_buffer<idx_t, int64_t>(
                        handle,
                        reinterpret_cast<idx_t*>(args.outIndices),
                        raft::matrix_extent<int64_t>(num_queries, (int64_t)k));
        auto dists =
                raft::make_writeback_temporary_device_buffer<float, int64_t>(
                        handle,
                        reinterpret_cast<float*>(args.outDistances),
                        raft::matrix_extent<int64_t>(num_queries, (int64_t)k));

        if (args.queriesRowMajor) {
            auto index = raft::make_readonly_temporary_device_buffer<
                    const float,
                    int64_t,
                    raft::row_major>(
                    handle,
                    const_cast<float*>(
                            reinterpret_cast<const float*>(args.vectors)),
                    raft::matrix_extent<int64_t>(num_vectors, dims));

            auto search = raft::make_readonly_temporary_device_buffer<
                    const float,
                    int64_t,
                    raft::row_major>(
                    handle,
                    const_cast<float*>(
                            reinterpret_cast<const float*>(args.queries)),
                    raft::matrix_extent<int64_t>(num_queries, dims));

            // get device_vector_view to the precalculate norms if available
            std::optional<raft::temporary_device_buffer<
                    const float,
                    raft::vector_extent<int64_t>>>
                    norms;
            std::optional<raft::device_vector_view<const float, int64_t>>
                    norms_view;
            if (args.vectorNorms) {
                norms = raft::make_readonly_temporary_device_buffer<
                        const float,
                        int64_t>(
                        handle,
                        args.vectorNorms,
                        raft::vector_extent<int64_t>(num_queries));
                norms_view = norms->view();
            }

            cuvs::neighbors::brute_force::index<float> idx(
                    handle, index.view(), norms_view, distance, metric_arg);
            cuvs::neighbors::brute_force::search(
                    handle,
                    idx,
                    search.view(),
                    inds.view(),
                    dists.view(),
                    std::nullopt);
        } else {
            auto index = raft::make_readonly_temporary_device_buffer<
                    const float,
                    int64_t,
                    raft::col_major>(
                    handle,
                    const_cast<float*>(
                            reinterpret_cast<const float*>(args.vectors)),
                    raft::matrix_extent<int64_t>(num_vectors, dims));

            auto search = raft::make_readonly_temporary_device_buffer<
                    const float,
                    int64_t,
                    raft::col_major>(
                    handle,
                    const_cast<float*>(
                            reinterpret_cast<const float*>(args.queries)),
                    raft::matrix_extent<int64_t>(num_queries, dims));

            std::optional<raft::temporary_device_buffer<
                    const float,
                    raft::vector_extent<int64_t>>>
                    norms;
            std::optional<raft::device_vector_view<const float, int64_t>>
                    norms_view;
            if (args.vectorNorms) {
                norms = raft::make_readonly_temporary_device_buffer<
                        const float,
                        int64_t>(
                        handle,
                        args.vectorNorms,
                        raft::vector_extent<int64_t>(num_queries));
                norms_view = norms->view();
            }

            cuvs::neighbors::brute_force::index<float> idx(
                    handle, index.view(), norms_view, distance, metric_arg);
            cuvs::neighbors::brute_force::search(
                    handle,
                    idx,
                    search.view(),
                    inds.view(),
                    dists.view(),
                    std::nullopt);
        }

        if (args.metric == MetricType::METRIC_Lp) {
            raft::linalg::unary_op(
                    handle,
                    raft::make_const_mdspan(dists.view()),
                    dists.view(),
                    [metric_arg] __device__(const float& a) {
                        return powf(a, metric_arg);
                    });
        } else if (args.metric == MetricType::METRIC_JensenShannon) {
            raft::linalg::unary_op(
                    handle,
                    raft::make_const_mdspan(dists.view()),
                    dists.view(),
                    [] __device__(const float& a) { return powf(a, 2); });
        }

        handle.sync_stream();
    } else
#else
    if (should_use_cuvs(args)) {
        FAISS_THROW_IF_NOT_MSG(
                !should_use_cuvs(args),
                "cuVS has not been compiled into the current version so it cannot be used.");
    } else
#endif
            if (args.vectorType == DistanceDataType::F32) {
        bfKnnConvert<float>(prov, args);
    } else if (args.vectorType == DistanceDataType::F16) {
        bfKnnConvert<half>(prov, args);
    } else if (args.vectorType == DistanceDataType::BF16) {
        if (prov->getResources()->supportsBFloat16CurrentDevice()) {
            bfKnnConvert<__nv_bfloat16>(prov, args);
        } else {
            FAISS_THROW_MSG("not compiled with bfloat16 support");
        }
    } else {
        FAISS_THROW_MSG("unknown vectorType");
    }
}

template <class C>
void bfKnn_shard_database(
        GpuResourcesProvider* prov,
        const GpuDistanceParams& args,
        size_t shard_size,
        size_t distance_size) {
    std::vector<typename C::T> heaps_distances;
    if (args.ignoreOutDistances) {
        heaps_distances.resize(args.numQueries * args.k, 0);
    }
    HeapArray<C> heaps = {
            (size_t)args.numQueries,
            (size_t)args.k,
            (typename C::TI*)args.outIndices,
            args.ignoreOutDistances ? heaps_distances.data()
                                    : args.outDistances};
    heaps.heapify();
    std::vector<typename C::TI> labels(args.numQueries * args.k);
    std::vector<typename C::T> distances(args.numQueries * args.k);
    GpuDistanceParams args_batch = args;
    args_batch.outDistances = distances.data();
    args_batch.ignoreOutDistances = false;
    args_batch.outIndices = labels.data();
    for (idx_t i = 0; i < args.numVectors; i += shard_size) {
        args_batch.numVectors = min(shard_size, args.numVectors - i);
        args_batch.vectors =
                (char*)args.vectors + distance_size * args.dims * i;
        args_batch.vectorNorms =
                args.vectorNorms ? args.vectorNorms + i : nullptr;
        bfKnn(prov, args_batch);
        for (auto& label : labels) {
            label += i;
        }
        heaps.addn_with_ids(args.k, distances.data(), labels.data(), args.k);
    }
    heaps.reorder();
}

void bfKnn_single_query_shard(
        GpuResourcesProvider* prov,
        const GpuDistanceParams& args,
        size_t vectorsMemoryLimit) {
    if (vectorsMemoryLimit == 0) {
        bfKnn(prov, args);
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            args.numVectors > 0, "bfKnn_tiling: numVectors must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.vectors,
            "bfKnn_tiling: vectors must be provided (passed null)");
    FAISS_THROW_IF_NOT_MSG(
            getDeviceForAddress(args.vectors) == -1,
            "bfKnn_tiling: vectors should be in CPU memory when vectorsMemoryLimit > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.vectorsRowMajor,
            "bfKnn_tiling: tiling vectors is only supported in row major mode");
    FAISS_THROW_IF_NOT_MSG(
            args.k > 0,
            "bfKnn_tiling: tiling vectors is only supported for k > 0");
    size_t distance_size = args.vectorType == DistanceDataType::F32 ? 4
            : (args.vectorType == DistanceDataType::F16 ||
               args.vectorType == DistanceDataType::BF16)
            ? 2
            : 0;
    FAISS_THROW_IF_NOT_MSG(
            distance_size > 0, "bfKnn_tiling: unknown vectorType");
    size_t shard_size = vectorsMemoryLimit / (args.dims * distance_size);
    FAISS_THROW_IF_NOT_MSG(
            shard_size > 0, "bfKnn_tiling: vectorsMemoryLimit is too low");
    if (args.numVectors <= shard_size) {
        bfKnn(prov, args);
        return;
    }
    if (is_similarity_metric(args.metric)) {
        if (args.outIndicesType == IndicesDataType::I64) {
            bfKnn_shard_database<CMin<float, int64_t>>(
                    prov, args, shard_size, distance_size);
        } else if (args.outIndicesType == IndicesDataType::I32) {
            bfKnn_shard_database<CMin<float, int32_t>>(
                    prov, args, shard_size, distance_size);
        } else {
            FAISS_THROW_MSG("bfKnn_tiling: unknown outIndicesType");
        }
    } else {
        if (args.outIndicesType == IndicesDataType::I64) {
            bfKnn_shard_database<CMax<float, int64_t>>(
                    prov, args, shard_size, distance_size);
        } else if (args.outIndicesType == IndicesDataType::I32) {
            bfKnn_shard_database<CMax<float, int32_t>>(
                    prov, args, shard_size, distance_size);
        } else {
            FAISS_THROW_MSG("bfKnn_tiling: unknown outIndicesType");
        }
    }
}

void bfKnn_tiling(
        GpuResourcesProvider* prov,
        const GpuDistanceParams& args,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit) {
    if (queriesMemoryLimit == 0) {
        bfKnn_single_query_shard(prov, args, vectorsMemoryLimit);
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            args.numQueries > 0, "bfKnn_tiling: numQueries must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.queries,
            "bfKnn_tiling: queries must be provided (passed null)");
    FAISS_THROW_IF_NOT_MSG(
            getDeviceForAddress(args.queries) == -1,
            "bfKnn_tiling: queries should be in CPU memory when queriesMemoryLimit > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.queriesRowMajor,
            "bfKnn_tiling: tiling queries is only supported in row major mode");
    FAISS_THROW_IF_NOT_MSG(
            args.k > 0,
            "bfKnn_tiling: tiling queries is only supported for k > 0");
    size_t distance_size = args.queryType == DistanceDataType::F32 ? 4
            : (args.queryType == DistanceDataType::F16 ||
               args.queryType == DistanceDataType::BF16)
            ? 2
            : 0;
    FAISS_THROW_IF_NOT_MSG(
            distance_size > 0, "bfKnn_tiling: unknown queryType");
    size_t label_size = args.outIndicesType == IndicesDataType::I64 ? 8
            : args.outIndicesType == IndicesDataType::I32           ? 4
                                                                    : 0;
    FAISS_THROW_IF_NOT_MSG(
            distance_size > 0, "bfKnn_tiling: unknown outIndicesType");
    size_t shard_size = queriesMemoryLimit /
            (args.k * (distance_size + label_size) + args.dims * distance_size);
    FAISS_THROW_IF_NOT_MSG(
            shard_size > 0, "bfKnn_tiling: queriesMemoryLimit is too low");
    FAISS_THROW_IF_NOT_MSG(
            args.outIndices,
            "bfKnn: outIndices must be provided (passed null)");
    for (idx_t i = 0; i < args.numQueries; i += shard_size) {
        GpuDistanceParams args_batch = args;
        args_batch.numQueries = min(shard_size, args.numQueries - i);
        args_batch.queries =
                (char*)args.queries + distance_size * args.dims * i;
        if (!args_batch.ignoreOutDistances) {
            args_batch.outDistances = args.outDistances + args.k * i;
        }
        args_batch.outIndices =
                (char*)args.outIndices + args.k * label_size * i;
        bfKnn_single_query_shard(prov, args_batch, vectorsMemoryLimit);
    }
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
