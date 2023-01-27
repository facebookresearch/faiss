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

namespace faiss {
namespace gpu {

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
        // The brute-force API only supports an interface for i32 indices only,
        // so we must create an output i32 buffer then convert back
        DeviceTensor<int, 2, true> tOutIntIndices(
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
                tOutIntIndices,
                args.ignoreOutDistances);

        // Convert and copy int indices out
        auto tOutIndices = toDeviceTemporary<idx_t, 2>(
                res,
                device,
                (idx_t*)args.outIndices,
                stream,
                {args.numQueries, args.k});

        // Convert int to idx_t
        convertTensor<int, idx_t, 2>(stream, tOutIntIndices, tOutIndices);

        // Copy back if necessary
        fromDevice<idx_t, 2>(tOutIndices, (idx_t*)args.outIndices, stream);

    } else if (args.outIndicesType == IndicesDataType::I32) {
        // We can use the brute-force API directly, as it takes i32 indices
        // FIXME: convert to int32_t everywhere?
        static_assert(sizeof(int) == 4, "");

        auto tOutIntIndices = toDeviceTemporary<int, 2>(
                res,
                device,
                (int*)args.outIndices,
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
                tOutIntIndices,
                args.ignoreOutDistances);

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

    if (args.vectorType == DistanceDataType::F32) {
        bfKnnConvert<float>(res, args);
    } else if (args.vectorType == DistanceDataType::F16) {
        bfKnnConvert<half>(res, args);
    } else {
        FAISS_THROW_MSG("unknown vectorType");
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
        int numVectors,
        // A region of memory size numQueries x dims, with dims
        // innermost
        const float* queries,
        bool queriesRowMajor,
        int numQueries,
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
