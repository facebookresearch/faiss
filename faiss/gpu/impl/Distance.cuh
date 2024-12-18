/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/impl/GeneralDistance.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

/// Calculates brute-force L2 distance between `vectors` and `queries`, not
/// performing top-k filtering.
/// FIXME: the output distances must fit in GPU memory
void runAllPairwiseL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 2, true>& vectors,
        bool vectorsRowMajor,
        // can be optionally pre-computed; nullptr if we
        // have to compute it upon the call
        Tensor<float, 1, true>* vectorNorms,
        Tensor<float, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances);

void runAllPairwiseL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<half, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances);

void runAllPairwiseL2Distance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<__nv_bfloat16, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<__nv_bfloat16, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances);

void runAllPairwiseIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances);

void runAllPairwiseIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<half, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances);

void runAllPairwiseIPDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<__nv_bfloat16, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<__nv_bfloat16, 2, true>& queries,
        bool queriesRowMajor,
        Tensor<float, 2, true>& outDistances);

/// Calculates brute-force L2 distance between `vectors` and
/// `queries`, returning the k closest results seen
void runL2Distance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<float, 2, true>& vectors,
        bool vectorsRowMajor,
        // can be optionally pre-computed; nullptr if we
        // have to compute it upon the call
        Tensor<float, 1, true>* vectorNorms,
        Tensor<float, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        // Do we care about `outDistances`? If not, we can
        // take shortcuts.
        bool ignoreOutDistances = false);

void runL2Distance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<half, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<half, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool ignoreOutDistances = false);

void runL2Distance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<__nv_bfloat16, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<__nv_bfloat16, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool ignoreOutDistances = false);

/// Calculates brute-force inner product distance between `vectors`
/// and `queries`, returning the k closest results seen
void runIPDistance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<float, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices);

void runIPDistance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<half, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<half, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices);

void runIPDistance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<__nv_bfloat16, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<__nv_bfloat16, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices);

//
// General distance implementation, assumes that all arguments are on the
// device. This is the top-level internal distance function to call to dispatch
// based on metric type.
//
template <typename T>
void allPairwiseDistanceOnDevice(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<T, 2, true>& queries,
        bool queriesRowMajor,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances) {
    DeviceScope ds(device);
    // We are guaranteed that all data arguments are resident on our preferred
    // `device` here

    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    if ((metric == faiss::MetricType::METRIC_L2) ||
        (metric == faiss::MetricType::METRIC_Lp && metricArg == 2)) {
        runAllPairwiseL2Distance(
                resources,
                stream,
                vectors,
                vectorsRowMajor,
                vectorNorms,
                queries,
                queriesRowMajor,
                outDistances);
    } else if (metric == faiss::MetricType::METRIC_INNER_PRODUCT) {
        runAllPairwiseIPDistance(
                resources,
                stream,
                vectors,
                vectorsRowMajor,
                queries,
                queriesRowMajor,
                outDistances);
    } else {
        //
        // General pairwise distance kernel
        //
        // The general distance kernel does not have specializations for
        // transpositions (NN, NT, TN); instead, the transposition is just
        // handled upon data load for now, which could result in poor data
        // loading behavior for NT / TN. This can be fixed at a later date if
        // desired, but efficiency is low versus GEMM anyways.
        //

        Tensor<T, 2> tVectorsDimInnermost = vectorsRowMajor
                ? vectors.transposeInnermost(1)
                : vectors.transposeInnermost(0);
        Tensor<T, 2> tQueriesDimInnermost = queriesRowMajor
                ? queries.transposeInnermost(1)
                : queries.transposeInnermost(0);

        if ((metric == faiss::MetricType::METRIC_L1) ||
            (metric == faiss::MetricType::METRIC_Lp && metricArg == 1)) {
            runGeneralDistanceKernel(
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    outDistances,
                    L1Distance(),
                    stream);
        } else if (metric == faiss::MetricType::METRIC_Lp && metricArg == -1) {
            // A way to test L2 distance
            runGeneralDistanceKernel(
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    outDistances,
                    L2Distance(),
                    stream);
        } else if (metric == faiss::MetricType::METRIC_Lp) {
            runGeneralDistanceKernel(
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    outDistances,
                    LpDistance(metricArg),
                    stream);
        } else if (metric == faiss::MetricType::METRIC_Linf) {
            runGeneralDistanceKernel(
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    outDistances,
                    LinfDistance(),
                    stream);
        } else if (metric == faiss::MetricType::METRIC_Canberra) {
            runGeneralDistanceKernel(
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    outDistances,
                    CanberraDistance(),
                    stream);
        } else if (metric == faiss::MetricType::METRIC_BrayCurtis) {
            runGeneralDistanceKernel(
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    outDistances,
                    BrayCurtisDistance(),
                    stream);
        } else if (metric == faiss::MetricType::METRIC_JensenShannon) {
            runGeneralDistanceKernel(
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    outDistances,
                    JensenShannonDistance(),
                    stream);
        } else if (metric == faiss::MetricType::METRIC_Jaccard) {
            runGeneralDistanceKernel(
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    outDistances,
                    JaccardSimilarity(),
                    stream);
        } else {
            FAISS_THROW_FMT("unimplemented metric type %d", metric);
        }
    }
}

//
// General distance implementation, assumes that all arguments are on the
// device. This is the top-level internal distance function to call to dispatch
// based on metric type.
//
template <typename T>
void bfKnnOnDevice(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 2, true>& vectors,
        bool vectorsRowMajor,
        Tensor<float, 1, true>* vectorNorms,
        Tensor<T, 2, true>& queries,
        bool queriesRowMajor,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool ignoreOutDistances) {
    DeviceScope ds(device);
    // We are guaranteed that all data arguments are resident on our preferred
    // `device` here

    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    if ((metric == faiss::MetricType::METRIC_L2) ||
        (metric == faiss::MetricType::METRIC_Lp && metricArg == 2)) {
        runL2Distance(
                resources,
                stream,
                vectors,
                vectorsRowMajor,
                vectorNorms,
                queries,
                queriesRowMajor,
                k,
                outDistances,
                outIndices);
    } else if (metric == faiss::MetricType::METRIC_INNER_PRODUCT) {
        runIPDistance(
                resources,
                stream,
                vectors,
                vectorsRowMajor,
                queries,
                queriesRowMajor,
                k,
                outDistances,
                outIndices);
    } else {
        //
        // General pairwise distance kernel
        //
        // The general distance kernel does not have specializations for
        // transpositions (NN, NT, TN); instead, the transposition is just
        // handled upon data load for now, which could result in poor data
        // loading behavior for NT / TN. This can be fixed at a later date if
        // desired, but efficiency is low versus GEMM anyways.
        //

        Tensor<T, 2> tVectorsDimInnermost = vectorsRowMajor
                ? vectors.transposeInnermost(1)
                : vectors.transposeInnermost(0);
        Tensor<T, 2> tQueriesDimInnermost = queriesRowMajor
                ? queries.transposeInnermost(1)
                : queries.transposeInnermost(0);

        if ((metric == faiss::MetricType::METRIC_L1) ||
            (metric == faiss::MetricType::METRIC_Lp && metricArg == 1)) {
            runGeneralDistance(
                    resources,
                    stream,
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    k,
                    L1Distance(),
                    outDistances,
                    outIndices);
        } else if (metric == faiss::MetricType::METRIC_Lp && metricArg == -1) {
            // A way to test L2 distance
            runGeneralDistance(
                    resources,
                    stream,
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    k,
                    L2Distance(),
                    outDistances,
                    outIndices);
        } else if (metric == faiss::MetricType::METRIC_Lp) {
            runGeneralDistance(
                    resources,
                    stream,
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    k,
                    LpDistance(metricArg),
                    outDistances,
                    outIndices);
        } else if (metric == faiss::MetricType::METRIC_Linf) {
            runGeneralDistance(
                    resources,
                    stream,
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    k,
                    LinfDistance(),
                    outDistances,
                    outIndices);
        } else if (metric == faiss::MetricType::METRIC_Canberra) {
            runGeneralDistance(
                    resources,
                    stream,
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    k,
                    CanberraDistance(),
                    outDistances,
                    outIndices);
        } else if (metric == faiss::MetricType::METRIC_BrayCurtis) {
            runGeneralDistance(
                    resources,
                    stream,
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    k,
                    BrayCurtisDistance(),
                    outDistances,
                    outIndices);
        } else if (metric == faiss::MetricType::METRIC_JensenShannon) {
            runGeneralDistance(
                    resources,
                    stream,
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    k,
                    JensenShannonDistance(),
                    outDistances,
                    outIndices);
        } else if (metric == faiss::MetricType::METRIC_Jaccard) {
            runGeneralDistance(
                    resources,
                    stream,
                    tVectorsDimInnermost,
                    tQueriesDimInnermost,
                    k,
                    JaccardSimilarity(),
                    outDistances,
                    outIndices);
        } else {
            FAISS_THROW_FMT("unimplemented metric type %d", metric);
        }
    }
}

} // namespace gpu
} // namespace faiss
