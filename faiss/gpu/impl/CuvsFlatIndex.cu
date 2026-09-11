// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <faiss/gpu/utils/CuvsFilterConvert.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsFlatIndex.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>

#include <optional>
#include <vector>

#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/unary_op.cuh>

namespace faiss {
namespace gpu {

// Compatibility note: release/26.10's brute-force C API does not accept
// caller-provided L2 norms. Preserve Faiss's cached-norm optimization through
// C++ only for L2; all other flat searches use the C API below.
template <typename T>
void searchWithPrecomputedL2Norms(
        GpuResources* resources,
        Tensor<T, 2, true>& database,
        Tensor<float, 1, true>& norms,
        Tensor<T, 2, true>& queries,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        float metricArg,
        const IDSelector* sel) {
    auto& handle = resources->getRaftHandleCurrentDevice();
    auto databaseView = raft::make_device_matrix_view<const T, int64_t>(
            database.data(), database.getSize(0), database.getSize(1));
    auto queryView = raft::make_device_matrix_view<const T, int64_t>(
            queries.data(), queries.getSize(0), queries.getSize(1));
    auto indexView = raft::make_device_matrix_view<idx_t, int64_t>(
            outIndices.data(), outIndices.getSize(0), outIndices.getSize(1));
    auto distanceView = raft::make_device_matrix_view<float, int64_t>(
            outDistances.data(),
            outDistances.getSize(0),
            outDistances.getSize(1));
    std::optional<raft::device_vector_view<const float, int64_t>> normsView =
            raft::make_device_vector_view(norms.data(), norms.getSize(0));
    cuvs::neighbors::brute_force::index<T, float> index(
            handle,
            databaseView,
            normsView,
            cuvs::distance::DistanceType::L2Expanded,
            metricArg);

    if (sel) {
        raft::core::bitset<uint32_t, int64_t> bitset(
                handle, database.getSize(0), false);
        convert_to_bitset(resources, *sel, bitset.view());
        cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(
                bitset.view());
        cuvs::neighbors::brute_force::search(
                handle,
                cuvs::neighbors::brute_force::search_params{},
                index,
                queryView,
                indexView,
                distanceView,
                filter);
    } else {
        cuvs::neighbors::brute_force::search(
                handle,
                cuvs::neighbors::brute_force::search_params{},
                index,
                queryView,
                indexView,
                distanceView);
    }
}

CuvsFlatIndex::CuvsFlatIndex(
        GpuResources* res,
        int dim,
        bool useFloat16,
        MemorySpace space)
        : FlatIndex(res, dim, useFloat16, space) {}

void CuvsFlatIndex::query(
        Tensor<float, 2, true>& input,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool exactDistance,
        const IDSelector* sel) {
    if (useFloat16_) {
        // We need to convert the input to float16 for comparison to ourselves
        auto stream = resources_->getDefaultStreamCurrentDevice();
        auto inputHalf = convertTensorTemporary<float, half, 2>(
                resources_, stream, input);
        CuvsFlatIndex::query(
                inputHalf,
                k,
                metric,
                metricArg,
                outDistances,
                outIndices,
                exactDistance,
                sel);
    } else {
        raft::device_resources& handle =
                resources_->getRaftHandleCurrentDevice();

        auto inds = raft::make_device_matrix_view<idx_t, int64_t>(
                outIndices.data(),
                outIndices.getSize(0),
                outIndices.getSize(1));
        auto dists = raft::make_device_matrix_view<float, int64_t>(
                outDistances.data(),
                outDistances.getSize(0),
                outDistances.getSize(1));

        auto distance = metricFaissToCuvs(metric, exactDistance);
        if (metric == MetricType::METRIC_L2) {
            searchWithPrecomputedL2Norms(
                    resources_,
                    vectors_,
                    norms_,
                    input,
                    outDistances,
                    outIndices,
                    metricArg,
                    sel);
        } else {
            auto indexTensor = makeCuvsTensor(
                    vectors_.data(), vectors_.getSize(0), vectors_.getSize(1));
            auto queryTensor = makeCuvsTensor(
                    input.data(), input.getSize(0), input.getSize(1));
            auto indicesTensor = makeCuvsTensor(
                    outIndices.data(),
                    outIndices.getSize(0),
                    outIndices.getSize(1));
            auto distancesTensor = makeCuvsTensor(
                    outDistances.data(),
                    outDistances.getSize(0),
                    outDistances.getSize(1));

            cuvsBruteForceIndex_t index = nullptr;
            cuvsCheck(
                    cuvsBruteForceIndexCreate(&index),
                    "cuvsBruteForceIndexCreate");
            CuvsUniquePtr<cuvsBruteForceIndex, cuvsBruteForceIndexDestroy>
                    indexHolder(index);
            auto cuvsResources = cuvsResourcesFromGpuResources(resources_);
            cuvsCheck(
                    cuvsBruteForceBuild(
                            cuvsResources,
                            indexTensor.get(),
                            distance,
                            metricArg,
                            index),
                    "cuvsBruteForceBuild");
            CuvsFilter filter(resources_, sel, vectors_.getSize(0));
            cuvsCheck(
                    cuvsBruteForceSearch(
                            cuvsResources,
                            index,
                            queryTensor.get(),
                            indicesTensor.get(),
                            distancesTensor.get(),
                            filter.get()),
                    "cuvsBruteForceSearch");
        }
        if (metric == MetricType::METRIC_Lp) {
            raft::linalg::unary_op(
                    handle,
                    raft::make_const_mdspan(dists),
                    dists,
                    [metricArg] __device__(const float& a) {
                        return powf(a, metricArg);
                    });
        } else if (metric == MetricType::METRIC_JensenShannon) {
            raft::linalg::unary_op(
                    handle,
                    raft::make_const_mdspan(dists),
                    dists,
                    [] __device__(const float& a) { return powf(a, 2); });
        }
    }
}

void CuvsFlatIndex::query(
        Tensor<half, 2, true>& vecs,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool exactDistance,
        const IDSelector* sel) {
    FAISS_ASSERT(useFloat16_);

    raft::device_resources& handle = resources_->getRaftHandleCurrentDevice();

    auto dists = raft::make_device_matrix_view<float, int64_t>(
            outDistances.data(),
            outDistances.getSize(0),
            outDistances.getSize(1));

    auto distance = metricFaissToCuvs(metric, exactDistance);

    if (metric == MetricType::METRIC_L2) {
        searchWithPrecomputedL2Norms(
                resources_,
                vectorsHalf_,
                norms_,
                vecs,
                outDistances,
                outIndices,
                metricArg,
                sel);
    } else {
        auto indexTensor = makeCuvsTensor(
                vectorsHalf_.data(),
                vectorsHalf_.getSize(0),
                vectorsHalf_.getSize(1));
        auto queryTensor =
                makeCuvsTensor(vecs.data(), vecs.getSize(0), vecs.getSize(1));
        auto indicesTensor = makeCuvsTensor(
                outIndices.data(),
                outIndices.getSize(0),
                outIndices.getSize(1));
        auto distancesTensor = makeCuvsTensor(
                outDistances.data(),
                outDistances.getSize(0),
                outDistances.getSize(1));

        cuvsBruteForceIndex_t index = nullptr;
        cuvsCheck(
                cuvsBruteForceIndexCreate(&index), "cuvsBruteForceIndexCreate");
        CuvsUniquePtr<cuvsBruteForceIndex, cuvsBruteForceIndexDestroy>
                indexHolder(index);
        auto cuvsResources = cuvsResourcesFromGpuResources(resources_);
        cuvsCheck(
                cuvsBruteForceBuild(
                        cuvsResources,
                        indexTensor.get(),
                        distance,
                        metricArg,
                        index),
                "cuvsBruteForceBuild");
        CuvsFilter filter(resources_, sel, vectorsHalf_.getSize(0));
        cuvsCheck(
                cuvsBruteForceSearch(
                        cuvsResources,
                        index,
                        queryTensor.get(),
                        indicesTensor.get(),
                        distancesTensor.get(),
                        filter.get()),
                "cuvsBruteForceSearch");
    }
    if (metric == MetricType::METRIC_Lp) {
        raft::linalg::unary_op(
                handle,
                raft::make_const_mdspan(dists),
                dists,
                [metricArg] __device__(const float& a) {
                    return powf(a, metricArg);
                });
    } else if (metric == MetricType::METRIC_JensenShannon) {
        raft::linalg::unary_op(
                handle,
                raft::make_const_mdspan(dists),
                dists,
                [] __device__(const float& a) { return powf(a, 2); });
    }
}

} // namespace gpu
} // namespace faiss
