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

#include <cuda_fp16.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/Dataset.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/neighborhood_recall.cuh>

// Explicit owner kept by each test for the lifetime of its borrowing index.
faiss::gpu::Dataset makeSearchDataset(
        faiss::gpu::GpuResourcesProvider* resources,
        faiss::idx_t n,
        int dim,
        const void* data,
        faiss::NumericType type,
        int device) {
    auto source = faiss::gpu::Dataset::viewHost(data, n, dim, type, dim);
    return faiss::gpu::Dataset::makeDevicePadded(resources, source, device);
}

struct Options {
    Options() {
        numTrain = 2 * faiss::gpu::randVal(2000, 5000);
        dim = faiss::gpu::randVal(4, 10);
        numAdd = faiss::gpu::randVal(1000, 3000);

        graphDegree = faiss::gpu::randSelect({32, 64});
        intermediateGraphDegree = faiss::gpu::randSelect({64, 98});
        buildAlgo = faiss::gpu::randSelect(
                {faiss::gpu::graph_build_algo::IVF_PQ,
                 faiss::gpu::graph_build_algo::NN_DESCENT});

        numQuery = faiss::gpu::randVal(32, 100);
        k = faiss::gpu::randVal(10, 30);

        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    }

    std::string toString() const {
        std::stringstream str;
        str << "CAGRA device " << device << " numVecs " << numTrain << " dim "
            << dim << " graphDegree " << graphDegree
            << " intermediateGraphDegree " << intermediateGraphDegree
            << "buildAlgo " << static_cast<int>(buildAlgo) << " numQuery "
            << numQuery << " k " << k;

        return str.str();
    }

    int numTrain;
    int numAdd;
    int dim;
    size_t graphDegree;
    size_t intermediateGraphDegree;
    faiss::gpu::graph_build_algo buildAlgo;
    int numQuery;
    int k;
    int device;
};

void queryTest(faiss::MetricType metric, double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // train gpu index
        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);
        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim,
                trainVecs.data(),
                faiss::NumericType::Float32,
                opt.device));
        gpuIndex.train(*trainingDataset);

        // query
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        std::vector<float> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParams;
        cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data(),
                &cpuSearchParams);

        // test quality of searches
        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        gpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                testDistance.data(),
                testIndices.data());

        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                refDistanceDev.data(), opt.numQuery, opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                testDistance.data(), opt.numQuery, opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                test_ind_mds,
                ref_ind_mds,
                recall_score.view(),
                test_dis_mds_opt,
                ref_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float32_Query_L2) {
    queryTest(faiss::METRIC_L2, 0.98);
}

TEST(TestGpuIndexCagra, Float32_Query_IP) {
    queryTest(faiss::METRIC_INNER_PRODUCT, 0.98);
}

void queryTestFP16(faiss::MetricType metric, double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // train gpu index
        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);

        // Create half vector
        std::vector<__half> trainVecs_half(trainVecs.size());

        for (size_t i = 0; i < trainVecs.size(); ++i) {
            trainVecs_half[i] = __float2half(trainVecs[i]);
        }

        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim,
                trainVecs_half.data(),
                faiss::NumericType::Float16,
                opt.device));
        gpuIndex.train(*trainingDataset);

        // query
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        std::vector<float> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParams;
        cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data(),
                &cpuSearchParams);

        // test quality of searches
        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        // Create half vector
        std::vector<__half> queryVecs_half(queryVecs.size());

        for (size_t i = 0; i < queryVecs.size(); ++i) {
            queryVecs_half[i] = __float2half(queryVecs[i]);
        }
        gpuIndex.search_ex(
                opt.numQuery,
                queryVecs_half.data(),
                faiss::NumericType::Float16,
                opt.k,
                testDistance.data(),
                testIndices.data());

        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                refDistanceDev.data(), opt.numQuery, opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                testDistance.data(), opt.numQuery, opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                test_ind_mds,
                ref_ind_mds,
                recall_score.view(),
                test_dis_mds_opt,
                ref_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float16_Query_L2) {
    queryTestFP16(faiss::METRIC_L2, 0.98);
}

TEST(TestGpuIndexCagra, Float16_Query_IP) {
    queryTestFP16(faiss::METRIC_INNER_PRODUCT, 0.98);
}

void copyToTest(
        faiss::MetricType metric,
        double expected_recall,
        bool base_level_only) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numAdd, opt.dim, addVecs.data());
        }

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // train gpu index and copy to cpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim,
                trainVecs.data(),
                faiss::NumericType::Float32,
                opt.device));
        gpuIndex.train(*trainingDataset);

        faiss::IndexHNSWCagra copiedCpuIndex(
                opt.dim, opt.graphDegree / 2, metric);
        copiedCpuIndex.base_level_only = base_level_only;
        gpuIndex.copyTo(&copiedCpuIndex);
        copiedCpuIndex.hnsw.efConstruction = opt.k * 2;

        // add more vecs to copied cpu index
        if (!base_level_only) {
            copiedCpuIndex.add(opt.numAdd, addVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // add more vecs to cpu index
        if (!base_level_only) {
            cpuIndex.add(opt.numAdd, addVecs.data());
        }

        // query indexes
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        std::vector<float> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParams;
        cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data(),
                &cpuSearchParams);

        std::vector<float> copyRefDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> copyRefIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParamstwo;
        cpuSearchParamstwo.efSearch = opt.k * 2;
        copiedCpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                copyRefDistance.data(),
                copyRefIndices.data(),
                &cpuSearchParamstwo);

        // test quality of search
        auto gpuRes = res.getResources();

        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto copyRefDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                copyRefDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto copyRefIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                copyRefIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                refDistanceDev.data(), opt.numQuery, opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto copy_ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                copyRefDistanceDev.data(), opt.numQuery, opt.k);
        auto copy_ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        copy_ref_dis_mds);
        auto copy_ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        copyRefIndicesDev.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                copy_ref_ind_mds,
                ref_ind_mds,
                recall_score.view(),
                copy_ref_dis_mds_opt,
                ref_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float32_CopyTo_L2) {
    copyToTest(faiss::METRIC_L2, 0.98, false);
}

TEST(TestGpuIndexCagra, Float32_CopyTo_L2_BaseLevelOnly) {
    copyToTest(faiss::METRIC_L2, 0.98, true);
}

TEST(TestGpuIndexCagra, Float32_CopyTo_IP) {
    copyToTest(faiss::METRIC_INNER_PRODUCT, 0.98, false);
}

TEST(TestGpuIndexCagra, Float32_CopyTo_IP_BaseLevelOnly) {
    copyToTest(faiss::METRIC_INNER_PRODUCT, 0.98, true);
}

void copyToTestFP16(
        faiss::MetricType metric,
        double expected_recall,
        bool base_level_only) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numAdd, opt.dim, addVecs.data());
        }

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // train gpu index and copy to cpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        // Create half vector
        std::vector<__half> trainVecs_half(trainVecs.size());

        for (size_t i = 0; i < trainVecs.size(); ++i) {
            trainVecs_half[i] = __float2half(trainVecs[i]);
        }

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim,
                trainVecs_half.data(),
                faiss::NumericType::Float16,
                opt.device));
        gpuIndex.train(*trainingDataset);

        faiss::IndexHNSWCagra copiedCpuIndex(
                opt.dim,
                opt.graphDegree / 2,
                metric,
                faiss::NumericType::Float16);
        copiedCpuIndex.base_level_only = base_level_only;
        gpuIndex.copyTo(&copiedCpuIndex);
        copiedCpuIndex.hnsw.efConstruction = opt.k * 2;

        // train cpu index
        faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // query indexes
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        std::vector<float> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParams;
        cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data(),
                &cpuSearchParams);

        std::vector<float> copyRefDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> copyRefIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParamstwo;
        cpuSearchParamstwo.efSearch = opt.k * 2;
        copiedCpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                copyRefDistance.data(),
                copyRefIndices.data(),
                &cpuSearchParamstwo);

        // test quality of search
        auto gpuRes = res.getResources();

        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto copyRefDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                copyRefDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto copyRefIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                copyRefIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                refDistanceDev.data(), opt.numQuery, opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto copy_ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                copyRefDistanceDev.data(), opt.numQuery, opt.k);
        auto copy_ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        copy_ref_dis_mds);
        auto copy_ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        copyRefIndicesDev.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                copy_ref_ind_mds,
                ref_ind_mds,
                recall_score.view(),
                copy_ref_dis_mds_opt,
                ref_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

// For fp16, only base level copy is supported
TEST(TestGpuIndexCagra, Float16_CopyTo_L2_BaseLevelOnly) {
    copyToTestFP16(faiss::METRIC_L2, 0.98, true);
}

TEST(TestGpuIndexCagra, Float16_CopyTo_IP_BaseLevelOnly) {
    copyToTestFP16(faiss::METRIC_INNER_PRODUCT, 0.98, true);
}

void copyFromTest(faiss::MetricType metric, double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWCagra cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // convert to gpu index
        auto importedDataset = faiss::gpu::Dataset::makeDevicePaddedFromIndex(
                &res, cpuIndex, 0);
        faiss::gpu::GpuIndexCagra copiedGpuIndex(&res, opt.dim, metric);
        copiedGpuIndex.copyFrom(&cpuIndex, importedDataset);

        // train gpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim,
                trainVecs.data(),
                faiss::NumericType::Float32,
                opt.device));
        gpuIndex.train(*trainingDataset);

        // query
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<float, 2, true> copyTestDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> copyTestIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        copiedGpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                copyTestDistance.data(),
                copyTestIndices.data());

        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        gpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                testDistance.data(),
                testIndices.data());

        // test quality of searches
        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                testDistance.data(), opt.numQuery, opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        auto copy_test_dis_mds =
                raft::make_device_matrix_view<const float, int>(
                        copyTestDistance.data(), opt.numQuery, opt.k);
        auto copy_test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        copy_test_dis_mds);

        auto copy_test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        copyTestIndices.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                copy_test_ind_mds,
                test_ind_mds,
                recall_score.view(),
                copy_test_dis_mds_opt,
                test_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float32_CopyFrom_L2) {
    copyFromTest(faiss::METRIC_L2, 0.98);
}

TEST(TestGpuIndexCagra, Float32_CopyFrom_IP) {
    copyFromTest(faiss::METRIC_INNER_PRODUCT, 0.98);
}

void copyFromTestFP16(faiss::MetricType metric, double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWCagra cpuIndex(
                opt.dim,
                opt.graphDegree / 2,
                metric,
                faiss::NumericType::Float16);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // convert to gpu index
        auto importedDataset = faiss::gpu::Dataset::makeDevicePaddedFromIndex(
                &res, cpuIndex, 0);
        faiss::gpu::GpuIndexCagra copiedGpuIndex(&res, opt.dim, metric);
        copiedGpuIndex.copyFrom(&cpuIndex, importedDataset);

        // train gpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        // faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        // gpuIndex.train(opt.numTrain, trainVecs.data());

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);

        // Create half vector
        std::vector<__half> trainVecs_half(trainVecs.size());

        for (size_t i = 0; i < trainVecs.size(); ++i) {
            trainVecs_half[i] = __float2half(trainVecs[i]);
        }

        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim,
                trainVecs_half.data(),
                faiss::NumericType::Float16,
                opt.device));
        gpuIndex.train(*trainingDataset);

        // query
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        // Create half vector
        std::vector<__half> queryVecs_half(queryVecs.size());

        for (size_t i = 0; i < queryVecs.size(); ++i) {
            queryVecs_half[i] = __float2half(queryVecs[i]);
        }

        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<float, 2, true> copyTestDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> copyTestIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        copiedGpuIndex.search_ex(
                opt.numQuery,
                queryVecs_half.data(),
                faiss::NumericType::Float16,
                opt.k,
                copyTestDistance.data(),
                copyTestIndices.data());

        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        gpuIndex.search_ex(
                opt.numQuery,
                queryVecs_half.data(),
                faiss::NumericType::Float16,
                opt.k,
                testDistance.data(),
                testIndices.data());

        // test quality of searches
        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                testDistance.data(), opt.numQuery, opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        auto copy_test_dis_mds =
                raft::make_device_matrix_view<const float, int>(
                        copyTestDistance.data(), opt.numQuery, opt.k);
        auto copy_test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        copy_test_dis_mds);

        auto copy_test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        copyTestIndices.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                copy_test_ind_mds,
                test_ind_mds,
                recall_score.view(),
                copy_test_dis_mds_opt,
                test_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float16_CopyFrom_L2) {
    copyFromTestFP16(faiss::METRIC_L2, 0.98);
}

TEST(TestGpuIndexCagra, Float16_CopyFrom_IP) {
    copyFromTestFP16(faiss::METRIC_INNER_PRODUCT, 0.98);
}

void testIDSelectorCagra(faiss::MetricType metricType) {
    Options opt;
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    if (metricType == faiss::METRIC_INNER_PRODUCT) {
        faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
    }

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexCagraConfig config;
    config.device = opt.device;
    config.graph_degree = opt.graphDegree;
    config.intermediate_graph_degree = opt.intermediateGraphDegree;
    // Use only IVF_PQ to avoid NN_DESCENT + IP combination
    config.build_algo = faiss::gpu::graph_build_algo::IVF_PQ;

    std::optional<faiss::gpu::Dataset> trainingDataset;

    faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metricType, config);
    trainingDataset.emplace(makeSearchDataset(
            &res,
            opt.numTrain,
            opt.dim,
            trainVecs.data(),
            faiss::NumericType::Float32,
            opt.device));
    gpuIndex.train(*trainingDataset);

    auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
    if (metricType == faiss::METRIC_INNER_PRODUCT) {
        faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
    }

    faiss::gpu::TestIDSelectorStruct selector_struct(opt.numTrain);
    faiss::gpu::SearchParametersCagra search_params;
    for (auto& [selectorName, selector] : selector_struct.selector_map) {
        search_params.sel = selector.get();
        faiss::gpu::testIDSelectorSearch(
                &gpuIndex,
                &search_params,
                queryVecs,
                opt.numQuery,
                opt.k,
                selectorName);
    }
}

TEST(TestGpuIndexCagra, IDSelector_L2) {
    testIDSelectorCagra(faiss::METRIC_L2);
}

TEST(TestGpuIndexCagra, IDSelector_IP) {
    testIDSelectorCagra(faiss::METRIC_INNER_PRODUCT);
}

namespace {
using faiss::idx_t;
using faiss::NumericType;
using faiss::gpu::Dataset;
using faiss::gpu::GpuIndexCagra;
using faiss::gpu::StandardGpuResources;

struct TrainDataset {
    template <typename Index, typename Data>
    auto operator()(Index& index, Data&& data) const
            -> decltype(index.train(std::forward<Data>(data)));
};

struct AddDataset {
    template <typename Index, typename Data>
    auto operator()(Index& index, Data&& data) const
            -> decltype(index.add(std::forward<Data>(data)));
};

struct UpdateDataset {
    template <typename Index, typename Data>
    auto operator()(Index& index, Data&& data) const
            -> decltype(index.updateDataset(std::forward<Data>(data)));
};

struct CopyDataset {
    template <typename Index, typename CpuIndex, typename Data>
    auto operator()(Index& index, const CpuIndex* source, Data&& data) const
            -> decltype(index.copyFrom(source, std::forward<Data>(data)));
};

struct CloneDataset {
    template <typename Data>
    auto operator()(Data&& data) const -> decltype(faiss::gpu::indexCpuToGpu(
            nullptr,
            0,
            nullptr,
            std::forward<Data>(data)));
};

template <typename Operation, typename... Args>
constexpr bool acceptsDatasetLvaluesOnly =
        std::is_invocable_v<Operation, Args..., Dataset&> &&
        std::is_invocable_v<Operation, Args..., const Dataset&> &&
        !std::is_invocable_v<Operation, Args..., Dataset&&> &&
        !std::is_invocable_v<Operation, Args..., const Dataset&&>;

static_assert(acceptsDatasetLvaluesOnly<TrainDataset, GpuIndexCagra&>);
static_assert(acceptsDatasetLvaluesOnly<AddDataset, GpuIndexCagra&>);
static_assert(acceptsDatasetLvaluesOnly<UpdateDataset, GpuIndexCagra&>);
static_assert(acceptsDatasetLvaluesOnly<
              CopyDataset,
              GpuIndexCagra&,
              const faiss::IndexHNSWCagra*>);
static_assert(acceptsDatasetLvaluesOnly<CloneDataset>);

constexpr idx_t kRows = 512;

faiss::gpu::GpuIndexCagraConfig config() {
    faiss::gpu::GpuIndexCagraConfig out;
    out.device = 0;
    out.graph_degree = 16;
    out.intermediate_graph_degree = 32;
    out.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
    return out;
}

std::vector<float> floatData(int dim) {
    std::mt19937 random(1234);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> data(kRows * dim);
    for (auto& x : data) {
        x = value(random);
    }
    return data;
}

void checkFloatSearch(faiss::Index& index, const float* queries) {
    std::vector<float> distances(8);
    std::vector<idx_t> labels(8);
    index.search(8, queries, 1, distances.data(), labels.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int selfHits = 0;
    for (int i = 0; i < 8; ++i) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], kRows);
        selfHits += labels[i] == i;
    }
    EXPECT_GE(selfHits, 6);
}

TEST(TestGpuIndexCagra, PointerConstructionRejectedThroughBaseInterfaces) {
    StandardGpuResources resources;
    auto floats = floatData(16);
    auto floatView = Dataset::viewHost(
            floats.data(), kRows, 16, NumericType::Float32, 16);
    GpuIndexCagra cagra(&resources, 16, faiss::METRIC_L2, config());
    faiss::Index* base = &cagra;
    EXPECT_THROW(base->train(kRows, floats.data()), faiss::FaissException);
    EXPECT_THROW(base->add(kRows, floats.data()), faiss::FaissException);
    EXPECT_THROW(
            base->train_ex(kRows, floats.data(), NumericType::Float32),
            faiss::FaissException);
    EXPECT_THROW(
            base->add_ex(kRows, floats.data(), NumericType::Float32),
            faiss::FaissException);
    EXPECT_FALSE(cagra.is_trained);

    faiss::gpu::GpuIndexFlatL2 flat(&resources, 16);
    base = &flat;
    EXPECT_THROW(base->train(floatView), faiss::FaissException);
    EXPECT_THROW(base->add(floatView), faiss::FaissException);
    EXPECT_NO_THROW(base->add(kRows, floats.data()));
    EXPECT_EQ(flat.ntotal, kRows);
}

TEST(TestGpuIndexCagra, BruteForceRequiresMultipleDevices) {
    StandardGpuResources resources;
    auto values = floatData(20);
    auto source = Dataset::viewHost(
            values.data(), kRows, 20, NumericType::Float32, 20);
    auto options = config();
    options.build_algo = faiss::gpu::graph_build_algo::BRUTE_FORCE;
    GpuIndexCagra index(&resources, 20, faiss::METRIC_L2, options);
    try {
        index.train(source);
        FAIL() << "Single-device BRUTE_FORCE must be rejected";
    } catch (const faiss::FaissException& error) {
        EXPECT_NE(
                std::string(error.what()).find("BRUTE_FORCE"),
                std::string::npos);
    }
    EXPECT_FALSE(index.is_trained);
}

TEST(TestGpuIndexCagra, CompatibleStandardDeviceBorrowsWithoutOwning) {
    // Twenty Float32 columns satisfy 16-byte rows without dim % 16 == 0.
    constexpr int dim = 20;
    StandardGpuResources resources;
    auto values = floatData(dim);
    auto source = Dataset::viewHost(
            values.data(), kRows, dim, NumericType::Float32, dim);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, config());
    {
        auto standard = Dataset::viewDevice(
                storage.getData(), kRows, dim, NumericType::Float32, dim, 0);
        EXPECT_EQ(standard.getLayout(), Dataset::Layout::Standard);
        EXPECT_EQ(standard.getData(), storage.getData());
        EXPECT_FALSE(standard.ownsData());
        EXPECT_TRUE(standard.isCagraCompatible());
        faiss::Index* base = &index;
        base->train(standard);
    }
    EXPECT_TRUE(index.isSearchReady());
    auto movedStorage = std::move(storage);
    checkFloatSearch(index, values.data());
    EXPECT_THROW(index.train(movedStorage), faiss::FaissException);
    index.reset();
    EXPECT_FALSE(index.isSearchReady());
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);
    index.add(movedStorage);
    checkFloatSearch(index, values.data());
}

TEST(TestGpuIndexCagra, GraphOnlyHostExportAndExplicitAttachment) {
    constexpr int dim = 19;
    constexpr int stride = 23;
    StandardGpuResources resources;
    auto values = floatData(dim);
    std::vector<float> strided(kRows * stride, 1234.0f);
    for (idx_t i = 0; i < kRows; ++i) {
        std::copy_n(values.data() + i * dim, dim, strided.data() + i * stride);
    }
    auto source = Dataset::viewHost(
            strided.data(), kRows, dim, NumericType::Float32, stride);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, config());
    faiss::Index* base = &index;
    base->train(source);
    EXPECT_TRUE(index.is_trained);
    EXPECT_FALSE(index.isSearchReady());
    float distance;
    idx_t label;
    EXPECT_THROW(
            index.search(1, values.data(), 1, &distance, &label),
            faiss::FaissException);

    faiss::IndexHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    std::vector<float> exported(values.size());
    cpu.reconstruct_n(0, kRows, exported.data());
    EXPECT_EQ(exported, values);
    EXPECT_EQ(strided[stride - 1], 1234.0f);
    faiss::RangeSearchResult range(8);
    cpu.range_search(8, values.data(), 1.0e9f, &range);
    for (int i = 0; i < 8; ++i) {
        EXPECT_GT(range.lims[i + 1], range.lims[i]);
    }

    index.updateDataset(storage);
    EXPECT_TRUE(index.isSearchReady());
    const auto graph = index.get_knngraph();
    auto wrongRows = Dataset::viewDevice(
            storage.getData(),
            kRows - 1,
            dim,
            NumericType::Float32,
            storage.getRowStride(),
            0);
    EXPECT_THROW(index.updateDataset(wrongRows), faiss::FaissException);
    auto wrongType = Dataset::viewDevice(
            storage.getData(),
            kRows,
            dim,
            NumericType::UInt8,
            storage.getRowStride() * sizeof(float),
            0);
    EXPECT_THROW(index.updateDataset(wrongType), faiss::FaissException);
    auto wrongDim = Dataset::viewDevice(
            storage.getData(),
            kRows,
            dim - 1,
            NumericType::Float32,
            storage.getRowStride(),
            0);
    EXPECT_THROW(index.updateDataset(wrongDim), faiss::FaissException);
    EXPECT_THROW(index.updateDataset(source), faiss::FaissException);
    if (faiss::gpu::getNumDevices() > 1) {
        auto wrongDevice = Dataset::makeDevicePadded(&resources, source, 1);
        EXPECT_THROW(index.updateDataset(wrongDevice), faiss::FaissException);
    }
    EXPECT_TRUE(index.isSearchReady());
    EXPECT_EQ(index.get_knngraph(), graph);
    checkFloatSearch(index, values.data());

    // CPU serialization and re-import retain their existing ownership behavior.
    faiss::VectorIOWriter writer;
    faiss::write_index(&cpu, &writer);
    faiss::VectorIOReader reader;
    reader.data = std::move(writer.data);
    std::unique_ptr<faiss::Index> restored(faiss::read_index(&reader));
    auto* restoredCagra = dynamic_cast<faiss::IndexHNSWCagra*>(restored.get());
    ASSERT_NE(restoredCagra, nullptr);
    auto importedStorage =
            Dataset::makeDevicePaddedFromIndex(&resources, *restoredCagra, 0);
    std::unique_ptr<faiss::Index> imported(
            faiss::gpu::indexCpuToGpu(
                    &resources, 0, restored.get(), importedStorage));
    EXPECT_THROW(
            faiss::gpu::index_cpu_to_gpu(&resources, 0, restored.get()),
            faiss::FaissException);
    restored.reset();
    index.reset();
    checkFloatSearch(*imported, values.data());
    checkFloatSearch(cpu, values.data());
}

TEST(TestGpuIndexCagra, DatasetAsyncRebindResetAndDestruction) {
    constexpr int dim = 19;
    constexpr int nq = 64;
    auto values = floatData(dim);
    faiss::gpu::DeviceScope scope(0);
    cudaStream_t alternateStream;
    ASSERT_EQ(
            cudaStreamCreateWithFlags(&alternateStream, cudaStreamNonBlocking),
            cudaSuccess);
    const auto destroyStream = [](cudaStream_t stream) {
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    };
    // cuVS descriptor caches retain their allocation stream in the resource
    // handle, so the external stream must also outlive those resources.
    std::unique_ptr<
            std::remove_pointer_t<cudaStream_t>,
            decltype(destroyStream)>
            streamOwner(alternateStream, destroyStream);
    StandardGpuResources resources;
    auto source = Dataset::viewHost(
            values.data(), kRows, dim, NumericType::Float32, dim);
    std::optional<Dataset> first(
            Dataset::makeDevicePadded(&resources, source, 0));
    std::optional<Dataset> second(
            Dataset::makeDevicePadded(&resources, source, 0));
    auto index = std::make_unique<GpuIndexCagra>(
            &resources, dim, faiss::METRIC_L2, config());
    index->train(*first);

    auto gpuResources = resources.getResources();
    auto originalStream = gpuResources->getDefaultStream(0);
    auto queries = faiss::gpu::toDeviceTemporary(
            gpuResources.get(), values, originalStream);
    auto alloc = faiss::gpu::makeTempAlloc(
            faiss::gpu::AllocType::Other, originalStream);
    faiss::gpu::DeviceTensor<float, 2, true> distances(
            gpuResources.get(), alloc, {nq, 1});
    faiss::gpu::DeviceTensor<idx_t, 2, true> labels(
            gpuResources.get(), alloc, {nq, 1});
    resources.setDefaultStream(0, alternateStream);
    auto checkResults = [&](cudaStream_t stream) {
        std::vector<float> actualDistances(nq);
        std::vector<idx_t> actualLabels(nq);
        faiss::gpu::fromDevice(distances, actualDistances.data(), stream);
        faiss::gpu::fromDevice(labels, actualLabels.data(), stream);
        int selfHits = 0;
        for (int i = 0; i < nq; ++i) {
            EXPECT_GE(actualLabels[i], 0);
            EXPECT_LT(actualLabels[i], kRows);
            if (actualLabels[i] == i) {
                ++selfHits;
                EXPECT_NEAR(actualDistances[i], 0, 1e-5);
            }
        }
        EXPECT_GE(selfHits, nq * 3 / 4);
    };

    // These searches keep both inputs and outputs on device and queue work.
    // Rebind/reset must finish use of the old allocation before its release.
    index->search(nq, queries.data(), 1, distances.data(), labels.data());
    index->updateDataset(*second);
    first.reset();
    checkResults(alternateStream);
    index->search(nq, queries.data(), 1, distances.data(), labels.data());
    index->reset();
    second.reset();
    checkResults(alternateStream);

    // Build on B, then search on A and destroy without an explicit reset.
    // Keep the caller-owned data alive to isolate graph-storage destruction.
    std::optional<Dataset> third(
            Dataset::makeDevicePadded(&resources, source, 0));
    index->train(*third);
    resources.revertDefaultStream(0);
    index->search(nq, queries.data(), 1, distances.data(), labels.data());
    index.reset();
    checkResults(originalStream);
    third.reset();
}

TEST(TestGpuIndexCagra, MultiGpuDatasetBuildAndCpuExport) {
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
        GTEST_SKIP() << "Requires at least two GPUs";
    }
    constexpr int dim = 32;
    constexpr idx_t rows = 4096;
    StandardGpuResources resources;
    std::mt19937 random(1234);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> values(rows * dim);
    for (auto& x : values) {
        x = value(random);
    }
    auto source = Dataset::viewHost(
            values.data(), rows, dim, NumericType::Float32, dim);
    auto options = config();
    options.devices = {0, 1};
    // Exact cluster graphs bound the work while exercising the full
    // multi-GPU graph-build and CPU-export path.
    options.build_algo = faiss::gpu::graph_build_algo::BRUTE_FORCE;
    options.graph_degree = 32;
    options.intermediate_graph_degree = 64;
    options.all_neighbors_params.n_clusters = 4;
    options.all_neighbors_params.overlap_factor = 2;
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, options);
    index.train(source);
    ASSERT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, rows);
    EXPECT_FALSE(index.isSearchReady());
    EXPECT_EQ(index.get_knngraph().size(), rows * options.graph_degree);
    EXPECT_THROW(index.updateDataset(source), faiss::FaissException);
    faiss::IndexHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    EXPECT_EQ(cpu.ntotal, rows);
    std::vector<float> exported(values.size());
    cpu.reconstruct_n(0, rows, exported.data());
    EXPECT_EQ(exported, values);
    index.reset();
    float distance;
    idx_t label;
    cpu.search(1, values.data(), 1, &distance, &label);
    EXPECT_EQ(label, 0);
    EXPECT_FLOAT_EQ(distance, 0.0f);
}

template <typename T>
void checkStridedDeviceGraphOnly(NumericType type) {
    constexpr int dim = 19;
    constexpr int stride = 23;
    faiss::gpu::DeviceScope scope(0);
    StandardGpuResources resources;
    std::vector<T> values(kRows * dim);
    if constexpr (std::is_same_v<T, float>) {
        values = floatData(dim);
    } else {
        std::mt19937 random(9012);
        for (auto& x : values) {
            x = static_cast<T>(static_cast<int>(random() % 201) - 100);
        }
    }
    // Device row gaps must be zero, even when the physical stride is not
    // compatible with CAGRA search storage.
    std::vector<T> strided(kRows * stride, static_cast<T>(0));
    for (idx_t i = 0; i < kRows; ++i) {
        std::copy_n(values.data() + i * dim, dim, strided.data() + i * stride);
    }
    auto gpuResources = resources.getResources();
    auto stream = gpuResources->getDefaultStream(0);
    auto deviceStorage =
            faiss::gpu::toDeviceTemporary(gpuResources.get(), strided, stream);
    auto source = Dataset::viewDevice(
            deviceStorage.data(), kRows, dim, type, stride, 0);
    EXPECT_EQ(source.getLayout(), Dataset::Layout::Standard);
    EXPECT_FALSE(source.ownsData());
    EXPECT_FALSE(source.isCagraCompatible());
    std::optional<Dataset> searchStorage;
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, config());
    index.train(source);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, kRows);
    EXPECT_FALSE(index.isSearchReady());
    float distance;
    idx_t label;
    EXPECT_THROW(
            index.search_ex(1, values.data(), type, 1, &distance, &label),
            faiss::FaissException);

    faiss::IndexHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    EXPECT_EQ(cpu.get_numeric_type(), type);
    std::vector<float> exported(values.size());
    cpu.reconstruct_n(0, kRows, exported.data());
    std::vector<float> expected(values.size());
    std::transform(values.begin(), values.end(), expected.begin(), [](T x) {
        return static_cast<float>(x);
    });
    EXPECT_EQ(exported, expected);
    EXPECT_FALSE(index.isSearchReady());
    EXPECT_THROW(index.updateDataset(source), faiss::FaissException);
    const auto graph = index.get_knngraph();

    searchStorage.emplace(Dataset::makeDevicePadded(&resources, source, 0));
    index.updateDataset(*searchStorage);
    EXPECT_TRUE(index.isSearchReady());
    EXPECT_EQ(index.get_knngraph(), graph);
    std::vector<float> distances(8);
    std::vector<idx_t> labels(8);
    index.search_ex(8, values.data(), type, 1, distances.data(), labels.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int selfHits = 0;
    for (int i = 0; i < 8; ++i) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], kRows);
        if (labels[i] == i) {
            ++selfHits;
            EXPECT_NEAR(distances[i], 0.0f, 1.0f);
        }
    }
    EXPECT_GE(selfHits, 6);
    std::vector<T> after(strided.size());
    faiss::gpu::fromDevice(deviceStorage, after.data(), stream);
    EXPECT_EQ(
            std::memcmp(
                    after.data(), strided.data(), strided.size() * sizeof(T)),
            0);
}

TEST(TestGpuIndexCagra, StridedDeviceFloat32GraphOnly) {
    checkStridedDeviceGraphOnly<float>(NumericType::Float32);
}

TEST(TestGpuIndexCagra, StridedDeviceFloat16GraphOnly) {
    checkStridedDeviceGraphOnly<half>(NumericType::Float16);
}

TEST(TestGpuIndexCagra, StridedDeviceInt8GraphOnly) {
    checkStridedDeviceGraphOnly<int8_t>(NumericType::Int8);
}

template <typename T>
void checkTypedExport(NumericType type) {
    constexpr int dim = 19;
    StandardGpuResources resources;
    std::mt19937 random(9012);
    std::vector<T> values(kRows * dim);
    for (auto& x : values) {
        x = static_cast<T>(static_cast<int>(random() % 201) - 100);
    }
    auto original = values;
    auto source = Dataset::viewHost(values.data(), kRows, dim, type, dim);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, config());
    index.train(storage);
    faiss::IndexHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    EXPECT_EQ(cpu.get_numeric_type(), type);
    std::vector<T> afterExport(values.size());
    storage.copyToHost(afterExport.data());
    EXPECT_EQ(
            std::memcmp(
                    afterExport.data(),
                    original.data(),
                    original.size() * sizeof(T)),
            0);
    EXPECT_EQ(
            std::memcmp(
                    values.data(),
                    original.data(),
                    original.size() * sizeof(T)),
            0);

    auto importedStorage =
            Dataset::makeDevicePaddedFromIndex(&resources, cpu, 0);
    EXPECT_EQ(importedStorage.getNumericType(), type);
    std::vector<T> decoded(values.size());
    importedStorage.copyToHost(decoded.data());
    EXPECT_EQ(
            std::memcmp(
                    decoded.data(),
                    original.data(),
                    original.size() * sizeof(T)),
            0);
    GpuIndexCagra imported(&resources, dim, faiss::METRIC_L2, config());
    imported.copyFrom(&cpu, importedStorage);
    EXPECT_TRUE(imported.isSearchReady());
    EXPECT_EQ(imported.get_knngraph(), index.get_knngraph());
    // Exercise the unchanged typed query interface for both the original and
    // re-imported graph, including Int8 whose Python build path is unavailable.
    for (auto* searched : {&index, &imported}) {
        std::vector<float> distances(8);
        std::vector<idx_t> labels(8);
        searched->search_ex(
                8, original.data(), type, 1, distances.data(), labels.data());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        int selfHits = 0;
        for (int i = 0; i < 8; ++i) {
            EXPECT_GE(labels[i], 0);
            EXPECT_LT(labels[i], kRows);
            if (labels[i] == i) {
                ++selfHits;
                EXPECT_NEAR(distances[i], 0.0f, 1.0f);
            }
        }
        EXPECT_GE(selfHits, 6);
    }
}

TEST(TestGpuIndexCagra, HalfAndInt8ExportPreserveCallerData) {
    checkTypedExport<half>(NumericType::Float16);
    checkTypedExport<int8_t>(NumericType::Int8);
}
} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
