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
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/utils/distances.h>
#include <cstddef>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <optional>
#include <vector>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/neighborhood_recall.cuh>

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

        faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

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

        faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);

        // Create half vector
        std::vector<__half> trainVecs_half(trainVecs.size());

        for (size_t i = 0; i < trainVecs.size(); ++i) {
            trainVecs_half[i] = __float2half(trainVecs[i]);
        }

        gpuIndex.train(
                opt.numTrain,
                static_cast<void*>(trainVecs_half.data()),
                faiss::NumericType::Float16);

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
        gpuIndex.search(
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

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

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

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        gpuIndex.train(
                opt.numTrain,
                static_cast<void*>(trainVecs_half.data()),
                faiss::NumericType::Float16);

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
        faiss::gpu::GpuIndexCagra copiedGpuIndex(&res, cpuIndex.d, metric);
        copiedGpuIndex.copyFrom(&cpuIndex);

        // train gpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

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
        faiss::gpu::GpuIndexCagra copiedGpuIndex(&res, cpuIndex.d, metric);
        copiedGpuIndex.copyFrom(&cpuIndex, faiss::NumericType::Float16);

        // train gpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        // faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        // gpuIndex.train(opt.numTrain, trainVecs.data());

        faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);

        // Create half vector
        std::vector<__half> trainVecs_half(trainVecs.size());

        for (size_t i = 0; i < trainVecs.size(); ++i) {
            trainVecs_half[i] = __float2half(trainVecs[i]);
        }

        gpuIndex.train(
                opt.numTrain,
                static_cast<void*>(trainVecs_half.data()),
                faiss::NumericType::Float16);

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
        copiedGpuIndex.search(
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
        gpuIndex.search(
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

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
