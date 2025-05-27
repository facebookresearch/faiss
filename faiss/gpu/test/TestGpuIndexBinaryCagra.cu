// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <faiss/IndexBinaryHNSW.h>
#include <faiss/gpu/GpuIndexBinaryCagra.h>
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
        dim = faiss::gpu::randVal(1, 20) * 8;
        numAdd = faiss::gpu::randVal(1000, 3000);

        graphDegree = faiss::gpu::randSelect({32, 64});
        intermediateGraphDegree = faiss::gpu::randSelect({64, 98});

        numQuery = faiss::gpu::randVal(32, 100);
        k = faiss::gpu::randVal(10, 30);

        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    }

    std::string toString() const {
        std::stringstream str;
        str << "CAGRA device " << device << " numVecs " << numTrain << " dim "
            << dim << " graphDegree " << graphDegree
            << " intermediateGraphDegree " << intermediateGraphDegree
            << " numQuery " << numQuery << " k " << k;

        return str.str();
    }

    int numTrain;
    int numAdd;
    int dim;
    size_t graphDegree;
    size_t intermediateGraphDegree;
    int numQuery;
    int k;
    int device;
};

void queryTest(double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;

        auto trainVecs = faiss::gpu::randBinaryVecs(opt.numTrain, opt.dim);

        // train cpu index
        faiss::IndexBinaryHNSW cpuIndex(opt.dim, opt.graphDegree / 2);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // train gpu index
        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;

        faiss::gpu::GpuIndexBinaryCagra gpuIndex(&res, cpuIndex.d, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

        // query
        auto queryVecs = faiss::gpu::randBinaryVecs(opt.numQuery, opt.dim);

        std::vector<int> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        // faiss::SearchParametersHNSW cpuSearchParams;
        // cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data());

        // test quality of searches
        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<int, 2, true> testDistance(
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
                reinterpret_cast<const float*>(refDistanceDev.data()),
                opt.numQuery,
                opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                reinterpret_cast<const float*>(testDistance.data()),
                opt.numQuery,
                opt.k);
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

TEST(TestGpuIndexBinaryCagra, Query) {
    queryTest(0.98);
}

void copyToTest(double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;

        auto trainVecs = faiss::gpu::randBinaryVecs(opt.numTrain, opt.dim);
        auto addVecs = faiss::gpu::randBinaryVecs(opt.numAdd, opt.dim);

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // train gpu index and copy to cpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;

        faiss::gpu::GpuIndexBinaryCagra gpuIndex(&res, opt.dim, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

        faiss::IndexBinaryHNSW copiedCpuIndex(opt.dim, opt.graphDegree / 2);
        gpuIndex.copyTo(&copiedCpuIndex);
        copiedCpuIndex.hnsw.efConstruction = opt.k * 2;

        // add more vecs to copied cpu index
        copiedCpuIndex.add(opt.numAdd, addVecs.data());

        // train cpu index
        faiss::IndexBinaryHNSW cpuIndex(opt.dim, opt.graphDegree / 2);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // add more vecs to cpu index
        cpuIndex.add(opt.numAdd, addVecs.data());

        // query indexes
        auto queryVecs = faiss::gpu::randBinaryVecs(opt.numQuery, opt.dim);

        std::vector<int> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        // faiss::SearchParametersHNSW cpuSearchParams;
        // cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data());

        std::vector<int> copyRefDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> copyRefIndices(opt.numQuery * opt.k, -1);
        // faiss::SearchParametersHNSW cpuSearchParamstwo;
        // cpuSearchParamstwo.efSearch = opt.k * 2;
        copiedCpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                copyRefDistance.data(),
                copyRefIndices.data());

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
                reinterpret_cast<const float*>(refDistanceDev.data()),
                opt.numQuery,
                opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto copy_ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                reinterpret_cast<const float*>(copyRefDistanceDev.data()),
                opt.numQuery,
                opt.k);
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

TEST(TestGpuIndexBinaryCagra, CopyTo) {
    copyToTest(0.98);
}

void copyFromTest(double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;

        auto trainVecs = faiss::gpu::randBinaryVecs(opt.numTrain, opt.dim);

        // train cpu index
        faiss::IndexBinaryHNSW cpuIndex(opt.dim, opt.graphDegree / 2);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // convert to gpu index
        faiss::gpu::GpuIndexBinaryCagra copiedGpuIndex(&res, cpuIndex.d);
        copiedGpuIndex.copyFrom(&cpuIndex);

        // train gpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;

        faiss::gpu::GpuIndexBinaryCagra gpuIndex(&res, opt.dim, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

        // query
        auto queryVecs = faiss::gpu::randBinaryVecs(opt.numQuery, opt.dim);

        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<int, 2, true> copyTestDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> copyTestIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        copiedGpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                copyTestDistance.data(),
                copyTestIndices.data());

        faiss::gpu::DeviceTensor<int, 2, true> testDistance(
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
                reinterpret_cast<const float*>(testDistance.data()),
                opt.numQuery,
                opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        auto copy_test_dis_mds =
                raft::make_device_matrix_view<const float, int>(
                        reinterpret_cast<const float*>(copyTestDistance.data()),
                        opt.numQuery,
                        opt.k);
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

TEST(TestGpuIndexBinaryCagra, CopyFrom) {
    copyFromTest(0.98);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
