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

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/gpu/Dataset.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexBinaryCagra.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
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
        numTrain = 2 * faiss::gpu::randVal(1000, 2000);
        dim = faiss::gpu::randVal(1, 10) * 8;
        numAdd = faiss::gpu::randVal(500, 1500);

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

void queryTest(
        faiss::gpu::graph_build_algo build_algo,
        double expected_recall) {
    for (int tries = 0; tries < 3; ++tries) {
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
        config.build_algo = build_algo;
        if (build_algo == faiss::gpu::graph_build_algo::NN_DESCENT) {
            config.nn_descent_niter = 20;
        }

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexBinaryCagra gpuIndex(&res, cpuIndex.d, config);
        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim / 8,
                trainVecs.data(),
                faiss::NumericType::UInt8,
                opt.device));
        gpuIndex.train(*trainingDataset);

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

TEST(TestGpuIndexBinaryCagra, Query_NN_DESCENT) {
    queryTest(faiss::gpu::graph_build_algo::NN_DESCENT, 0.98);
}

TEST(TestGpuIndexBinaryCagra, Query_ITERATIVE_SEARCH) {
    queryTest(faiss::gpu::graph_build_algo::ITERATIVE_SEARCH, 0.98);
}

void copyToTest(
        faiss::gpu::graph_build_algo build_algo,
        double expected_recall) {
    for (int tries = 0; tries < 3; ++tries) {
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
        config.build_algo = build_algo;
        if (build_algo == faiss::gpu::graph_build_algo::NN_DESCENT) {
            config.nn_descent_niter = 20;
        }

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexBinaryCagra gpuIndex(&res, opt.dim, config);
        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim / 8,
                trainVecs.data(),
                faiss::NumericType::UInt8,
                opt.device));
        gpuIndex.train(*trainingDataset);

        faiss::IndexBinaryHNSWCagra copiedCpuIndex(
                opt.dim, opt.graphDegree / 2);
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

TEST(TestGpuIndexBinaryCagra, CopyTo_NN_DESCENT) {
    copyToTest(faiss::gpu::graph_build_algo::NN_DESCENT, 0.98);
}

TEST(TestGpuIndexBinaryCagra, CopyTo_ITERATIVE_SEARCH) {
    copyToTest(faiss::gpu::graph_build_algo::ITERATIVE_SEARCH, 0.98);
}

void copyFromTest(
        faiss::gpu::graph_build_algo build_algo,
        double expected_recall) {
    for (int tries = 0; tries < 3; ++tries) {
        Options opt;

        auto trainVecs = faiss::gpu::randBinaryVecs(opt.numTrain, opt.dim);

        // train cpu index
        faiss::IndexBinaryHNSWCagra cpuIndex(opt.dim, opt.graphDegree / 2);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // convert to gpu index
        const auto* cpuStorage =
                dynamic_cast<const faiss::IndexBinaryFlat*>(cpuIndex.storage);
        ASSERT_NE(cpuStorage, nullptr);
        auto cpuDataset = faiss::gpu::Dataset::viewHost(
                cpuStorage->xb.data(),
                cpuIndex.ntotal,
                cpuIndex.d / 8,
                faiss::NumericType::UInt8,
                cpuIndex.d / 8);
        auto importedDataset =
                faiss::gpu::Dataset::makeDevicePadded(&res, cpuDataset, 0);
        res.getResources()->syncDefaultStream(0);
        faiss::gpu::GpuIndexBinaryCagra copiedGpuIndex(&res, cpuIndex.d);
        copiedGpuIndex.copyFrom(&cpuIndex, importedDataset);

        // train gpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = build_algo;
        if (build_algo == faiss::gpu::graph_build_algo::NN_DESCENT) {
            config.nn_descent_niter = 20;
        }

        std::optional<faiss::gpu::Dataset> trainingDataset;

        faiss::gpu::GpuIndexBinaryCagra gpuIndex(&res, opt.dim, config);
        trainingDataset.emplace(makeSearchDataset(
                &res,
                opt.numTrain,
                opt.dim / 8,
                trainVecs.data(),
                faiss::NumericType::UInt8,
                opt.device));
        gpuIndex.train(*trainingDataset);

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

TEST(TestGpuIndexBinaryCagra, CopyFrom_NN_DESCENT) {
    copyFromTest(faiss::gpu::graph_build_algo::NN_DESCENT, 0.98);
}

TEST(TestGpuIndexBinaryCagra, CopyFrom_ITERATIVE_SEARCH) {
    copyFromTest(faiss::gpu::graph_build_algo::ITERATIVE_SEARCH, 0.98);
}

void testIDSelectorBinaryCagra(std::string selectorName) {
    Options opt;
    auto trainVecs = faiss::gpu::randBinaryVecs(opt.numTrain, opt.dim);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexCagraConfig config;
    config.device = opt.device;
    config.graph_degree = opt.graphDegree;
    config.intermediate_graph_degree = opt.intermediateGraphDegree;
    config.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
    config.nn_descent_niter = 20;

    std::optional<faiss::gpu::Dataset> trainingDataset;

    faiss::gpu::GpuIndexBinaryCagra gpuIndex(&res, opt.dim, config);
    trainingDataset.emplace(makeSearchDataset(
            &res,
            opt.numTrain,
            opt.dim / 8,
            trainVecs.data(),
            faiss::NumericType::UInt8,
            opt.device));
    gpuIndex.train(*trainingDataset);

    auto queryVecs = faiss::gpu::randBinaryVecs(opt.numQuery, opt.dim);
    faiss::gpu::TestIDSelectorStruct selector_struct(opt.numTrain);
    faiss::gpu::SearchParametersCagra search_params;
    // TODO: For CI test purposes only, remove this
    // for (auto& [selectorName, selector] : selector_struct.selector_map) {
    auto selector = selector_struct.selector_map[selectorName].get();
    search_params.sel = selector;
    std::vector<int> distances(opt.numQuery * opt.k, 0);
    std::vector<faiss::idx_t> labels(opt.numQuery * opt.k, -1);
    gpuIndex.search(
            opt.numQuery,
            queryVecs.data(),
            opt.k,
            distances.data(),
            labels.data(),
            &search_params);
    for (int i = 0; i < opt.numQuery * opt.k; ++i) {
        if (labels[i] >= 0) {
            EXPECT_TRUE(selector->is_member(labels[i]))
                    << "Label " << labels[i] << " @ " << i << " not in "
                    << selectorName << " selector";
        }
    }
    //}
}

TEST(TestGpuIndexBinaryCagra, IDSelector_Range) {
    testIDSelectorBinaryCagra("Range");
}
TEST(TestGpuIndexBinaryCagra, IDSelector_Array) {
    testIDSelectorBinaryCagra("Array");
}
TEST(TestGpuIndexBinaryCagra, IDSelector_Batch) {
    testIDSelectorBinaryCagra("Batch");
}
TEST(TestGpuIndexBinaryCagra, IDSelector_Bitmap) {
    testIDSelectorBinaryCagra("Bitmap");
}
TEST(TestGpuIndexBinaryCagra, IDSelector_Not) {
    testIDSelectorBinaryCagra("Not");
}
TEST(TestGpuIndexBinaryCagra, IDSelector_And) {
    testIDSelectorBinaryCagra("And");
}
TEST(TestGpuIndexBinaryCagra, IDSelector_Or) {
    testIDSelectorBinaryCagra("Or");
}
TEST(TestGpuIndexBinaryCagra, IDSelector_XOr) {
    testIDSelectorBinaryCagra("XOr");
}

namespace {
using faiss::idx_t;
using faiss::NumericType;
using faiss::gpu::Dataset;
using faiss::gpu::GpuIndexBinaryCagra;
using faiss::gpu::StandardGpuResources;

constexpr idx_t kRows = 512;

faiss::gpu::GpuIndexCagraConfig config() {
    faiss::gpu::GpuIndexCagraConfig out;
    out.device = 0;
    out.graph_degree = 16;
    out.intermediate_graph_degree = 32;
    out.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
    return out;
}

std::vector<uint8_t> binaryData(int bytes) {
    std::mt19937 random(4321);
    std::vector<uint8_t> data(kRows * bytes);
    for (auto& x : data) {
        x = static_cast<uint8_t>(random());
    }
    return data;
}

void checkBinarySearch(faiss::IndexBinary& index, const uint8_t* queries) {
    std::vector<int> distances(8);
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

TEST(TestGpuIndexBinaryCagra,
     PointerConstructionRejectedThroughBaseInterfaces) {
    StandardGpuResources resources;
    auto bytes = binaryData(16);
    GpuIndexBinaryCagra binary(&resources, 128, config());
    faiss::IndexBinary* binaryBase = &binary;
    EXPECT_THROW(binaryBase->train(kRows, bytes.data()), faiss::FaissException);
    EXPECT_THROW(binaryBase->add(kRows, bytes.data()), faiss::FaissException);
    EXPECT_THROW(
            binaryBase->train_ex(kRows, bytes.data(), NumericType::UInt8),
            faiss::FaissException);
    EXPECT_THROW(
            binaryBase->add_ex(kRows, bytes.data(), NumericType::UInt8),
            faiss::FaissException);
    EXPECT_FALSE(binary.is_trained);

    faiss::IndexBinaryFlat binaryFlat(128);
    binaryBase = &binaryFlat;
    EXPECT_NO_THROW(binaryBase->add(kRows, bytes.data()));
    EXPECT_EQ(binaryFlat.ntotal, kRows);
}

TEST(TestGpuIndexBinaryCagra, DatasetAsyncRebindResetAndDestruction) {
    constexpr int dim = 19;
    constexpr int nq = 64;
    auto values = binaryData(dim);
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
            values.data(), kRows, dim, NumericType::UInt8, dim);
    std::optional<Dataset> first(
            Dataset::makeDevicePadded(&resources, source, 0));
    std::optional<Dataset> second(
            Dataset::makeDevicePadded(&resources, source, 0));
    auto index = std::make_unique<GpuIndexBinaryCagra>(
            &resources, dim * 8, config());
    index->train(*first);

    auto gpuResources = resources.getResources();
    auto originalStream = gpuResources->getDefaultStream(0);
    auto queries = faiss::gpu::toDeviceTemporary(
            gpuResources.get(), values, originalStream);
    auto alloc = faiss::gpu::makeTempAlloc(
            faiss::gpu::AllocType::Other, originalStream);
    faiss::gpu::DeviceTensor<int, 2, true> distances(
            gpuResources.get(), alloc, {nq, 1});
    faiss::gpu::DeviceTensor<idx_t, 2, true> labels(
            gpuResources.get(), alloc, {nq, 1});
    resources.setDefaultStream(0, alternateStream);
    auto checkResults = [&](cudaStream_t stream) {
        std::vector<int> actualDistances(nq);
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

TEST(TestGpuIndexBinaryCagra, DatasetGraphOnlyAttachmentAndCpuRoundTrip) {
    constexpr int bytes = 19;
    StandardGpuResources resources;
    auto values = binaryData(bytes);
    auto source = Dataset::viewHost(
            values.data(), kRows, bytes, NumericType::UInt8, bytes);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    GpuIndexBinaryCagra index(&resources, bytes * 8, config());
    index.add(source);
    EXPECT_FALSE(index.isSearchReady());
    int distance;
    idx_t label;
    EXPECT_THROW(
            index.search(1, values.data(), 1, &distance, &label),
            faiss::FaissException);
    faiss::IndexBinaryHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    std::vector<uint8_t> exported(values.size());
    for (idx_t i = 0; i < kRows; ++i) {
        cpu.reconstruct(i, exported.data() + i * bytes);
    }
    EXPECT_EQ(exported, values);
    index.updateDataset(storage);
    EXPECT_TRUE(index.isSearchReady());
    auto wrongRows = Dataset::viewDevice(
            storage.getData(),
            kRows - 1,
            bytes,
            NumericType::UInt8,
            storage.getRowStride(),
            0);
    EXPECT_THROW(index.updateDataset(wrongRows), faiss::FaissException);
    EXPECT_TRUE(index.isSearchReady());
    checkBinarySearch(index, values.data());

    faiss::VectorIOWriter writer;
    faiss::write_index_binary(&cpu, &writer);
    faiss::VectorIOReader reader;
    reader.data = std::move(writer.data);
    std::unique_ptr<faiss::IndexBinary> restored(
            faiss::read_index_binary(&reader));
    auto* restoredCagra =
            dynamic_cast<faiss::IndexBinaryHNSWCagra*>(restored.get());
    ASSERT_NE(restoredCagra, nullptr);
    const auto* restoredStorage =
            dynamic_cast<const faiss::IndexBinaryFlat*>(restoredCagra->storage);
    ASSERT_NE(restoredStorage, nullptr);
    auto restoredDataset = Dataset::viewHost(
            restoredStorage->xb.data(),
            restoredCagra->ntotal,
            restoredCagra->d / 8,
            NumericType::UInt8,
            restoredCagra->d / 8);
    auto importedStorage =
            Dataset::makeDevicePadded(&resources, restoredDataset, 0);
    GpuIndexBinaryCagra imported(&resources, bytes * 8, config());
    imported.copyFrom(restoredCagra, importedStorage);
    EXPECT_THROW(
            faiss::gpu::index_binary_cpu_to_gpu(&resources, 0, restored.get()),
            faiss::FaissException);
    resources.getResources()->syncDefaultStream(0);
    restored.reset();
    index.reset();
    EXPECT_FALSE(index.isSearchReady());
    index.train(storage);
    checkBinarySearch(index, values.data());
    checkBinarySearch(imported, values.data());
}

void checkStridedDeviceGraphOnly(faiss::gpu::graph_build_algo buildAlgo) {
    constexpr int bytes = 19;
    constexpr int stride = 23;
    faiss::gpu::DeviceScope scope(0);
    StandardGpuResources resources;
    auto values = binaryData(bytes);
    // Device row gaps must be zero, even when the physical stride is not
    // compatible with CAGRA search storage.
    std::vector<uint8_t> strided(kRows * stride, 0);
    for (idx_t i = 0; i < kRows; ++i) {
        std::copy_n(
                values.data() + i * bytes, bytes, strided.data() + i * stride);
    }
    auto gpuResources = resources.getResources();
    auto stream = gpuResources->getDefaultStream(0);
    auto deviceStorage =
            faiss::gpu::toDeviceTemporary(gpuResources.get(), strided, stream);
    auto source = Dataset::viewDevice(
            deviceStorage.data(), kRows, bytes, NumericType::UInt8, stride, 0);
    EXPECT_EQ(source.getLayout(), Dataset::Layout::Standard);
    EXPECT_FALSE(source.ownsData());
    EXPECT_FALSE(source.isCagraCompatible());
    std::optional<Dataset> searchStorage;
    auto options = config();
    options.build_algo = buildAlgo;
    GpuIndexBinaryCagra index(&resources, bytes * 8, options);
    index.train(source);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, kRows);
    EXPECT_FALSE(index.isSearchReady());
    int distance;
    idx_t label;
    EXPECT_THROW(
            index.search(1, values.data(), 1, &distance, &label),
            faiss::FaissException);

    faiss::IndexBinaryHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    std::vector<uint8_t> exported(values.size());
    for (idx_t i = 0; i < kRows; ++i) {
        cpu.reconstruct(i, exported.data() + i * bytes);
    }
    EXPECT_EQ(exported, values);
    EXPECT_FALSE(index.isSearchReady());
    EXPECT_THROW(index.updateDataset(source), faiss::FaissException);
    const auto graph = index.get_knngraph();

    searchStorage.emplace(Dataset::makeDevicePadded(&resources, source, 0));
    index.updateDataset(*searchStorage);
    EXPECT_TRUE(index.isSearchReady());
    EXPECT_EQ(index.get_knngraph(), graph);
    checkBinarySearch(index, values.data());
    std::vector<uint8_t> after(strided.size());
    faiss::gpu::fromDevice(deviceStorage, after.data(), stream);
    EXPECT_EQ(after, strided);
}

TEST(TestGpuIndexBinaryCagra, StridedDeviceGraphOnly) {
    checkStridedDeviceGraphOnly(faiss::gpu::graph_build_algo::NN_DESCENT);
}

TEST(TestGpuIndexBinaryCagra, StridedDeviceIvfPqFallbackGraphOnly) {
    checkStridedDeviceGraphOnly(faiss::gpu::graph_build_algo::IVF_PQ);
}

TEST(TestGpuIndexBinaryCagra, DatasetAlignedStandardDeviceSearch) {
    constexpr int bytes = 16;
    StandardGpuResources resources;
    auto values = binaryData(bytes);
    auto source = Dataset::viewHost(
            values.data(), kRows, bytes, NumericType::UInt8, bytes);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    auto standard = Dataset::viewDevice(
            storage.getData(), kRows, bytes, NumericType::UInt8, bytes, 0);
    GpuIndexBinaryCagra index(&resources, bytes * 8, config());
    index.train(standard);
    EXPECT_TRUE(index.isSearchReady());
    EXPECT_FALSE(standard.ownsData());
    checkBinarySearch(index, values.data());
}

} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
