// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexIVFRaBitQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace {

struct Options {
    Options() {
        numAdd = faiss::gpu::randVal(2000, 5000);
        dim = faiss::gpu::randVal(32, 128);
        numCentroids = std::sqrt(static_cast<float>(numAdd));
        nprobe = std::min(faiss::gpu::randVal(1, 40), numCentroids);
        numQuery = faiss::gpu::randVal(4, 16);
        k = std::min(faiss::gpu::randVal(5, 20), numAdd / 40);
        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
        bitsPerDim = 3;
    }

    std::string toString() const {
        std::stringstream str;
        str << "IVF-RaBitQ device " << device << " numVecs " << numAdd
            << " dim " << dim << " numCentroids " << numCentroids << " nprobe "
            << nprobe << " numQuery " << numQuery << " k " << k
            << " bitsPerDim " << bitsPerDim;
        return str.str();
    }

    int numAdd;
    int dim;
    int numCentroids;
    int nprobe;
    int numQuery;
    int k;
    int device;
    uint32_t bitsPerDim;
};

void expectValidResults(
        const std::vector<float>& distances,
        const std::vector<faiss::idx_t>& labels,
        int numDatabaseVectors) {
    EXPECT_TRUE(
            std::all_of(distances.begin(), distances.end(), [](float distance) {
                return std::isfinite(distance);
            }));
    EXPECT_TRUE(
            std::all_of(
                    labels.begin(),
                    labels.end(),
                    [numDatabaseVectors](faiss::idx_t label) {
                        return label >= 0 && label < numDatabaseVectors;
                    }));
}

double recallAtK(
        const std::vector<faiss::idx_t>& reference,
        const std::vector<faiss::idx_t>& actual,
        int numQueries,
        int k) {
    size_t hits = 0;
    for (int query = 0; query < numQueries; ++query) {
        const auto begin = reference.begin() + query * k;
        std::unordered_set<faiss::idx_t> expected(begin, begin + k);
        for (int neighbor = 0; neighbor < k; ++neighbor) {
            hits += expected.erase(actual[query * k + neighbor]);
        }
    }
    return static_cast<double>(hits) / (numQueries * k);
}

void expectSortedDistances(
        const std::vector<float>& distances,
        int numQueries,
        int k) {
    for (int query = 0; query < numQueries; ++query) {
        for (int neighbor = 1; neighbor < k; ++neighbor) {
            EXPECT_LE(
                    distances[query * k + neighbor - 1],
                    distances[query * k + neighbor]);
        }
    }
}

TEST(TestGpuIndexIVFRaBitQ, BuildAndSearch) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    Options opt;

    auto database = faiss::gpu::randVecs(opt.numAdd, opt.dim);
    auto queries = faiss::gpu::randVecs(opt.numQuery, opt.dim);

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.device = opt.device;
    config.use_cuvs = true;
    config.bitsPerDim = opt.bitsPerDim;
    config.nprobe = opt.nprobe;

    faiss::gpu::GpuIndexIVFRaBitQ index(
            &resources, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    faiss::Index* genericIndex = &index;
    EXPECT_THROW(
            genericIndex->train_ex(
                    opt.numAdd, database.data(), faiss::NumericType::Float16),
            faiss::FaissException);
    genericIndex->train_ex(
            opt.numAdd, database.data(), faiss::NumericType::Float32);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, opt.numAdd);

    std::vector<float> distances(opt.numQuery * opt.k);
    std::vector<faiss::idx_t> labels(opt.numQuery * opt.k);
    index.search(
            opt.numQuery,
            queries.data(),
            opt.k,
            distances.data(),
            labels.data());

    expectValidResults(distances, labels, opt.numAdd);
    expectSortedDistances(distances, opt.numQuery, opt.k);

    faiss::gpu::SearchParametersIVFRaBitQ params;
    params.nprobe = opt.numCentroids;
    params.searchMode = faiss::gpu::IVFRaBitQSearchMode::LUT16;
    index.search(
            opt.numQuery,
            queries.data(),
            opt.k,
            distances.data(),
            labels.data(),
            &params);
    expectValidResults(distances, labels, opt.numAdd);
    expectSortedDistances(distances, opt.numQuery, opt.k);

    EXPECT_THROW(index.add(1, database.data()), faiss::FaissException);
    const faiss::idx_t customId = 42;
    EXPECT_THROW(
            genericIndex->add_with_ids(1, database.data(), &customId),
            faiss::FaissException);
    EXPECT_THROW(
            genericIndex->add_with_ids_ex(
                    1, database.data(), faiss::NumericType::Float32, &customId),
            faiss::FaissException);
    EXPECT_THROW(
            genericIndex->add_with_ids(1, nullptr, &customId),
            faiss::FaissException);
    EXPECT_THROW(
            genericIndex->add_with_ids_ex(
                    1, nullptr, faiss::NumericType::Float32, &customId),
            faiss::FaissException);
    EXPECT_EQ(index.ntotal, opt.numAdd);

    index.reset();
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);

    EXPECT_THROW(
            genericIndex->add_with_ids(1, database.data(), &customId),
            faiss::FaissException);
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_THROW(
            genericIndex->add_ex(
                    1, database.data(), faiss::NumericType::Float16),
            faiss::FaissException);
    genericIndex->add_ex(
            opt.numAdd, database.data(), faiss::NumericType::Float32);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, opt.numAdd);
    index.search(
            opt.numQuery,
            queries.data(),
            opt.k,
            distances.data(),
            labels.data());
    expectValidResults(distances, labels, opt.numAdd);
    expectSortedDistances(distances, opt.numQuery, opt.k);
}

TEST(TestGpuIndexIVFRaBitQ, RecallAgainstExactIndex) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    Options opt;
    opt.numAdd = 4096;
    opt.dim = 64;
    opt.numCentroids = 32;
    opt.nprobe = opt.numCentroids;
    opt.numQuery = 128;
    opt.k = 10;

    auto database = faiss::gpu::randVecs(opt.numAdd, opt.dim);
    auto queries = faiss::gpu::randVecs(opt.numQuery, opt.dim);

    faiss::IndexFlatL2 exactIndex(opt.dim);
    exactIndex.add(opt.numAdd, database.data());

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.device = opt.device;
    config.use_cuvs = true;
    config.bitsPerDim = opt.bitsPerDim;
    config.nprobe = opt.nprobe;
    config.searchMode = faiss::gpu::IVFRaBitQSearchMode::QUANT4;

    faiss::gpu::GpuIndexIVFRaBitQ gpu_index(
            &resources, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpu_index.add(opt.numAdd, database.data());

    std::vector<float> exactDistances(opt.numQuery * opt.k);
    std::vector<faiss::idx_t> exactLabels(opt.numQuery * opt.k);
    exactIndex.search(
            opt.numQuery,
            queries.data(),
            opt.k,
            exactDistances.data(),
            exactLabels.data());

    std::vector<float> actualDistances(opt.numQuery * opt.k);
    std::vector<faiss::idx_t> actualLabels(opt.numQuery * opt.k);
    for (const auto mode :
         {faiss::gpu::IVFRaBitQSearchMode::LUT16,
          faiss::gpu::IVFRaBitQSearchMode::LUT32,
          faiss::gpu::IVFRaBitQSearchMode::QUANT4,
          faiss::gpu::IVFRaBitQSearchMode::QUANT8}) {
        SCOPED_TRACE(static_cast<int>(mode));
        faiss::gpu::SearchParametersIVFRaBitQ params;
        params.nprobe = opt.numCentroids;
        params.searchMode = mode;
        gpu_index.search(
                opt.numQuery,
                queries.data(),
                opt.k,
                actualDistances.data(),
                actualLabels.data(),
                &params);

        expectValidResults(actualDistances, actualLabels, opt.numAdd);
        expectSortedDistances(actualDistances, opt.numQuery, opt.k);
        EXPECT_GE(
                recallAtK(exactLabels, actualLabels, opt.numQuery, opt.k), 0.5)
                << opt.toString();
    }
}

TEST(TestGpuIndexIVFRaBitQ, NormalizesMissingResults) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    constexpr int dim = 32;
    constexpr int numDatabaseVectors = 64;
    constexpr int numCentroids = 8;
    constexpr int k = numDatabaseVectors + 4;

    auto database = faiss::gpu::randVecs(numDatabaseVectors, dim);
    auto query = faiss::gpu::randVecs(1, dim);

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.use_cuvs = true;
    config.nprobe = numCentroids;

    faiss::gpu::GpuIndexIVFRaBitQ index(
            &resources, dim, numCentroids, faiss::METRIC_L2, config);
    index.add(numDatabaseVectors, database.data());

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    index.search(1, query.data(), k, distances.data(), labels.data());

    int validResults = 0;
    int missingResults = 0;
    for (int i = 0; i < k; ++i) {
        if (labels[i] == -1) {
            EXPECT_EQ(distances[i], std::numeric_limits<float>::max());
            ++missingResults;
        } else {
            EXPECT_GE(labels[i], 0);
            EXPECT_LT(labels[i], numDatabaseVectors);
            EXPECT_TRUE(std::isfinite(distances[i]));
            ++validResults;
        }
    }
    EXPECT_EQ(validResults, numDatabaseVectors);
    EXPECT_EQ(missingResults, k - numDatabaseVectors);
    expectSortedDistances(distances, 1, k);

    std::fill(
            query.begin(),
            query.end(),
            std::numeric_limits<float>::quiet_NaN());
    index.search(1, query.data(), k, distances.data(), labels.data());
    for (int i = 0; i < k; ++i) {
        EXPECT_EQ(labels[i], -1);
        EXPECT_EQ(distances[i], std::numeric_limits<float>::max());
    }
}

TEST(TestGpuIndexIVFRaBitQ, RequiresL2Metric) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.device = 0;
    config.use_cuvs = true;

    const auto construct_index = [&] {
        faiss::gpu::GpuIndexIVFRaBitQ index(
                &resources, 32, 16, faiss::METRIC_INNER_PRODUCT, config);
    };

    EXPECT_THROW(construct_index(), faiss::FaissException);
}

class TestGpuIndexIVFRaBitQBits : public testing::TestWithParam<uint32_t> {};

TEST_P(TestGpuIndexIVFRaBitQBits, SupportedEndpoints) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    constexpr int dim = 32;
    constexpr int nb = 1024;
    constexpr int nq = 8;
    constexpr int k = 10;
    auto database = faiss::gpu::randVecs(nb, dim);
    auto queries = faiss::gpu::randVecs(nq, dim);

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.use_cuvs = true;
    config.bitsPerDim = GetParam();
    config.nprobe = 8;
    faiss::gpu::GpuIndexIVFRaBitQ index(
            &resources, dim, 8, faiss::METRIC_L2, config);
    index.add(nb, database.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, queries.data(), k, distances.data(), labels.data());
    expectValidResults(distances, labels, nb);
    expectSortedDistances(distances, nq, k);
}

INSTANTIATE_TEST_SUITE_P(
        BitWidths,
        TestGpuIndexIVFRaBitQBits,
        testing::Values(1u, 9u));

class TestGpuIndexIVFRaBitQInput : public testing::TestWithParam<int> {};

TEST_P(TestGpuIndexIVFRaBitQInput, HostDeviceAndStreaming) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    constexpr int dim = 32;
    constexpr int nb = 1024;
    constexpr int nq = 8;
    constexpr int k = 10;
    const bool deviceInput = GetParam() == 1;
    const bool streaming = GetParam() == 2;
    auto database = faiss::gpu::randVecs(nb, dim);
    auto queries = faiss::gpu::randVecs(nq, dim);

    faiss::gpu::DeviceScope scope(0);
    faiss::gpu::StandardGpuResources resources;
    const auto gpuResources = resources.getResources();
    const auto stream = gpuResources->getDefaultStream(0);
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.use_cuvs = true;
    config.nprobe = 8;
    config.forceStreaming = streaming;
    // Several batches are needed to exercise streaming past the first batch.
    config.streamingBatchSize = 256;
    faiss::gpu::GpuIndexIVFRaBitQ index(
            &resources, dim, 8, faiss::METRIC_L2, config);
    if (deviceInput) {
        auto deviceDatabase = faiss::gpu::toDeviceTemporary(
                gpuResources.get(), database, stream);
        index.train(nb, deviceDatabase.data());
        gpuResources->syncDefaultStreamCurrentDevice();
    } else {
        index.train(nb, database.data());
    }
    ASSERT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, nb);

    std::vector<float> hostDistances(nq * k);
    std::vector<faiss::idx_t> hostLabels(nq * k);
    index.search(
            nq, queries.data(), k, hostDistances.data(), hostLabels.data());
    expectValidResults(hostDistances, hostLabels, nb);
    expectSortedDistances(hostDistances, nq, k);

    auto deviceQueries =
            faiss::gpu::toDeviceTemporary(gpuResources.get(), queries, stream);
    const auto alloc =
            faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream);
    faiss::gpu::DeviceTensor<float, 2, true> deviceDistances(
            gpuResources.get(), alloc, {nq, k});
    faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> deviceLabels(
            gpuResources.get(), alloc, {nq, k});
    index.search(
            nq,
            deviceQueries.data(),
            k,
            deviceDistances.data(),
            deviceLabels.data());
    std::vector<float> copiedDistances(nq * k);
    std::vector<faiss::idx_t> copiedLabels(nq * k);
    faiss::gpu::fromDevice(deviceDistances, copiedDistances.data(), stream);
    faiss::gpu::fromDevice(deviceLabels, copiedLabels.data(), stream);
    EXPECT_EQ(copiedLabels, hostLabels);
    for (size_t i = 0; i < hostDistances.size(); ++i) {
        EXPECT_FLOAT_EQ(copiedDistances[i], hostDistances[i]);
    }

    // The inherited mixed-precision query path must widen Float16 queries.
    std::vector<__half> halfQueries(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        halfQueries[i] = __float2half(queries[i]);
        queries[i] = __half2float(halfQueries[i]);
    }
    index.search(
            nq, queries.data(), k, hostDistances.data(), hostLabels.data());
    faiss::Index* genericIndex = &index;
    genericIndex->search_ex(
            nq,
            halfQueries.data(),
            faiss::NumericType::Float16,
            k,
            copiedDistances.data(),
            copiedLabels.data());
    EXPECT_EQ(copiedLabels, hostLabels);
    for (size_t i = 0; i < hostDistances.size(); ++i) {
        EXPECT_FLOAT_EQ(copiedDistances[i], hostDistances[i]);
    }

    if (deviceInput) {
        cudaStream_t alternateStream;
        ASSERT_EQ(
                cudaStreamCreateWithFlags(
                        &alternateStream, cudaStreamNonBlocking),
                cudaSuccess);
        resources.setDefaultStream(0, alternateStream);
        index.search(
                nq,
                deviceQueries.data(),
                k,
                deviceDistances.data(),
                deviceLabels.data());
        // reset() must wait for the async search before freeing cuVS storage,
        // even after changing the resource stream used during construction.
        index.reset();
        EXPECT_FALSE(index.is_trained);
        EXPECT_EQ(index.ntotal, 0);
        faiss::gpu::fromDevice(
                deviceDistances, copiedDistances.data(), alternateStream);
        faiss::gpu::fromDevice(
                deviceLabels, copiedLabels.data(), alternateStream);
        expectValidResults(copiedDistances, copiedLabels, nb);
        expectSortedDistances(copiedDistances, nq, k);
        resources.revertDefaultStream(0);
        EXPECT_EQ(cudaStreamDestroy(alternateStream), cudaSuccess);
    }
}

INSTANTIATE_TEST_SUITE_P(
        InputMemory,
        TestGpuIndexIVFRaBitQInput,
        testing::Values(0, 1, 2));

TEST(TestGpuIndexIVFRaBitQ, RejectsStreamingBatchSmallerThanCluster) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    constexpr int dim = 32;
    constexpr int nb = 1024;
    auto database = faiss::gpu::randVecs(nb, dim);
    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.use_cuvs = true;
    config.forceStreaming = true;
    config.streamingBatchSize = 1;
    faiss::gpu::GpuIndexIVFRaBitQ index(
            &resources, dim, 8, faiss::METRIC_L2, config);
    // cuVS rejects batches that cannot hold a complete coarse cluster.
    EXPECT_THROW(index.train(nb, database.data()), std::exception);
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);
    index.reset();
}

TEST(TestGpuIndexIVFRaBitQ, RejectsInvalidConfiguration) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.use_cuvs = true;
    const auto construct = [&](int dim, faiss::idx_t nlist) {
        faiss::gpu::GpuIndexIVFRaBitQ index(
                &resources, dim, nlist, faiss::METRIC_L2, config);
    };
    for (int dim : {0, -1}) {
        EXPECT_THROW(construct(dim, 8), faiss::FaissException);
    }
    EXPECT_THROW(construct(32, 0), faiss::FaissException);
    EXPECT_THROW(
            construct(
                    32,
                    static_cast<faiss::idx_t>(
                            std::numeric_limits<uint32_t>::max()) +
                            1),
            faiss::FaissException);
    for (uint32_t bits : {0u, 10u}) {
        config.bitsPerDim = bits;
        EXPECT_THROW(construct(32, 8), faiss::FaissException);
    }
    config.bitsPerDim = 3;
    config.use_cuvs = false;
    EXPECT_THROW(construct(32, 8), faiss::FaissException);
    config.use_cuvs = true;
    for (uint32_t probes : {0u, 16385u}) {
        config.nprobe = probes;
        EXPECT_THROW(construct(32, 8), faiss::FaissException);
    }
    config.nprobe = 8;
    config.searchMode = static_cast<faiss::gpu::IVFRaBitQSearchMode>(-1);
    EXPECT_THROW(construct(32, 8), faiss::FaissException);
    config.searchMode = faiss::gpu::IVFRaBitQSearchMode::QUANT4;
    config.kmeansNIterations = 0;
    EXPECT_THROW(construct(32, 8), faiss::FaissException);
    config.kmeansNIterations = 20;
    config.maxTrainPointsPerCluster = 0;
    EXPECT_THROW(construct(32, 8), faiss::FaissException);
    config.maxTrainPointsPerCluster = 256;
    config.streamingBatchSize = 0;
    EXPECT_THROW(construct(32, 8), faiss::FaissException);
}

TEST(TestGpuIndexIVFRaBitQ, RejectsInvalidBuildAndSearchInputs) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    constexpr int dim = 32;
    constexpr int nb = 1024;
    constexpr int nlist = 8;
    constexpr int k = 10;
    auto database = faiss::gpu::randVecs(nb, dim);
    auto query = faiss::gpu::randVecs(1, dim);
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.use_cuvs = true;
    config.nprobe = nlist;
    faiss::gpu::GpuIndexIVFRaBitQ index(
            &resources, dim, nlist, faiss::METRIC_L2, config);

    EXPECT_THROW(
            index.search(1, query.data(), k, distances.data(), labels.data()),
            faiss::FaissException);
    for (faiss::idx_t n :
         {faiss::idx_t(-1), faiss::idx_t(0), faiss::idx_t(7)}) {
        EXPECT_THROW(index.train(n, database.data()), faiss::FaissException);
        EXPECT_FALSE(index.is_trained);
        EXPECT_EQ(index.ntotal, 0);
    }
    EXPECT_THROW(index.train(nb, nullptr), faiss::FaissException);
    EXPECT_THROW(
            index.train(
                    static_cast<faiss::idx_t>(
                            std::numeric_limits<uint32_t>::max()) +
                            1,
                    database.data()),
            faiss::FaissException);
    index.train(nb, database.data());
    EXPECT_NO_THROW(index.search(0, nullptr, k, nullptr, nullptr));
    EXPECT_THROW(
            index.search(-1, query.data(), k, distances.data(), labels.data()),
            faiss::FaissException);
    EXPECT_THROW(
            index.search(1, nullptr, k, distances.data(), labels.data()),
            faiss::FaissException);
    EXPECT_THROW(
            index.search(1, query.data(), k, nullptr, labels.data()),
            faiss::FaissException);
    EXPECT_THROW(
            index.search(1, query.data(), k, distances.data(), nullptr),
            faiss::FaissException);
    EXPECT_THROW(
            index.search(1, query.data(), 0, distances.data(), labels.data()),
            faiss::FaissException);
    EXPECT_THROW(
            index.search(
                    1,
                    query.data(),
                    static_cast<faiss::idx_t>(std::numeric_limits<int>::max()) +
                            1,
                    distances.data(),
                    labels.data()),
            faiss::FaissException);

    faiss::SearchParameters wrongParams;
    EXPECT_THROW(
            index.search(
                    1,
                    query.data(),
                    k,
                    distances.data(),
                    labels.data(),
                    &wrongParams),
            faiss::FaissException);
    faiss::gpu::SearchParametersIVFRaBitQ params;
    for (uint32_t probes : {0u, 16385u}) {
        params.nprobe = probes;
        EXPECT_THROW(
                index.search(
                        1,
                        query.data(),
                        k,
                        distances.data(),
                        labels.data(),
                        &params),
                faiss::FaissException);
    }
    params.nprobe = nlist;
    params.searchMode = static_cast<faiss::gpu::IVFRaBitQSearchMode>(-1);
    EXPECT_THROW(
            index.search(
                    1,
                    query.data(),
                    k,
                    distances.data(),
                    labels.data(),
                    &params),
            faiss::FaissException);
    params.searchMode = faiss::gpu::IVFRaBitQSearchMode::QUANT4;
    faiss::IDSelectorRange selector(0, nb / 2);
    params.sel = &selector;
    EXPECT_THROW(
            index.search(
                    1,
                    query.data(),
                    k,
                    distances.data(),
                    labels.data(),
                    &params),
            faiss::FaissException);
    params.sel = nullptr;
    // nprobe above nlist is legal and clamps to the available lists.
    params.nprobe = nlist + 1;
    index.search(1, query.data(), k, distances.data(), labels.data(), &params);
    expectValidResults(distances, labels, nb);
    expectSortedDistances(distances, 1, k);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, nb);
}

} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
