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
    index.train(opt.numAdd, database.data());
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

    index.reset();
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);

    faiss::Index* genericIndex = &index;
    EXPECT_THROW(
            genericIndex->add_ex(
                    1, database.data(), faiss::NumericType::Float16),
            faiss::FaissException);
    genericIndex->add_ex(
            opt.numAdd, database.data(), faiss::NumericType::Float32);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, opt.numAdd);
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
    gpu_index.search(
            opt.numQuery,
            queries.data(),
            opt.k,
            actualDistances.data(),
            actualLabels.data());

    expectValidResults(actualDistances, actualLabels, opt.numAdd);
    expectSortedDistances(actualDistances, opt.numQuery, opt.k);
    EXPECT_GE(recallAtK(exactLabels, actualLabels, opt.numQuery, opt.k), 0.5)
            << opt.toString();
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

} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
