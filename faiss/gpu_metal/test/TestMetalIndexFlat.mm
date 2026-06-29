// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Minimal C++ test for MetalIndexFlat: add, search, reset; compare to CPU
 * IndexFlat.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexFlat.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>
#import <cmath>
#import <memory>
#import <vector>

namespace {

constexpr float kTolerance = 1e-5f;

void compareSearchResults(
        int nq,
        int k,
        const float* refDist,
        const faiss::idx_t* refLab,
        const float* testDist,
        const faiss::idx_t* testLab) {
    for (int i = 0; i < nq * k; ++i) {
        EXPECT_NEAR(
                refDist[i],
                testDist[i],
                kTolerance * (std::fabs(refDist[i]) + 1.0f))
                << "i=" << i;
        EXPECT_EQ(refLab[i], testLab[i]) << "i=" << i;
    }
}

} // namespace

class TestMetalIndexFlat : public ::testing::Test {
   protected:
    void SetUp() override {
        resources_ = std::make_shared<faiss::gpu_metal::MetalResources>();
        if (!resources_->isAvailable()) {
            GTEST_SKIP() << "Metal not available (no device or queue)";
        }
    }
    std::shared_ptr<faiss::gpu_metal::MetalResources> resources_;
};

TEST_F(TestMetalIndexFlat, L2_AddAndSearch) {
    const int dim = 4;
    const int numVecs = 50;
    const int numQuery = 5;
    const int k = 3;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(
            numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(
            numQuery,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(TestMetalIndexFlat, IP_AddAndSearch) {
    const int dim = 4;
    const int numVecs = 50;
    const int numQuery = 5;
    const int k = 3;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatIP cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_INNER_PRODUCT, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(
            numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(
            numQuery,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(TestMetalIndexFlat, AddWithIdsThrows) {
    const int dim = 4;
    const int numVecs = 10;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    std::vector<faiss::idx_t> ids(numVecs);
    for (int i = 0; i < numVecs; ++i) {
        ids[i] = 1000 + (faiss::idx_t)i;
    }

    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    EXPECT_THROW(
            metalIndex.add_with_ids(numVecs, vecs.data(), ids.data()),
            faiss::FaissException);
}

TEST_F(TestMetalIndexFlat, Reset) {
    const int dim = 4;
    const int numVecs = 10;
    const int numQuery = 2;
    const int k = 1;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 99);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 100);

    faiss::gpu_metal::MetalIndexFlat index(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    index.add(numVecs, vecs.data());
    EXPECT_EQ(index.ntotal, numVecs);

    index.reset();
    EXPECT_EQ(index.ntotal, 0);

    std::vector<float> dists((size_t)numQuery * k);
    std::vector<faiss::idx_t> labels((size_t)numQuery * k, -2);
    index.search(numQuery, queries.data(), k, dists.data(), labels.data());
    for (int i = 0; i < numQuery * k; ++i) {
        EXPECT_EQ(labels[i], -1) << "after reset, labels should be -1";
    }
}

TEST_F(TestMetalIndexFlat, EmptySearch) {
    const int dim = 4;
    const int numQuery = 2;
    const int k = 1;

    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 101);

    faiss::gpu_metal::MetalIndexFlat index(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    std::vector<float> dists((size_t)numQuery * k);
    std::vector<faiss::idx_t> labels((size_t)numQuery * k, -2);
    index.search(numQuery, queries.data(), k, dists.data(), labels.data());
    for (int i = 0; i < numQuery * k; ++i) {
        EXPECT_EQ(labels[i], -1);
    }
}

TEST_F(TestMetalIndexFlat, GetNumGpus) {
    int n = faiss::gpu_metal::get_num_gpus();
    EXPECT_GE(n, 0);
    EXPECT_LE(n, 1);
    if (resources_->isAvailable()) {
        EXPECT_EQ(n, 1);
    }
}

TEST_F(TestMetalIndexFlat, IndexCpuToMetalGpu) {
    const int dim = 4;
    const int numVecs = 30;
    const int numQuery = 3;
    const int k = 2;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 200);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 201);

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(numVecs, vecs.data());

    faiss::gpu_metal::StandardMetalResources res;
    faiss::Index* metalIndex =
            faiss::gpu_metal::index_cpu_to_metal_gpu(&res, 0, &cpuIndex);
    ASSERT_NE(metalIndex, nullptr);
    EXPECT_EQ(metalIndex->ntotal, numVecs);

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);
    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex->search(
            numQuery, queries.data(), k, testDist.data(), testLab.data());
    compareSearchResults(
            numQuery,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());

    delete metalIndex;
}

TEST_F(TestMetalIndexFlat, IndexMetalGpuToCpu) {
    const int dim = 4;
    const int numVecs = 20;
    const int numQuery = 2;
    const int k = 2;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 300);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 301);

    faiss::IndexFlatL2 cpuOrig(dim);
    cpuOrig.add(numVecs, vecs.data());

    faiss::gpu_metal::StandardMetalResources res;
    faiss::Index* metalIndex =
            faiss::gpu_metal::index_cpu_to_metal_gpu(&res, 0, &cpuOrig);
    ASSERT_NE(metalIndex, nullptr);
    faiss::Index* cpuBack =
            faiss::gpu_metal::index_metal_gpu_to_cpu(metalIndex);
    ASSERT_NE(cpuBack, nullptr);
    EXPECT_EQ(cpuBack->ntotal, numVecs);

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);
    cpuOrig.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    cpuBack->search(
            numQuery, queries.data(), k, testDist.data(), testLab.data());
    compareSearchResults(
            numQuery,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());

    delete cpuBack;
    delete metalIndex;
}
