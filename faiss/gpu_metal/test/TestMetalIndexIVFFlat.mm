// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Tests for MetalIndexIVFFlat: train, add, search, reset, cloner round-trip.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexIVFFlat.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>
#import <cmath>
#import <memory>
#import <vector>

namespace {

constexpr float kTolerance = 1e-4f;

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

class TestMetalIndexIVFFlat : public ::testing::Test {
   protected:
    void SetUp() override {
        resources_ = std::make_shared<faiss::gpu_metal::MetalResources>();
        if (!resources_->isAvailable()) {
            GTEST_SKIP() << "Metal not available";
        }
    }
    std::shared_ptr<faiss::gpu_metal::MetalResources> resources_;
};

TEST_F(TestMetalIndexIVFFlat, L2_TrainAddSearch) {
    const int dim = 64;
    const int nb = 1000;
    const int nq = 10;
    const int nlist = 8;
    const int nprobe = 4;
    const int k = 5;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFFlat cpuIndex(&quantizer, dim, nlist);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIndex(
            resources_, &cpuIndex);
    ASSERT_EQ(metalIndex.ntotal, nb);

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(
            nq, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(
            nq, k, refDist.data(), refLab.data(),
            testDist.data(), testLab.data());
}

TEST_F(TestMetalIndexIVFFlat, IP_TrainAddSearch) {
    const int dim = 64;
    const int nb = 1000;
    const int nq = 10;
    const int nlist = 8;
    const int nprobe = 4;
    const int k = 5;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatIP quantizer(dim);
    faiss::IndexIVFFlat cpuIndex(
            &quantizer, dim, nlist, faiss::METRIC_INNER_PRODUCT);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIndex(
            resources_, &cpuIndex);

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(
            nq, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(
            nq, k, refDist.data(), refLab.data(),
            testDist.data(), testLab.data());
}

TEST_F(TestMetalIndexIVFFlat, Reset) {
    const int dim = 32;
    const int nb = 500;
    const int nlist = 4;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFFlat cpuIndex(&quantizer, dim, nlist);
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIndex(
            resources_, &cpuIndex);
    EXPECT_EQ(metalIndex.ntotal, nb);

    metalIndex.reset();
    EXPECT_EQ(metalIndex.ntotal, 0);
}

TEST_F(TestMetalIndexIVFFlat, CpuToGpuRoundTrip) {
    const int dim = 64;
    const int nb = 800;
    const int nq = 5;
    const int nlist = 8;
    const int nprobe = 4;
    const int k = 5;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 300);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 301);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFFlat cpuOrig(&quantizer, dim, nlist);
    cpuOrig.nprobe = nprobe;
    cpuOrig.train(nb, vecs.data());
    cpuOrig.add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources res;
    std::unique_ptr<faiss::Index> metalIndex(
            faiss::gpu_metal::index_cpu_to_metal_gpu(
                    &res, 0, &cpuOrig));
    ASSERT_NE(metalIndex, nullptr);
    EXPECT_EQ(metalIndex->ntotal, nb);

    std::unique_ptr<faiss::Index> cpuBack(
            faiss::gpu_metal::index_metal_gpu_to_cpu(
                    metalIndex.get()));
    ASSERT_NE(cpuBack, nullptr);
    EXPECT_EQ(cpuBack->ntotal, nb);

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    cpuOrig.search(nq, queries.data(), k, refDist.data(), refLab.data());
    cpuBack->search(
            nq, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(
            nq, k, refDist.data(), refLab.data(),
            testDist.data(), testLab.data());
}
