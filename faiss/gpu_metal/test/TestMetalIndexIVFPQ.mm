// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Tests for MetalIndexIVFPQ: train, add, search, reset, cloner round-trip.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexIVFPQ.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>
#import <cmath>
#import <memory>
#import <set>
#import <vector>

namespace {

void compareSearchLabels(
        int nq,
        int k,
        const faiss::idx_t* refLab,
        const faiss::idx_t* testLab) {
    int matches = 0;
    for (int q = 0; q < nq; ++q) {
        std::set<faiss::idx_t> refSet(refLab + q * k, refLab + (q + 1) * k);
        for (int j = 0; j < k; ++j) {
            if (refSet.count(testLab[q * k + j]))
                ++matches;
        }
    }
    EXPECT_GT(matches, nq * k / 2);
}

} // namespace

class TestMetalIndexIVFPQ : public ::testing::Test {
protected:
    void SetUp() override {
        resources_ = std::make_shared<faiss::gpu_metal::MetalResources>();
        if (!resources_->isAvailable()) {
            GTEST_SKIP() << "Metal not available";
        }
    }
    std::shared_ptr<faiss::gpu_metal::MetalResources> resources_;
};

TEST_F(TestMetalIndexIVFPQ, L2_TrainAddSearch) {
    const int dim = 64;
    const int nb = 2000;
    const int nq = 10;
    const int nlist = 8;
    const int nprobe = 4;
    const int k = 5;
    const int M = 8;
    const int nbits = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFPQ cpuIndex(&quantizer, dim, nlist, M, nbits);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFPQ metalIndex(resources_, &cpuIndex);
    ASSERT_EQ(metalIndex.ntotal, nb);

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> gpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> gpuLab((size_t)nq * k);
    metalIndex.search(nq, queries.data(), k, gpuDist.data(), gpuLab.data());

    compareSearchLabels(nq, k, refLab.data(), gpuLab.data());
}

TEST_F(TestMetalIndexIVFPQ, IP_TrainAddSearch) {
    const int dim = 64;
    const int nb = 2000;
    const int nq = 10;
    const int nlist = 8;
    const int nprobe = 4;
    const int k = 5;
    const int M = 8;
    const int nbits = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatIP quantizer(dim);
    faiss::IndexIVFPQ cpuIndex(&quantizer, dim, nlist, M, nbits,
                               faiss::METRIC_INNER_PRODUCT);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFPQ metalIndex(resources_, &cpuIndex);
    ASSERT_EQ(metalIndex.ntotal, nb);

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> gpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> gpuLab((size_t)nq * k);
    metalIndex.search(nq, queries.data(), k, gpuDist.data(), gpuLab.data());

    compareSearchLabels(nq, k, refLab.data(), gpuLab.data());
}

TEST_F(TestMetalIndexIVFPQ, Reset) {
    const int dim = 64;
    const int nb = 500;
    const int nlist = 4;
    const int M = 8;
    const int nbits = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);

    faiss::gpu_metal::MetalIndexIVFPQ metalIndex(
            resources_, dim, nlist, M, nbits, faiss::METRIC_L2);
    metalIndex.train(nb, vecs.data());
    metalIndex.add(nb, vecs.data());
    ASSERT_EQ(metalIndex.ntotal, nb);

    metalIndex.reset();
    ASSERT_EQ(metalIndex.ntotal, 0);
}

TEST_F(TestMetalIndexIVFPQ, CpuToGpuRoundTrip) {
    const int dim = 64;
    const int nb = 1000;
    const int nlist = 8;
    const int M = 8;
    const int nbits = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 99);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFPQ cpuIndex(&quantizer, dim, nlist, M, nbits);
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    auto* metalRaw = dynamic_cast<faiss::gpu_metal::MetalIndexIVFPQ*>(
            faiss::gpu_metal::index_cpu_to_metal_gpu(
                    &stdRes, 0, &cpuIndex));
    ASSERT_NE(metalRaw, nullptr);
    std::unique_ptr<faiss::gpu_metal::MetalIndexIVFPQ> metalIndex(metalRaw);
    ASSERT_EQ(metalIndex->ntotal, nb);

    auto* cpuBack = faiss::gpu_metal::index_metal_gpu_to_cpu(metalIndex.get());
    ASSERT_NE(cpuBack, nullptr);
    std::unique_ptr<faiss::Index> cpuBackPtr(cpuBack);
    ASSERT_EQ(cpuBack->ntotal, nb);
}
