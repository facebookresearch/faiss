/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryFlat.h>
#include <faiss/gpu/GpuIndexBinaryFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

void compareBinaryDist(
        const std::vector<int>& cpuDist,
        const std::vector<faiss::idx_t>& cpuLabels,
        const std::vector<int>& gpuDist,
        const std::vector<faiss::idx_t>& gpuLabels,
        int numQuery,
        int k) {
    for (int i = 0; i < numQuery; ++i) {
        // The index order can be permuted within a group that has the same
        // distance, since this is based on the order in which the algorithm
        // encounters the values. The last set of equivalent distances seen in
        // the min-k might be truncated, so we can't check that set, but all
        // others we can check.
        std::set<faiss::idx_t> cpuLabelSet;
        std::set<faiss::idx_t> gpuLabelSet;

        int curDist = -1;

        for (int j = 0; j < k; ++j) {
            int idx = i * k + j;

            if (curDist == -1) {
                curDist = cpuDist[idx];
            }

            if (curDist != cpuDist[idx]) {
                // Distances must be monotonically increasing
                EXPECT_LT(curDist, cpuDist[idx]);

                // This is a new set of distances
                EXPECT_EQ(cpuLabelSet, gpuLabelSet);
                curDist = cpuDist[idx];
                cpuLabelSet.clear();
                gpuLabelSet.clear();
            }

            cpuLabelSet.insert(cpuLabels[idx]);
            gpuLabelSet.insert(gpuLabels[idx]);

            // Because the distances are reproducible, they must be exactly the
            // same
            EXPECT_EQ(cpuDist[idx], gpuDist[idx]);
        }
    }
}

template <int DimMultiple>
void testGpuIndexBinaryFlat(int kOverride = -1) {
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexBinaryFlatConfig config;
    config.device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    // multiples of 8 and multiples of 32 use different implementations
    int dims = faiss::gpu::randVal(1, 20) * DimMultiple;
    faiss::gpu::GpuIndexBinaryFlat gpuIndex(&res, dims, config);

    faiss::IndexBinaryFlat cpuIndex(dims);

    int k = kOverride > 0
            ? kOverride
            : faiss::gpu::randVal(1, faiss::gpu::getMaxKSelection(false));
    int numVecs = faiss::gpu::randVal(k + 1, 20000);
    int numQuery = faiss::gpu::randVal(1, 1000);

    auto data = faiss::gpu::randBinaryVecs(numVecs, dims);
    gpuIndex.add(numVecs, data.data());
    cpuIndex.add(numVecs, data.data());

    auto query = faiss::gpu::randBinaryVecs(numQuery, dims);

    std::vector<int> cpuDist(numQuery * k);
    std::vector<faiss::idx_t> cpuLabels(numQuery * k);

    cpuIndex.search(
            numQuery, query.data(), k, cpuDist.data(), cpuLabels.data());

    std::vector<int> gpuDist(numQuery * k);
    std::vector<faiss::idx_t> gpuLabels(numQuery * k);

    gpuIndex.search(
            numQuery, query.data(), k, gpuDist.data(), gpuLabels.data());

    compareBinaryDist(cpuDist, cpuLabels, gpuDist, gpuLabels, numQuery, k);
}

TEST(TestGpuIndexBinaryFlat, Test8) {
    for (int tries = 0; tries < 4; ++tries) {
        testGpuIndexBinaryFlat<8>();
    }
}

TEST(TestGpuIndexBinaryFlat, Test32) {
    for (int tries = 0; tries < 4; ++tries) {
        testGpuIndexBinaryFlat<32>();
    }
}

TEST(TestGpuIndexBinaryFlat, LargeIndex) {
    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    // Skip this device if we do not have sufficient memory
    constexpr size_t kMem = size_t(8) * 1024 * 1024 * 1024;

    if (faiss::gpu::getFreeMemory(device) < kMem) {
        std::cerr << "TestGpuIndexFlat.LargeIndex: skipping due "
                     "to insufficient device memory\n";
        return;
    }

    std::cerr << "Running LargeIndex test\n";

    faiss::gpu::GpuIndexBinaryFlatConfig config;
    config.device = device;

    int dims = 1250 * 8;
    faiss::gpu::GpuIndexBinaryFlat gpuIndex(&res, dims, config);

    faiss::IndexBinaryFlat cpuIndex(dims);

    int k = 10;
    int nb = 4000000;
    int nq = 10;

    auto xb = faiss::gpu::randBinaryVecs(nb, dims);
    auto xq = faiss::gpu::randBinaryVecs(nq, dims);
    gpuIndex.add(nb, xb.data());
    cpuIndex.add(nb, xb.data());

    std::vector<int> cpuDist(nq * k);
    std::vector<faiss::idx_t> cpuLabels(nq * k);

    cpuIndex.search(nq, xq.data(), k, cpuDist.data(), cpuLabels.data());

    std::vector<int> gpuDist(nq * k);
    std::vector<faiss::idx_t> gpuLabels(nq * k);

    gpuIndex.search(nq, xq.data(), k, gpuDist.data(), gpuLabels.data());

    compareBinaryDist(cpuDist, cpuLabels, gpuDist, gpuLabels, nq, k);
}

TEST(TestGpuIndexBinaryFlat, Reconstruct) {
    int n = 1000;
    std::vector<uint8_t> xb(8 * n);
    faiss::byte_rand(xb.data(), xb.size(), 123);
    std::unique_ptr<faiss::IndexBinaryFlat> index(
            new faiss::IndexBinaryFlat(64));
    index->add(n, xb.data());

    std::vector<uint8_t> xb3(8 * n);
    index->reconstruct_n(0, index->ntotal, xb3.data());
    EXPECT_EQ(xb, xb3);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    std::unique_ptr<faiss::gpu::GpuIndexBinaryFlat> index2(
            new faiss::gpu::GpuIndexBinaryFlat(&res, index.get()));

    std::vector<uint8_t> xb2(8 * n);

    index2->reconstruct_n(0, index->ntotal, xb2.data());
    EXPECT_EQ(xb2, xb3);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
