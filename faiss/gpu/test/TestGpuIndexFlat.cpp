/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <sstream>
#include <unordered_map>
#include <vector>

// FIXME: figure out a better way to test fp16
constexpr float kF16MaxRelErr = 0.07f;
constexpr float kF32MaxRelErr = 6e-3f;

struct TestFlatOptions {
    TestFlatOptions()
            : metric(faiss::MetricType::METRIC_L2),
              metricArg(0),
              useFloat16(false),
              numVecsOverride(-1),
              numQueriesOverride(-1),
              kOverride(-1),
              dimOverride(-1),
              use_cuvs(false) {}

    faiss::MetricType metric;
    float metricArg;

    bool useFloat16;
    int numVecsOverride;
    int numQueriesOverride;
    int kOverride;
    int dimOverride;
    bool use_cuvs;
};

void testFlat(const TestFlatOptions& opt) {
    int numVecs = opt.numVecsOverride > 0 ? opt.numVecsOverride
                                          : faiss::gpu::randVal(1000, 5000);
    int dim = opt.dimOverride > 0 ? opt.dimOverride
                                  : faiss::gpu::randVal(50, 800);
    int numQuery = opt.numQueriesOverride > 0 ? opt.numQueriesOverride
                                              : faiss::gpu::randVal(1, 512);

    // Due to loss of precision in a float16 accumulator, for large k,
    // the number of differences is pretty huge. Restrict ourselves to a
    // fairly small `k` for float16
    int k = opt.useFloat16
            ? std::min(faiss::gpu::randVal(1, 50), numVecs)
            : std::min(
                      faiss::gpu::randVal(
                              1, faiss::gpu::getMaxKSelection(opt.use_cuvs)),
                      numVecs);
    if (opt.kOverride > 0) {
        k = opt.kOverride;
    }

    faiss::IndexFlat cpuIndex(dim, opt.metric);
    cpuIndex.metric_arg = opt.metricArg;

    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    config.useFloat16 = opt.useFloat16;
    config.use_cuvs = opt.use_cuvs;

    faiss::gpu::GpuIndexFlat gpuIndex(&res, dim, opt.metric, config);
    gpuIndex.metric_arg = opt.metricArg;

    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
    cpuIndex.add(numVecs, vecs.data());
    gpuIndex.add(numVecs, vecs.data());

    std::stringstream str;
    str << "metric " << opt.metric << " marg " << opt.metricArg << " numVecs "
        << numVecs << " dim " << dim << " useFloat16 " << opt.useFloat16
        << " numQuery " << numQuery << " k " << k;

    // To some extent, we depend upon the relative error for the test
    // for float16
    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            numQuery,
            dim,
            k,
            str.str(),
            opt.useFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
            // FIXME: the fp16 bounds are
            // useless when math (the accumulator) is
            // in fp16. Figure out another way to test
            opt.useFloat16 ? 0.99f : 0.1f,
            opt.useFloat16 ? 0.65f : 0.015f);
}

TEST(TestGpuIndexFlat, IP_Float32) {
    for (int tries = 0; tries < 3; ++tries) {
        TestFlatOptions opt;
        opt.metric = faiss::MetricType::METRIC_INNER_PRODUCT;
        opt.useFloat16 = false;

        testFlat(opt);

#if defined USE_NVIDIA_CUVS
        opt.use_cuvs = true;
        testFlat(opt);
#endif
    }
}

TEST(TestGpuIndexFlat, L1_Float32) {
    TestFlatOptions opt;
    opt.metric = faiss::MetricType::METRIC_L1;
    opt.useFloat16 = false;

    testFlat(opt);

#if defined USE_NVIDIA_CUVS
    opt.use_cuvs = true;
    testFlat(opt);
#endif
}

TEST(TestGpuIndexFlat, Lp_Float32) {
    TestFlatOptions opt;
    opt.metric = faiss::MetricType::METRIC_Lp;
    opt.metricArg = 5;
    opt.useFloat16 = false;

    testFlat(opt);
#if defined USE_NVIDIA_CUVS
    opt.use_cuvs = true;
    testFlat(opt);
#endif
}

TEST(TestGpuIndexFlat, L2_Float32) {
    for (int tries = 0; tries < 3; ++tries) {
        TestFlatOptions opt;
        opt.metric = faiss::MetricType::METRIC_L2;

        opt.useFloat16 = false;

        testFlat(opt);
#if defined USE_NVIDIA_CUVS
        opt.use_cuvs = true;
        testFlat(opt);
#endif
    }
}

// At least one test for the k > 1024 select
TEST(TestGpuIndexFlat, L2_k_2048) {
    if (faiss::gpu::getMaxKSelection(false) >= 2048) {
        TestFlatOptions opt;
        opt.metric = faiss::MetricType::METRIC_L2;
        opt.useFloat16 = false;
        opt.kOverride = 2048;
        opt.dimOverride = 128;
        opt.numVecsOverride = 10000;

        testFlat(opt);
#if defined USE_NVIDIA_CUVS
        opt.use_cuvs = true;
        testFlat(opt);
#endif
    }
}

// test specialized k == 1 codepath
TEST(TestGpuIndexFlat, L2_Float32_K1) {
    for (int tries = 0; tries < 3; ++tries) {
        TestFlatOptions opt;
        opt.metric = faiss::MetricType::METRIC_L2;
        opt.useFloat16 = false;
        opt.kOverride = 1;

        testFlat(opt);
#if defined USE_NVIDIA_CUVS
        opt.use_cuvs = true;
        testFlat(opt);
#endif
    }
}

TEST(TestGpuIndexFlat, IP_Float16) {
    for (int tries = 0; tries < 3; ++tries) {
        TestFlatOptions opt;
        opt.metric = faiss::MetricType::METRIC_INNER_PRODUCT;
        opt.useFloat16 = true;

        testFlat(opt);
#if defined USE_NVIDIA_CUVS
        opt.use_cuvs = true;
        testFlat(opt);
#endif
    }
}

TEST(TestGpuIndexFlat, L2_Float16) {
    for (int tries = 0; tries < 3; ++tries) {
        TestFlatOptions opt;
        opt.metric = faiss::MetricType::METRIC_L2;
        opt.useFloat16 = true;

        testFlat(opt);
#if defined USE_NVIDIA_CUVS
        opt.use_cuvs = true;
        testFlat(opt);
#endif
    }
}

// test specialized k == 1 codepath
TEST(TestGpuIndexFlat, L2_Float16_K1) {
    for (int tries = 0; tries < 3; ++tries) {
        TestFlatOptions opt;
        opt.metric = faiss::MetricType::METRIC_L2;
        opt.useFloat16 = true;
        opt.kOverride = 1;

        testFlat(opt);
#if defined USE_NVIDIA_CUVS
        opt.use_cuvs = true;
        testFlat(opt);
#endif
    }
}

// test tiling along a huge vector set
TEST(TestGpuIndexFlat, L2_Tiling) {
    for (int tries = 0; tries < 2; ++tries) {
        TestFlatOptions opt;
        opt.metric = faiss::MetricType::METRIC_L2;
        opt.useFloat16 = false;
        opt.numVecsOverride = 1000000;

        // keep the rest of the problem reasonably small
        opt.numQueriesOverride = 4;
        opt.dimOverride = 64;
        opt.kOverride = 64;

        testFlat(opt);
#if defined USE_NVIDIA_CUVS
        opt.use_cuvs = true;
        testFlat(opt);
#endif
    }
}

TEST(TestGpuIndexFlat, QueryEmpty) {
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexFlatConfig config;
    config.device = 0;
    config.useFloat16 = false;
    config.use_cuvs = false;
    int dim = 128;
    faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, dim, config);

    // Querying an empty index should not blow up, and just return
    // (FLT_MAX, -1)
    int numQuery = 10;
    int k = 50;
    std::vector<float> queries(numQuery * dim, 1.0f);

    std::vector<float> dist(numQuery * k, 0);
    std::vector<faiss::idx_t> ind(numQuery * k);

    gpuIndex.search(numQuery, queries.data(), k, dist.data(), ind.data());

    for (auto d : dist) {
        EXPECT_EQ(d, std::numeric_limits<float>::max());
    }

    for (auto i : ind) {
        EXPECT_EQ(i, -1);
    }
}

void testCopyFrom(bool use_cuvs) {
    int numVecs = faiss::gpu::randVal(100, 200);
    int dim = faiss::gpu::randVal(1, 1000);

    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(numVecs, vecs.data());

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    for (bool useFloat16 : {false, true}) {
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = device;
        config.useFloat16 = useFloat16;
        config.use_cuvs = use_cuvs;

        // Fill with garbage values
        faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, 2000, config);
        gpuIndex.copyFrom(&cpuIndex);

        EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
        EXPECT_EQ(gpuIndex.ntotal, numVecs);

        EXPECT_EQ(cpuIndex.d, gpuIndex.d);
        EXPECT_EQ(cpuIndex.d, dim);

        std::vector<float> gpuVals(numVecs * dim);
        gpuIndex.reconstruct_n(0, gpuIndex.ntotal, gpuVals.data());

        std::vector<float> cpuVals(numVecs * dim);
        cpuIndex.reconstruct_n(0, gpuIndex.ntotal, cpuVals.data());

        // The CPU is the source of (float32) truth here, while the GPU index
        // may be in float16 mode and thus was subject to rounding
        if (useFloat16) {
            EXPECT_EQ(gpuVals, faiss::gpu::roundToHalf(cpuVals));
        } else {
            // Should be exactly the same
            EXPECT_EQ(gpuVals, cpuVals);
        }
    }
}

TEST(TestGpuIndexFlat, CopyFrom) {
    testCopyFrom(false);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuIndexFlat, CopyFrom) {
    testCopyFrom(true);
}
#endif

void testCopyTo(bool use_cuvs) {
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    int numVecs = faiss::gpu::randVal(100, 200);
    int dim = faiss::gpu::randVal(1, 1000);

    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);

    for (bool useFloat16 : {false, true}) {
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = device;
        config.useFloat16 = useFloat16;
        config.use_cuvs = use_cuvs;

        faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, dim, config);
        gpuIndex.add(numVecs, vecs.data());

        // Fill with garbage values
        faiss::IndexFlatL2 cpuIndex(2000);
        gpuIndex.copyTo(&cpuIndex);

        EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
        EXPECT_EQ(gpuIndex.ntotal, numVecs);

        EXPECT_EQ(cpuIndex.d, gpuIndex.d);
        EXPECT_EQ(cpuIndex.d, dim);

        std::vector<float> gpuVals(numVecs * dim);
        gpuIndex.reconstruct_n(0, gpuIndex.ntotal, gpuVals.data());

        std::vector<float> cpuVals(numVecs * dim);
        cpuIndex.reconstruct_n(0, gpuIndex.ntotal, cpuVals.data());

        // The GPU is the source of truth here, so the float32 exact comparison
        // even if the index uses float16 is ok
        EXPECT_EQ(gpuVals, cpuVals);
    }
}

TEST(TestGpuIndexFlat, CopyTo) {
    testCopyTo(false);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuIndexFlat, CopyTo) {
    testCopyTo(true);
}
#endif

void testUnifiedMemory(bool use_cuvs) {
    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    if (!faiss::gpu::getFullUnifiedMemSupport(device)) {
        return;
    }

    int dim = 256;

    // FIXME: GpuIndexFlat doesn't support > 2^31 (vecs * dims) due to
    // kernel indexing, so we can't test unified memory for memory
    // oversubscription.
    size_t numVecs = 50000;
    int numQuery = 10;
    int k = 10;

    faiss::IndexFlatL2 cpuIndexL2(dim);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    config.memorySpace = faiss::gpu::MemorySpace::Unified;
    config.use_cuvs = use_cuvs;

    faiss::gpu::GpuIndexFlatL2 gpuIndexL2(&res, dim, config);

    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
    cpuIndexL2.add(numVecs, vecs.data());
    gpuIndexL2.add(numVecs, vecs.data());

    // To some extent, we depend upon the relative error for the test
    // for float16
    faiss::gpu::compareIndices(
            cpuIndexL2,
            gpuIndexL2,
            numQuery,
            dim,
            k,
            "Unified Memory",
            kF32MaxRelErr,
            0.1f,
            0.015f);
}

TEST(TestGpuIndexFlat, UnifiedMemory) {
    testUnifiedMemory(false);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuIndexFlat, UnifiedMemory) {
    testUnifiedMemory(true);
}
#endif

void testLargeIndex(bool use_cuvs) {
    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    // Skip this device if we do not have sufficient memory
    constexpr size_t kMem = size_t(8) * 1024 * 1024 * 1024;

    if (faiss::gpu::getFreeMemory(device) < kMem) {
        std::cout << "TestGpuIndexFlat.LargeIndex: skipping due "
                     "to insufficient device memory\n";
        return;
    }

    std::cout << "Running LargeIndex test\n";

    size_t dim = 256; // each vec is sizeof(float) * 256 = 1 KiB in size
    size_t nb = 5000000;
    size_t nq = 10;

    auto xb = faiss::gpu::randVecs(nb, dim);

    int k = 10;

    faiss::IndexFlatL2 cpuIndexL2(dim);

    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    config.use_cuvs = use_cuvs;
    faiss::gpu::GpuIndexFlatL2 gpuIndexL2(&res, dim, config);

    cpuIndexL2.add(nb, xb.data());
    gpuIndexL2.add(nb, xb.data());

    // To some extent, we depend upon the relative error for the test
    // for float16
    faiss::gpu::compareIndices(
            cpuIndexL2,
            gpuIndexL2,
            nq,
            dim,
            k,
            "LargeIndex",
            kF32MaxRelErr,
            0.1f,
            0.015f);
}

TEST(TestGpuIndexFlat, LargeIndex) {
    testLargeIndex(false);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuIndexFlat, LargeIndex) {
    testLargeIndex(true);
}
#endif

void testResidual(bool use_cuvs) {
    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    config.use_cuvs = use_cuvs;

    int dim = 32;
    faiss::IndexFlat cpuIndex(dim, faiss::MetricType::METRIC_L2);
    faiss::gpu::GpuIndexFlat gpuIndex(
            &res, dim, faiss::MetricType::METRIC_L2, config);

    int numVecs = 100;
    auto vecs = faiss::gpu::randVecs(numVecs, dim);
    cpuIndex.add(numVecs, vecs.data());
    gpuIndex.add(numVecs, vecs.data());

    auto indexVecs = std::vector<faiss::idx_t>{0, 2, 4, 6, 8};
    auto queryVecs = faiss::gpu::randVecs(indexVecs.size(), dim);

    auto residualsCpu = std::vector<float>(indexVecs.size() * dim);
    auto residualsGpu = std::vector<float>(indexVecs.size() * dim);

    cpuIndex.compute_residual_n(
            indexVecs.size(),
            queryVecs.data(),
            residualsCpu.data(),
            indexVecs.data());
    gpuIndex.compute_residual_n(
            indexVecs.size(),
            queryVecs.data(),
            residualsGpu.data(),
            indexVecs.data());

    // Should be exactly the same, as this is just a single float32 subtraction
    EXPECT_EQ(residualsCpu, residualsGpu);
}

TEST(TestGpuIndexFlat, Residual) {
    testResidual(false);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuIndexFlat, Residual) {
    testResidual(true);
}
#endif

void testReconstruct(bool use_cuvs) {
    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    int dim = 32;
    int numVecs = 100;
    auto vecs = faiss::gpu::randVecs(numVecs, dim);
    auto vecs16 = faiss::gpu::roundToHalf(vecs);

    for (bool useFloat16 : {false, true}) {
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = device;
        config.useFloat16 = useFloat16;
        config.use_cuvs = use_cuvs;

        faiss::gpu::GpuIndexFlat gpuIndex(
                &res, dim, faiss::MetricType::METRIC_L2, config);

        gpuIndex.add(numVecs, vecs.data());

        // Test reconstruct
        {
            auto reconstructVecs = std::vector<float>(dim);
            gpuIndex.reconstruct(15, reconstructVecs.data());

            auto& ref = useFloat16 ? vecs16 : vecs;

            for (int i = 0; i < dim; ++i) {
                EXPECT_EQ(reconstructVecs[i], ref[15 * dim + i]);
            }
        }

        // Test reconstruct_n
        if (false) {
            auto reconstructVecs = std::vector<float>((numVecs - 1) * dim);

            int startVec = 5;
            int endVec = numVecs - 1;
            int numReconstructVec = endVec - startVec + 1;

            gpuIndex.reconstruct_n(
                    startVec, numReconstructVec, reconstructVecs.data());

            auto& ref = useFloat16 ? vecs16 : vecs;

            for (int i = 0; i < numReconstructVec; ++i) {
                for (int j = 0; j < dim; ++j) {
                    EXPECT_EQ(
                            reconstructVecs[i * dim + j],
                            ref[(i + startVec) * dim + j]);
                }
            }
        }

        // Test reconstruct_batch
        if (false) {
            auto reconstructKeys = std::vector<faiss::idx_t>{1, 3, 5};
            auto reconstructVecs =
                    std::vector<float>(reconstructKeys.size() * dim);

            gpuIndex.reconstruct_batch(
                    reconstructKeys.size(),
                    reconstructKeys.data(),
                    reconstructVecs.data());

            auto& ref = useFloat16 ? vecs16 : vecs;

            for (int i = 0; i < reconstructKeys.size(); ++i) {
                for (int j = 0; j < dim; ++j) {
                    EXPECT_EQ(
                            reconstructVecs[i * dim + j],
                            ref[reconstructKeys[i] * dim + j]);
                }
            }
        }
    }
}

TEST(TestGpuIndexFlat, Reconstruct) {
    testReconstruct(false);
}
#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuIndexFlat, Reconstruct) {
    testReconstruct(true);
}
#endif

void testSearchAndReconstruct(bool use_cuvs) {
    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    size_t dim = 32;
    size_t nb = 5000;
    size_t nq = 10;
    int k = 10;

    auto xb = faiss::gpu::randVecs(nb, dim);
    auto xq = faiss::gpu::randVecs(nq, dim);

    faiss::IndexFlatL2 cpuIndex(dim);

    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    config.use_cuvs = use_cuvs;
    faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, dim, config);

    cpuIndex.add(nb, xb.data());
    gpuIndex.add(nb, xb.data());

    std::vector<float> refDistance(nq * k, 0);
    std::vector<faiss::idx_t> refIndices(nq * k, -1);
    std::vector<float> refReconstruct(nq * k * dim, 0);
    cpuIndex.search_and_reconstruct(
            nq,
            xq.data(),
            k,
            refDistance.data(),
            refIndices.data(),
            refReconstruct.data());

    std::vector<float> testDistance(nq * k, 0);
    std::vector<faiss::idx_t> testIndices(nq * k, -1);
    std::vector<float> testReconstruct(nq * k * dim, 0);
    gpuIndex.search_and_reconstruct(
            nq,
            xq.data(),
            k,
            testDistance.data(),
            testIndices.data(),
            testReconstruct.data());

    // This handles the search results
    faiss::gpu::compareLists(
            refDistance.data(),
            refIndices.data(),
            testDistance.data(),
            testIndices.data(),
            nq,
            k,
            "SearchAndReconstruct",
            true,
            false,
            true,
            kF32MaxRelErr,
            0.1f,
            0.015f);

    // As the search results may be slightly different (though compareLists
    // above will ensure a decent number of matches), reconstruction should be
    // the same for the vectors that do match
    for (int i = 0; i < nq; ++i) {
        std::unordered_map<faiss::idx_t, int> refLocation;

        for (int j = 0; j < k; ++j) {
            refLocation.insert(std::make_pair(refIndices[i * k + j], j));
        }

        for (int j = 0; j < k; ++j) {
            auto idx = testIndices[i * k + j];
            auto it = refLocation.find(idx);
            if (it != refLocation.end()) {
                for (int d = 0; d < dim; ++d) {
                    EXPECT_EQ(
                            refReconstruct[(i * k + it->second) * dim + d],
                            testReconstruct[(i * k + j) * dim + d]);
                }
            }
        }
    }
}
TEST(TestGpuIndexFlat, SearchAndReconstruct) {
    testSearchAndReconstruct(false);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuIndexFlat, SearchAndReconstruct) {
    testSearchAndReconstruct(true);
}
#endif

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
