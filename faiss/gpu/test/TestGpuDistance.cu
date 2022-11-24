/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <sstream>
#include <vector>

void testTransposition(
        bool colMajorVecs,
        bool colMajorQueries,
        faiss::MetricType metric,
        float metricArg = 0) {
    using namespace faiss::gpu;

    int device = randVal(0, getNumDevices() - 1);

    StandardGpuResources res;
    res.noTempMemory();

    int dim = randVal(20, 150);
    int numVecs = randVal(10, 30000);
    int numQuery = randVal(1, 1024);
    int k = std::min(numVecs, randVal(20, 70));

    // Input data for CPU
    std::vector<float> vecs = randVecs(numVecs, dim);
    std::vector<float> queries = randVecs(numQuery, dim);

    if (metric == faiss::MetricType::METRIC_JensenShannon) {
        // make values positive
        for (auto& v : vecs) {
            v = std::abs(v);
            if (v == 0) {
                v = 1e-6;
            }
        }

        for (auto& q : queries) {
            q = std::abs(q);
            if (q == 0) {
                q = 1e-6;
            }
        }
    }

    // The CPU index is our reference for the results
    faiss::IndexFlat cpuIndex(dim, metric);
    cpuIndex.metric_arg = metricArg;
    cpuIndex.add(numVecs, vecs.data());

    std::vector<float> cpuDistance(numQuery * k, 0);
    std::vector<faiss::idx_t> cpuIndices(numQuery * k, -1);

    cpuIndex.search(
            numQuery, queries.data(), k, cpuDistance.data(), cpuIndices.data());

    // The transpose and distance code assumes the desired device is already set
    DeviceScope scope(device);
    auto stream = res.getDefaultStream(device);

    // Copy input data to GPU, and pre-transpose both vectors and queries for
    // passing
    auto gpuVecs = toDeviceNonTemporary<float, 2>(
            res.getResources().get(),
            device,
            vecs.data(),
            stream,
            {numVecs, dim});
    auto gpuQueries = toDeviceNonTemporary<float, 2>(
            res.getResources().get(),
            device,
            queries.data(),
            stream,
            {numQuery, dim});

    DeviceTensor<float, 2, true> vecsT(
            res.getResources().get(),
            makeDevAlloc(AllocType::Other, stream),
            {dim, numVecs});
    runTransposeAny(gpuVecs, 0, 1, vecsT, stream);

    DeviceTensor<float, 2, true> queriesT(
            res.getResources().get(),
            makeDevAlloc(AllocType::Other, stream),
            {dim, numQuery});
    runTransposeAny(gpuQueries, 0, 1, queriesT, stream);

    std::vector<float> gpuDistance(numQuery * k, 0);
    std::vector<faiss::idx_t> gpuIndices(numQuery * k, -1);

    GpuDistanceParams args;
    args.metric = metric;
    args.metricArg = metricArg;
    args.k = k;
    args.dims = dim;
    args.vectors = colMajorVecs ? vecsT.data() : gpuVecs.data();
    args.vectorsRowMajor = !colMajorVecs;
    args.numVectors = numVecs;
    args.queries = colMajorQueries ? queriesT.data() : gpuQueries.data();
    args.queriesRowMajor = !colMajorQueries;
    args.numQueries = numQuery;
    args.outDistances = gpuDistance.data();
    args.outIndices = gpuIndices.data();
    args.device = device;

    bfKnn(&res, args);

    std::stringstream str;
    str << "metric " << metric << " colMajorVecs " << colMajorVecs
        << " colMajorQueries " << colMajorQueries;

    compareLists(
            cpuDistance.data(),
            cpuIndices.data(),
            gpuDistance.data(),
            gpuIndices.data(),
            numQuery,
            k,
            str.str(),
            false,
            false,
            true,
            6e-3f,
            0.1f,
            0.015f);
}

// Test different memory layouts for brute-force k-NN
TEST(TestGpuDistance, Transposition_RR) {
    testTransposition(false, false, faiss::MetricType::METRIC_L2);
    testTransposition(false, false, faiss::MetricType::METRIC_INNER_PRODUCT);
}

TEST(TestGpuDistance, Transposition_RC) {
    testTransposition(false, true, faiss::MetricType::METRIC_L2);
}

TEST(TestGpuDistance, Transposition_CR) {
    testTransposition(true, false, faiss::MetricType::METRIC_L2);
}

TEST(TestGpuDistance, Transposition_CC) {
    testTransposition(true, true, faiss::MetricType::METRIC_L2);
}

TEST(TestGpuDistance, L1) {
    testTransposition(false, false, faiss::MetricType::METRIC_L1);
}

// Test other transpositions with the general distance kernel
TEST(TestGpuDistance, L1_RC) {
    testTransposition(false, true, faiss::MetricType::METRIC_L1);
}

TEST(TestGpuDistance, L1_CR) {
    testTransposition(true, false, faiss::MetricType::METRIC_L1);
}

TEST(TestGpuDistance, L1_CC) {
    testTransposition(true, true, faiss::MetricType::METRIC_L1);
}

// Test remainder of metric types
TEST(TestGpuDistance, Linf) {
    testTransposition(false, false, faiss::MetricType::METRIC_Linf);
}

TEST(TestGpuDistance, Lp) {
    testTransposition(false, false, faiss::MetricType::METRIC_Lp, 3);
}

TEST(TestGpuDistance, Canberra) {
    testTransposition(false, false, faiss::MetricType::METRIC_Canberra);
}

TEST(TestGpuDistance, BrayCurtis) {
    testTransposition(false, false, faiss::MetricType::METRIC_BrayCurtis);
}

TEST(TestGpuDistance, JensenShannon) {
    testTransposition(false, false, faiss::MetricType::METRIC_JensenShannon);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
