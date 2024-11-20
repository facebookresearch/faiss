// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <sstream>
#include <vector>

enum class TestThresholds {
    Normal,
    BF16,
    // Linf has worse error than the other metrics for bf16
    BF16_Linf,
};

void evaluate_bfknn(
        faiss::gpu::GpuDistanceParams& args,
        faiss::gpu::GpuResourcesProvider* res,
        std::vector<float>& cpuDistance,
        std::vector<faiss::idx_t>& cpuIndices,
        std::vector<float>& gpuDistance,
        std::vector<faiss::idx_t>& gpuIndices,
        int numQuery,
        int k,
        bool colMajorVecs,
        bool colMajorQueries,
        faiss::MetricType metric,
        TestThresholds thresh = TestThresholds::Normal) {
    using namespace faiss::gpu;

    bfKnn(res, args);

    std::stringstream str;
    str << "using cuVS " << args.use_cuvs << " metric " << metric
        << " colMajorVecs " << colMajorVecs << " colMajorQueries "
        << colMajorQueries;

    float maxRelativeError;
    float pctMaxDiff1;
    float pctMaxDiffN;

    switch (thresh) {
        case TestThresholds::Normal:
            maxRelativeError = 6e-3f;
            pctMaxDiff1 = 0.1f;
            pctMaxDiffN = 0.015f;
            break;
        case TestThresholds::BF16:
            maxRelativeError = 1.5e-2f;
            pctMaxDiff1 = 0.3f;
            pctMaxDiffN = 0.1f;
            break;
        case TestThresholds::BF16_Linf:
            maxRelativeError = 1.5e-2f;
            pctMaxDiff1 = 0.53f;
            pctMaxDiffN = 0.2f;
            break;
    }

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
            maxRelativeError,
            pctMaxDiff1,
            pctMaxDiffN);
}

void testTransposition(
        bool colMajorVecs,
        bool colMajorQueries,
        faiss::MetricType metric,
        bool use_cuvs = false,
        float metricArg = 0) {
    using namespace faiss::gpu;

    int device = randVal(0, getNumDevices() - 1);

    StandardGpuResources res;
    res.noTempMemory();

    // The transpose and distance code assumes the desired device is already set
    DeviceScope scope(device);
    auto stream = res.getDefaultStream(device);

    int dim = randVal(20, 150);
    int numVecs = randVal(10, 30000);
    int numQuery = randVal(1, 1024);
    int k = std::min(numVecs, randVal(20, 70));

    // Input data for CPU
    std::vector<float> vecs = randVecs(numVecs, dim);
    std::vector<float> queries = randVecs(numQuery, dim);

    if ((metric == faiss::MetricType::METRIC_JensenShannon) ||
        (metric == faiss::MetricType::METRIC_Jaccard)) {
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

#if defined USE_NVIDIA_CUVS
    args.use_cuvs = use_cuvs;
#else
    FAISS_THROW_IF_NOT_MSG(
            !use_cuvs,
            "cuVS has not been compiled into the current version so it cannot be used.");
#endif

    evaluate_bfknn(
            args,
            &res,
            cpuDistance,
            cpuIndices,
            gpuDistance,
            gpuIndices,
            numQuery,
            k,
            colMajorVecs,
            colMajorQueries,
            metric);
}

void testTransposition_bf16(
        bool colMajorVecs,
        bool colMajorQueries,
        faiss::MetricType metric,
        bool use_raft = false,
        float metricArg = 0) {
    using namespace faiss::gpu;

#ifdef USE_AMD_ROCM
    std::cout << "skipping bfloat16 test (no bfloat16 support on AMD)\n";
    EXPECT_TRUE(true);
    return;
#else
    int device = randVal(0, getNumDevices() - 1);

    StandardGpuResources res;
    if (!res.supportsBFloat16(device)) {
        std::cout << "skipping bfloat16 test (no bfloat16 support on device)\n";
        return;
    }

    res.noTempMemory();
    // The transpose and distance code assumes the desired device is already set
    DeviceScope scope(device);
    auto stream = res.getDefaultStream(device);

    int dim = randVal(20, 150);
    int numVecs = randVal(10, 30000);
    int numQuery = randVal(1, 1024);
    int k = std::min(numVecs, randVal(20, 70));

    // Input data for CPU
    std::vector<float> vecs = randVecs(numVecs, dim);
    std::vector<float> queries = randVecs(numQuery, dim);

    if ((metric == faiss::MetricType::METRIC_JensenShannon) ||
        (metric == faiss::MetricType::METRIC_Jaccard)) {
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

    // Convert float32 data to bfloat16 via truncation not rounding
    // (just copy high 2 bytes)
    std::vector<uint16_t> bf16_vecs(vecs.size());
    std::vector<uint16_t> bf16_queries(queries.size());

    auto fn_f32_bf16 = [](float v) {
        uint32_t vi;
        std::memcpy(&vi, &v, sizeof(uint32_t));
        return uint16_t(vi >> 16);
    };

    std::transform(vecs.begin(), vecs.end(), bf16_vecs.begin(), fn_f32_bf16);
    std::transform(
            queries.begin(), queries.end(), bf16_queries.begin(), fn_f32_bf16);

    // Copy input data to GPU, and pre-transpose both vectors and queries for
    // passing. Just use uint16_t in lieu of __nv_bfloat16
    auto gpuVecs = toDeviceNonTemporary<uint16_t, 2>(
            res.getResources().get(),
            device,
            bf16_vecs.data(),
            stream,
            {numVecs, dim});
    auto gpuQueries = toDeviceNonTemporary<uint16_t, 2>(
            res.getResources().get(),
            device,
            bf16_queries.data(),
            stream,
            {numQuery, dim});

    DeviceTensor<uint16_t, 2, true> vecsT(
            res.getResources().get(),
            makeDevAlloc(AllocType::Other, stream),
            {dim, numVecs});
    runTransposeAny(gpuVecs, 0, 1, vecsT, stream);

    DeviceTensor<uint16_t, 2, true> queriesT(
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
    args.vectorType = DistanceDataType::BF16;
    args.vectorsRowMajor = !colMajorVecs;
    args.numVectors = numVecs;
    args.queries = colMajorQueries ? queriesT.data() : gpuQueries.data();
    args.queryType = DistanceDataType::BF16;
    args.queriesRowMajor = !colMajorQueries;
    args.numQueries = numQuery;
    args.outDistances = gpuDistance.data();
    args.outIndices = gpuIndices.data();
    args.device = device;

    evaluate_bfknn(
            args,
            &res,
            cpuDistance,
            cpuIndices,
            gpuDistance,
            gpuIndices,
            numQuery,
            k,
            colMajorVecs,
            colMajorQueries,
            metric,
            metric == faiss::MetricType::METRIC_Linf ? TestThresholds::BF16_Linf
                                                     : TestThresholds::BF16);
#endif
}

// Test different memory layouts for brute-force k-NN
TEST(TestGpuDistance, Transposition_RR) {
    testTransposition(false, false, faiss::MetricType::METRIC_L2);
    testTransposition(false, false, faiss::MetricType::METRIC_INNER_PRODUCT);
}

TEST(TestGpuDistance, Transposition_RR_BF16) {
    testTransposition_bf16(false, false, faiss::MetricType::METRIC_L2);
    testTransposition_bf16(
            false, false, faiss::MetricType::METRIC_INNER_PRODUCT);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, Transposition_RR) {
    testTransposition(false, false, faiss::MetricType::METRIC_L2, true);
    testTransposition(
            false, false, faiss::MetricType::METRIC_INNER_PRODUCT, true);
}
#endif

TEST(TestGpuDistance, Transposition_RC) {
    testTransposition(false, true, faiss::MetricType::METRIC_L2);
}

TEST(TestGpuDistance, Transposition_RC_BF16) {
    testTransposition_bf16(false, true, faiss::MetricType::METRIC_L2);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, Transposition_RC) {
    testTransposition(false, true, faiss::MetricType::METRIC_L2, true);
}
#endif

TEST(TestGpuDistance, Transposition_CR) {
    testTransposition(true, false, faiss::MetricType::METRIC_L2);
}

TEST(TestGpuDistance, Transposition_CR_BF16) {
    testTransposition_bf16(true, false, faiss::MetricType::METRIC_L2);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, Transposition_CR) {
    testTransposition(true, false, faiss::MetricType::METRIC_L2, true);
}
#endif

TEST(TestGpuDistance, Transposition_CC) {
    testTransposition(true, true, faiss::MetricType::METRIC_L2);
}

TEST(TestGpuDistance, Transposition_CC_BF16) {
    testTransposition_bf16(true, true, faiss::MetricType::METRIC_L2);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, Transposition_CC) {
    testTransposition(true, true, faiss::MetricType::METRIC_L2, true);
}
#endif

TEST(TestGpuDistance, L1) {
    testTransposition(false, false, faiss::MetricType::METRIC_L1);
}

TEST(TestGpuDistance, L1_BF16) {
    testTransposition_bf16(false, false, faiss::MetricType::METRIC_L1);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, L1) {
    testTransposition(false, false, faiss::MetricType::METRIC_L1, true);
}
#endif

// Test other transpositions with the general distance kernel
TEST(TestGpuDistance, L1_RC) {
    testTransposition(false, true, faiss::MetricType::METRIC_L1);
}

#if defined USE_NVIDIA_CUVS
// Test other transpositions with the general distance kernel
TEST(TestCuvsGpuDistance, L1_RC) {
    testTransposition(false, true, faiss::MetricType::METRIC_L1, true);
}
#endif

TEST(TestGpuDistance, L1_RC_BF16) {
    testTransposition_bf16(false, true, faiss::MetricType::METRIC_L1);
}

TEST(TestGpuDistance, L1_CR) {
    testTransposition(true, false, faiss::MetricType::METRIC_L1);
}

TEST(TestGpuDistance, L1_CR_BF16) {
    testTransposition_bf16(true, false, faiss::MetricType::METRIC_L1);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, L1_CR) {
    testTransposition(true, false, faiss::MetricType::METRIC_L1, true);
}
#endif

TEST(TestGpuDistance, L1_CC) {
    testTransposition(true, true, faiss::MetricType::METRIC_L1);
}

TEST(TestGpuDistance, L1_CC_BF16) {
    testTransposition_bf16(true, true, faiss::MetricType::METRIC_L1);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, L1_CC) {
    testTransposition(true, true, faiss::MetricType::METRIC_L1, true);
}
#endif

// Test remainder of metric types
TEST(TestGpuDistance, Linf) {
    testTransposition(false, false, faiss::MetricType::METRIC_Linf);
}

#if defined USE_NVIDIA_CUVS
// Test remainder of metric types
TEST(TestCuvsGpuDistance, Linf) {
    testTransposition(false, false, faiss::MetricType::METRIC_Linf, true);
}
#endif

TEST(TestGpuDistance, Linf_BF16) {
    testTransposition_bf16(false, false, faiss::MetricType::METRIC_Linf);
}

TEST(TestGpuDistance, Lp) {
    testTransposition(false, false, faiss::MetricType::METRIC_Lp, false, 3);
}

TEST(TestGpuDistance, Lp_BF16) {
    testTransposition_bf16(
            false, false, faiss::MetricType::METRIC_Lp, false, 3);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, Lp) {
    testTransposition(false, false, faiss::MetricType::METRIC_Lp, true, 3);
}
#endif

TEST(TestGpuDistance, Canberra) {
    testTransposition(false, false, faiss::MetricType::METRIC_Canberra);
}

TEST(TestGpuDistance, Canberra_BF16) {
    testTransposition_bf16(false, false, faiss::MetricType::METRIC_Canberra);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, Canberra) {
    testTransposition(false, false, faiss::MetricType::METRIC_Canberra, true);
}
#endif

TEST(TestGpuDistance, BrayCurtis) {
    testTransposition(false, false, faiss::MetricType::METRIC_BrayCurtis);
}

TEST(TestGpuDistance, BrayCurtis_BF16) {
    testTransposition_bf16(false, false, faiss::MetricType::METRIC_BrayCurtis);
}

TEST(TestGpuDistance, JensenShannon) {
    testTransposition(false, false, faiss::MetricType::METRIC_JensenShannon);
}

TEST(TestGpuDistance, JensenShannon_BF16) {
    testTransposition_bf16(
            false, false, faiss::MetricType::METRIC_JensenShannon);
}

#if defined USE_NVIDIA_CUVS
TEST(TestCuvsGpuDistance, JensenShannon) {
    testTransposition(
            false, false, faiss::MetricType::METRIC_JensenShannon, true);
}
#endif

TEST(TestGpuDistance, Jaccard) {
    testTransposition(false, false, faiss::MetricType::METRIC_Jaccard);
}

TEST(TestGpuDistance, Jaccard_BF16) {
    testTransposition_bf16(false, false, faiss::MetricType::METRIC_Jaccard);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
