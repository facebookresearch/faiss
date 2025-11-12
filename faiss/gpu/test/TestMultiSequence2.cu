/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Index.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/MultiSequence.cuh>
#include <random>

void fillWithRandom(
        std::mt19937& rng,
        std::uniform_real_distribution<>& distrib,
        float* array,
        int size) {
    for (int i = 0; i < size; i++) {
        array[i] = distrib(rng) * 1000.;
    }
}

template <typename T, typename TVec2>
void test(int w, int numOfQueries, unsigned short numCoarseDistances) {
    constexpr int NUM_CODEBOOKS = 2;
    unsigned inputSize = (unsigned)numOfQueries * numCoarseDistances;
    unsigned outputSize = (unsigned)numOfQueries * w;
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float *d, *d1, *d2;
    T *ids, *i1, *i2;

    d = new float[NUM_CODEBOOKS * inputSize];
    ids = new T[NUM_CODEBOOKS * inputSize];
    d1 = d;
    d2 = d + inputSize;
    i1 = ids;
    i2 = ids + inputSize;

    fillWithRandom(rng, distrib, d1, inputSize);
    fillWithRandom(rng, distrib, d2, inputSize);

    for (unsigned i = 0; i < numOfQueries; i++) {
        std::sort(
                d1 + i * numCoarseDistances,
                d1 + i * numCoarseDistances + numCoarseDistances);
        std::sort(
                d2 + i * numCoarseDistances,
                d2 + i * numCoarseDistances + numCoarseDistances);
    }

    for (unsigned i = 0; i < numOfQueries; i++) {
        for (unsigned j = 0; j < numCoarseDistances; j++) {
            i1[i * numCoarseDistances + j] = j;
            i2[i * numCoarseDistances + j] = j;
        }
    }

    for (unsigned i = 0; i < numOfQueries; i++) {
        std::random_shuffle(
                &i1[i * numCoarseDistances], &i1[(i + 1) * numCoarseDistances]);
        std::random_shuffle(
                &i2[i * numCoarseDistances], &i2[(i + 1) * numCoarseDistances]);
    }

    faiss::gpu::StandardGpuResources provider;
    int device = 0;
    cudaStream_t stream =
            provider.getResources()->getDefaultStreamCurrentDevice();

    auto inDistances = faiss::gpu::toDeviceTemporary<float, 3>(
            provider.getResources().get(),
            device,
            const_cast<float*>(d),
            stream,
            {NUM_CODEBOOKS, numOfQueries, numCoarseDistances});

    auto inIndices = faiss::gpu::toDeviceTemporary<T, 3>(
            provider.getResources().get(),
            device,
            const_cast<T*>(ids),
            stream,
            {NUM_CODEBOOKS, numOfQueries, numCoarseDistances});

    faiss::gpu::DeviceTensor<float, 2, true> outDistances(
            provider.getResources().get(),
            faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
            {numOfQueries, w});

    faiss::gpu::DeviceTensor<TVec2, 2, true> outIndices(
            provider.getResources().get(),
            faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
            {numOfQueries, w});

    faiss::gpu::runMultiSequence2(
            numOfQueries,
            numCoarseDistances,
            w,
            inDistances,
            inIndices,
            outDistances,
            outIndices,
            provider.getResources().get());

    float* dr;
    TVec2* ir;
    dr = new float[outputSize];
    ir = new TVec2[outputSize];

    faiss::gpu::fromDevice<float, 2>(outDistances, dr, stream);
    faiss::gpu::fromDevice<TVec2, 2>(outIndices, ir, stream);

    faiss::gpu::CudaEvent copyEnd(stream);
    copyEnd.cpuWaitOnEvent();

    // checking for errors
    bool error = false;
    for (unsigned i = 0; i < numOfQueries; i++) {
        for (unsigned j = 1; j < w; j++) {
            int pos = i * w + j;
            error = error || dr[pos - 1] > dr[pos] ||
                    (ir[pos - 1].x == ir[pos].x && ir[pos - 1].y == ir[pos].y);
        }
    }
    EXPECT_FALSE(error) << "Error with w: " << w
                        << " numOfQueries: " << numOfQueries
                        << " numCoarseDistances: " << numCoarseDistances;

    delete[] d;
    delete[] ids;
    delete[] dr;
    delete[] ir;
}

template <typename T>
void testUnifiedIndex(
        int w,
        int numOfQueries,
        unsigned short numCoarseDistances,
        int codebookSize) {
    constexpr int NUM_CODEBOOKS = 2;
    unsigned inputSize = (unsigned)numOfQueries * numCoarseDistances;
    unsigned outputSize = (unsigned)numOfQueries * w;
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float *d, *d1, *d2;
    T *ids, *i1, *i2;

    d = new float[NUM_CODEBOOKS * inputSize];
    ids = new T[NUM_CODEBOOKS * inputSize];
    d1 = d;
    d2 = d + inputSize;
    i1 = ids;
    i2 = ids + inputSize;

    fillWithRandom(rng, distrib, d1, inputSize);
    fillWithRandom(rng, distrib, d2, inputSize);

    for (unsigned i = 0; i < numOfQueries; i++) {
        std::sort(
                d1 + i * numCoarseDistances,
                d1 + i * numCoarseDistances + numCoarseDistances);
        std::sort(
                d2 + i * numCoarseDistances,
                d2 + i * numCoarseDistances + numCoarseDistances);
    }

    for (unsigned i = 0; i < numOfQueries; i++) {
        for (unsigned j = 0; j < numCoarseDistances; j++) {
            i1[i * numCoarseDistances + j] = j;
            i2[i * numCoarseDistances + j] = j;
        }
    }

    for (unsigned i = 0; i < numOfQueries; i++) {
        std::random_shuffle(
                &i1[i * numCoarseDistances], &i1[(i + 1) * numCoarseDistances]);
        std::random_shuffle(
                &i2[i * numCoarseDistances], &i2[(i + 1) * numCoarseDistances]);
    }

    faiss::gpu::StandardGpuResources provider;
    int device = 0;
    cudaStream_t stream =
            provider.getResources()->getDefaultStreamCurrentDevice();

    auto inDistances = faiss::gpu::toDeviceTemporary<float, 3>(
            provider.getResources().get(),
            device,
            const_cast<float*>(d),
            stream,
            {NUM_CODEBOOKS, numOfQueries, numCoarseDistances});

    auto inIndices = faiss::gpu::toDeviceTemporary<T, 3>(
            provider.getResources().get(),
            device,
            const_cast<T*>(ids),
            stream,
            {NUM_CODEBOOKS, numOfQueries, numCoarseDistances});

    faiss::gpu::DeviceTensor<float, 2, true> outDistances(
            provider.getResources().get(),
            faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
            {numOfQueries, w});

    faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> outIndices(
            provider.getResources().get(),
            faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
            {numOfQueries, w});

    faiss::gpu::runMultiSequence2(
            numOfQueries,
            numCoarseDistances,
            w,
            inDistances,
            inIndices,
            outDistances,
            codebookSize,
            outIndices,
            provider.getResources().get());

    float* dr;
    faiss::idx_t* ir;
    dr = new float[outputSize];
    ir = new faiss::idx_t[outputSize];

    faiss::gpu::fromDevice<float, 2>(outDistances, dr, stream);
    faiss::gpu::fromDevice<faiss::idx_t, 2>(outIndices, ir, stream);

    faiss::gpu::CudaEvent copyEnd(stream);
    copyEnd.cpuWaitOnEvent();

    // checking for errors
    bool error = false;
    for (unsigned i = 0; i < numOfQueries; i++) {
        for (unsigned j = 1; j < w; j++) {
            int pos = i * w + j;
            error = error || dr[pos - 1] > dr[pos] || ir[pos - 1] == ir[pos];
        }
    }
    EXPECT_FALSE(error) << "Error with w: " << w
                        << " numOfQueries: " << numOfQueries
                        << " numCoarseDistances: " << numCoarseDistances;

    delete[] d;
    delete[] ids;
    delete[] dr;
    delete[] ir;
}

TEST(TestGpuMultiSequence2, testWEqualKShort) {
    std::vector<int> wList = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    std::vector<int> nuMQuerisList = {1, 1024, 32768};
    for (int i = 0; i < wList.size(); i++) {
        for (int j = 0; j < nuMQuerisList.size(); j++) {
            test<ushort, ushort2>(wList[i], nuMQuerisList[j], wList[i]);
        }
    }
}

TEST(TestGpuMultiSequence2, testWLowerThanKShort) {
    std::vector<int> kList = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < kList.size(); i++) {
        test<ushort, ushort2>(kList[i] / 2, 1024, kList[i]);
    }
}

TEST(TestGpuMultiSequence2, testWEqualKInt) {
    std::vector<int> wList = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    std::vector<int> nuMQuerisList = {1, 1024, 32768};
    for (int i = 0; i < wList.size(); i++) {
        for (int j = 0; j < nuMQuerisList.size(); j++) {
            test<faiss::idx_t, int2>(wList[i], nuMQuerisList[j], wList[i]);
        }
    }
}

TEST(TestGpuMultiSequence2, testWLowerThanKInt) {
    std::vector<int> kList = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < kList.size(); i++) {
        test<faiss::idx_t, int2>(kList[i] / 2, 1024, kList[i]);
    }
}

TEST(TestGpuMultiSequence2, testWEqualKUnified) {
    std::vector<int> wList = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    std::vector<int> nuMQuerisList = {1, 1024, 32768};
    for (int i = 0; i < wList.size(); i++) {
        for (int j = 0; j < nuMQuerisList.size(); j++) {
            testUnifiedIndex<ushort>(
                    wList[i], nuMQuerisList[j], wList[i], wList[i]);
        }
    }
}

TEST(TestGpuMultiSequence2, testWLowerThanKUnified) {
    std::vector<int> kList = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < kList.size(); i++) {
        testUnifiedIndex<ushort>(kList[i] / 2, 1024, kList[i], kList[i]);
    }
}

TEST(TestGpuMultiSequence2, testLimits) {
    test<ushort, ushort2>(36, 1, 6);
    test<ushort, ushort2>(36, 1024, 6);
    test<ushort, ushort2>(200, 1, 64);
    test<ushort, ushort2>(200, 1024, 64);
    test<ushort, ushort2>(2048, 1, 1024);
    test<ushort, ushort2>(2048, 1024, 1024);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    return 0;
}
