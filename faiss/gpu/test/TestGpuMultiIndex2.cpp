/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/IndexPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuMultiIndex2.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/utils/utils.h>
#include <gtest/gtest.h>
#include <ctime>
#include <iostream>
#include <utility>

constexpr float kF32MaxRelErr = 6e-3f;

void testTrain(int d, int numCentroidsPerCodebook, int numOfTrainingVecs) {
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuMultiIndex2 gpuMultiIndex(&res, d, numCentroidsPerCodebook);

    std::vector<float> trainvecs = faiss::gpu::randVecs(numOfTrainingVecs, d);
    gpuMultiIndex.train(numOfTrainingVecs, trainvecs.data());
    EXPECT_EQ(numCentroidsPerCodebook, gpuMultiIndex.getCodebookSize());
    EXPECT_EQ(
            gpuMultiIndex.getNumVecs(),
            gpuMultiIndex.getCodebookSize() * gpuMultiIndex.getCodebookSize());
}

void testSearch(
        int d,
        int numCentroidsPerCodebook,
        int k,
        int numOfQueries,
        int numOfTrainingVecs) {
    faiss::gpu::StandardGpuResources res;
    // res.noTempMemory();
    faiss::gpu::GpuMultiIndex2 gpuMultiIndex(&res, d, numCentroidsPerCodebook);

    { // training
        std::vector<float> trainvecs =
                faiss::gpu::randVecs(numOfTrainingVecs, d);
        gpuMultiIndex.train(numOfTrainingVecs, trainvecs.data());
    }

    { // search
        std::vector<faiss::idx_t> outLabels(k * numOfQueries);
        std::vector<float> outDistances(k * numOfQueries);
        std::vector<float> queries = faiss::gpu::randVecs(numOfQueries, d);

        gpuMultiIndex.search(
                numOfQueries,
                queries.data(),
                k,
                outDistances.data(),
                outLabels.data());

        bool kSequenceError = false;
        for (int i = 0; i < numOfQueries; i++) {
            for (int j = 0; j < k - 1; j++) {
                kSequenceError = kSequenceError ||
                        outDistances[i * k + j] > outDistances[i * k + j + 1];
            }
        }
        EXPECT_FALSE(kSequenceError);

        std::vector<std::pair<ushort, ushort>> outLabelsPair(k * numOfQueries);
        std::vector<float> outDistancesPair(k * numOfQueries);

        gpuMultiIndex.search_pair(
                numOfQueries,
                queries.data(),
                k,
                outDistancesPair.data(),
                outLabelsPair.data());

        bool searchPairError = false;
        for (int i = 0; i < numOfQueries; i++) {
            for (int j = 0; j < k; j++) {
                searchPairError = kSequenceError ||
                        outDistances[i * k + j] !=
                                outDistancesPair[i * k + j] ||
                        outLabels[i * k + j] !=
                                outLabelsPair[i * k + j].first +
                                        numCentroidsPerCodebook *
                                                outLabelsPair[i * k + j].second;
            }
        }
        EXPECT_FALSE(searchPairError);
    }
}

void testComputeResidualPair(int d, int numCentroidsPerCodebook) {
    FAISS_ASSERT(d % 2 == 0 && d > 0);
    FAISS_ASSERT(numCentroidsPerCodebook > 0);

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuMultiIndex2 gpuMultiIndex(&res, d, numCentroidsPerCodebook);
    int numOfTrainingVecs = numCentroidsPerCodebook * 39;

    { // training
        std::vector<float> trainvecs =
                faiss::gpu::randVecs(numOfTrainingVecs, d);
        gpuMultiIndex.train(numOfTrainingVecs, trainvecs.data());
    }

    { // computing residual
        std::vector<float> centroids = gpuMultiIndex.getCentroids();

        std::vector<float> querie = faiss::gpu::randVecs(1, d);
        std::vector<float> residual(d);
        std::vector<int> keyList = {
                0,
                (gpuMultiIndex.getCodebookSize() - 1) / 2,
                gpuMultiIndex.getCodebookSize() - 1};
        for (int i = 0; i < keyList.size(); i++) {
            for (int j = 0; j < keyList.size(); j++) {
                std::pair<ushort, ushort> key(keyList[i], keyList[j]);
                gpuMultiIndex.compute_residual_pair(
                        querie.data(), residual.data(), key);
                for (int k = 0; k < gpuMultiIndex.getSubDim(); k++) {
                    int firstCentroidIdx =
                            key.first * gpuMultiIndex.getSubDim() + k;
                    float expectedValue =
                            querie[k] - centroids[firstCentroidIdx];
                    EXPECT_EQ(residual[k], expectedValue);
                }
                for (int k = gpuMultiIndex.getSubDim(); k < d; k++) {
                    int secondCentroidIdx =
                            (gpuMultiIndex.getCodebookSize() + key.second) *
                                    gpuMultiIndex.getSubDim() +
                            (k - gpuMultiIndex.getSubDim());
                    float expectedValue =
                            querie[k] - centroids[secondCentroidIdx];
                    EXPECT_EQ(residual[k], expectedValue);
                }
            }
        }
    }
}

void testComputeResidualNearestN(int d, int numCentroidsPerCodebook) {
    FAISS_ASSERT(d % 2 == 0 && d > 0);
    FAISS_ASSERT(numCentroidsPerCodebook > 0);

    std::vector<int> numOfQueriesList = {1, 2};
    for (int i = 0; i < numOfQueriesList.size(); i++) {
        faiss::gpu::StandardGpuResources res;
        // res.noTempMemory();
        faiss::gpu::GpuMultiIndex2 gpuMultiIndex(
                &res, d, numCentroidsPerCodebook);
        int numOfTrainingVecs = numCentroidsPerCodebook * 39;

        { // training
            std::vector<float> trainvecs =
                    faiss::gpu::randVecs(numOfTrainingVecs, d);
            gpuMultiIndex.train(numOfTrainingVecs, trainvecs.data());
        }

        { // computing residual
            std::vector<float> centroids = gpuMultiIndex.getCentroids();

            std::vector<float> queries =
                    faiss::gpu::randVecs(numOfQueriesList[i], d);
            std::vector<float> residuals(numOfQueriesList[i] * d);

            gpuMultiIndex.compute_nearest_residual_n(
                    numOfQueriesList[i], queries.data(), residuals.data());

            std::vector<std::pair<ushort, ushort>> keys(numOfQueriesList[i]);
            gpuMultiIndex.assign_pair(
                    numOfQueriesList[i], queries.data(), keys.data());

            int numCodebooks = gpuMultiIndex.getNumCodebooks();
            int subDim = gpuMultiIndex.getSubDim();
            std::vector<float> subQueries(
                    numCodebooks * numOfQueriesList[i] * subDim);
            faiss::fvec_split(
                    subQueries.data(),
                    numCodebooks,
                    queries.data(),
                    (size_t)numOfQueriesList[i],
                    subDim);

            for (int q = 0; q < numOfQueriesList[i]; q++) {
                for (int k = 0; k < subDim; k++) {
                    int firstQueriesIdx = q * subDim + k;
                    int firstCentroidIdx = keys[q].first * subDim + k;
                    float expectedValue = subQueries[firstQueriesIdx] -
                            centroids[firstCentroidIdx];
                    int residualIdx = q * d + k;
                    EXPECT_EQ(residuals[residualIdx], expectedValue);
                }
            }
            for (int q = 0; q < numOfQueriesList[i]; q++) {
                for (int k = subDim; k < d; k++) {
                    int secondQueriesIdx =
                            (numOfQueriesList[i] + q) * subDim + (k - subDim);
                    int secondCentroidIdx =
                            (gpuMultiIndex.getCodebookSize() + keys[q].second) *
                                    subDim +
                            (k - subDim);
                    float expectedValue = subQueries[secondQueriesIdx] -
                            centroids[secondCentroidIdx];
                    int residualIdx = q * d + k;
                    EXPECT_EQ(residuals[residualIdx], expectedValue);
                }
            }
        }
    }
}

void testCopyFrom(
        int d,
        int nbits,
        int numCentroidsPerCodebook,
        int numOfTrainingVecs) {
    constexpr int M = 2;
    faiss::MultiIndexQuantizer cpuIndex(d, M, nbits);

    std::vector<float> vecs = faiss::gpu::randVecs(numOfTrainingVecs, d);
    cpuIndex.train(numOfTrainingVecs, vecs.data());

    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuMultiIndex2Config config;
    faiss::gpu::GpuMultiIndex2 gpuIndex(
            &res, d, numCentroidsPerCodebook, config);
    gpuIndex.copyFrom(&cpuIndex);

    EXPECT_EQ(gpuIndex.ntotal, cpuIndex.ntotal);
    EXPECT_EQ(gpuIndex.getCodebookSize(), numCentroidsPerCodebook);
    EXPECT_EQ(
            gpuIndex.ntotal, numCentroidsPerCodebook * numCentroidsPerCodebook);
    EXPECT_EQ(gpuIndex.getNumCodebooks(), cpuIndex.pq.M);
    EXPECT_EQ(gpuIndex.getSubDim(), cpuIndex.pq.dsub);

    std::vector<float> gpuCentroids = gpuIndex.getCentroids();

    EXPECT_EQ(gpuCentroids, cpuIndex.pq.centroids);
}

void testCopyTo(
        int d,
        int nbits,
        int numCentroidsPerCodebook,
        int numOfTrainingVecs) {
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuMultiIndex2Config config;

    faiss::gpu::GpuMultiIndex2 gpuIndex(
            &res, d, numCentroidsPerCodebook, config);

    std::vector<float> vecs = faiss::gpu::randVecs(numOfTrainingVecs, d);
    gpuIndex.train(numOfTrainingVecs, vecs.data());

    constexpr int M = 2;
    faiss::MultiIndexQuantizer cpuIndex(d, M, nbits);
    gpuIndex.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
    EXPECT_EQ(
            cpuIndex.ntotal, numCentroidsPerCodebook * numCentroidsPerCodebook);
    EXPECT_EQ(cpuIndex.pq.M, gpuIndex.getNumCodebooks());
    EXPECT_EQ(cpuIndex.pq.dsub, gpuIndex.getSubDim());

    std::vector<float> gpuCentroids = gpuIndex.getCentroids();

    EXPECT_EQ(cpuIndex.pq.centroids, gpuCentroids);
}

TEST(TestGpuMultiIndex2, testConstructor) {
    std::vector<int> dList = {2, 4};
    int numCentroidsPerCodebook = 1;
    faiss::gpu::StandardGpuResources res;
    // res.noTempMemory();
    for (int i = 0; i < dList.size(); i++) {
        faiss::gpu::GpuMultiIndex2 gpuMultiIndex(
                &res, dList[i], numCentroidsPerCodebook);
        EXPECT_EQ(gpuMultiIndex.getNumCodebooks(), 2);
        EXPECT_EQ(
                gpuMultiIndex.getSubDim(),
                dList[i] / gpuMultiIndex.getNumCodebooks());
    }
}

TEST(TestGpuMultiIndex2, testTrain) {
    std::vector<int> dList = {2, 4};
    std::vector<int> numCentroidsPerCodebookList = {1, 2};
    for (int i = 0; i < dList.size(); i++) {
        for (int j = 0; j < numCentroidsPerCodebookList.size(); j++) {
            int numTrainingVecs = numCentroidsPerCodebookList[j] * 39;
            testTrain(
                    dList[i], numCentroidsPerCodebookList[j], numTrainingVecs);
        }
    }
}

TEST(TestGpuMultiIndex2, testSearch) {
    int d, numCentroidsPerCodebook, k, numOfQueries, numOfTrainingVecs;
    d = 4;
    numCentroidsPerCodebook = 16;
    k = 8;
    numOfQueries = 1024;
    numOfTrainingVecs = numCentroidsPerCodebook * 39;
    testSearch(d, numCentroidsPerCodebook, k, numOfQueries, numOfTrainingVecs);
}

TEST(TestGpuMultiIndex2, testComputeResidualPair) {
    std::vector<int> dList = {2, 4};
    std::vector<int> numCentroidsPerCodebookList = {1, 2};
    for (int i = 0; i < dList.size(); i++) {
        for (int j = 0; j < numCentroidsPerCodebookList.size(); j++) {
            testComputeResidualPair(dList[i], numCentroidsPerCodebookList[j]);
        }
    }
}

TEST(TestGpuMultiIndex2, testComputeResidualNearestN) {
    std::vector<int> dList = {2, 4};
    std::vector<int> numCentroidsPerCodebookList = {1, 2};
    for (int i = 0; i < dList.size(); i++) {
        for (int j = 0; j < numCentroidsPerCodebookList.size(); j++) {
            testComputeResidualNearestN(
                    dList[i], numCentroidsPerCodebookList[j]);
        }
    }
}

TEST(TestGpuMultiIndex2, copyFrom) {
    std::vector<int> dList = {2, 4};
    std::vector<int> nbitsList = {0, 2};
    std::vector<int> numCentroidsPerCodebookList = {1, 4};
    for (int i = 0; i < dList.size(); i++) {
        for (int j = 0; j < numCentroidsPerCodebookList.size(); j++) {
            int numTrainingVecs = numCentroidsPerCodebookList[j] * 39;
            testCopyFrom(
                    dList[i],
                    nbitsList[j],
                    numCentroidsPerCodebookList[j],
                    numTrainingVecs);
        }
    }
}

TEST(TestGpuMultiIndex2, copyTo) {
    std::vector<int> dList = {2, 4};
    std::vector<int> nbitsList = {0, 2, 2};
    std::vector<int> numCentroidsPerCodebookList = {1, 4, 6};
    for (int i = 0; i < dList.size(); i++) {
        for (int j = 0; j < numCentroidsPerCodebookList.size(); j++) {
            int numTrainingVecs = numCentroidsPerCodebookList[j] * 39;
            testCopyTo(
                    dList[i],
                    nbitsList[j],
                    numCentroidsPerCodebookList[j],
                    numTrainingVecs);
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
