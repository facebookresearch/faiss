#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <gtest/gtest.h>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>

void testRunCalcResidualMultiIndex2(int numOfQueries, int d, int codebookSize) {
    FAISS_ASSERT(d % 2 == 0 && d > 0);
    FAISS_ASSERT(codebookSize > 0);
    FAISS_ASSERT(sizeof(ushort2) == sizeof(std::pair<ushort, ushort>));
    constexpr int NUM_CODEBOOKS = 2;
    std::vector<float> queries = faiss::gpu::randVecs(
            NUM_CODEBOOKS * numOfQueries, d / NUM_CODEBOOKS);
    std::vector<float> centroids = faiss::gpu::randVecs(
            NUM_CODEBOOKS * codebookSize, d / NUM_CODEBOOKS);

    faiss::gpu::StandardGpuResources provider;
    int device = 0;
    cudaStream_t stream =
            provider.getResources()->getDefaultStreamCurrentDevice();

    auto inQueries = faiss::gpu::toDeviceTemporary<float, 2>(
            provider.getResources().get(),
            device,
            const_cast<float*>(queries.data()),
            stream,
            {NUM_CODEBOOKS * numOfQueries, d / NUM_CODEBOOKS});

    auto inCentroids = faiss::gpu::toDeviceTemporary<float, 2>(
            provider.getResources().get(),
            device,
            const_cast<float*>(centroids.data()),
            stream,
            {NUM_CODEBOOKS * codebookSize, d / NUM_CODEBOOKS});

    faiss::gpu::DeviceTensor<float, 2, true> outResiduals(
            provider.getResources().get(),
            faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
            {numOfQueries, d});

    std::vector<int> keyList = {
            0, (codebookSize - 1) / NUM_CODEBOOKS, codebookSize - 1};
    for (int i = 0; i < keyList.size(); i++) {
        for (int j = 0; j < keyList.size(); j++) {
            std::vector<std::pair<ushort, ushort>> keys(numOfQueries);
            for (int k = 0; k < keys.size(); k++) {
                keys[k].first = keyList[i];
                keys[k].second = keyList[j];
            }

            auto inKeys = faiss::gpu::toDeviceTemporary<ushort2, 1>(
                    provider.getResources().get(),
                    device,
                    (ushort2*)(keys.data()),
                    stream,
                    {numOfQueries});

            faiss::gpu::runCalcResidual(
                    inQueries, inCentroids, inKeys, outResiduals, stream);

            std::vector<float> residuals(numOfQueries * d);
            faiss::gpu::fromDevice<float, 2>(
                    outResiduals, residuals.data(), stream);

            faiss::gpu::CudaEvent copyEnd(stream);
            copyEnd.cpuWaitOnEvent();

            for (int q = 0; q < numOfQueries; q++) {
                for (int k = 0; k < d / NUM_CODEBOOKS; k++) {
                    int firstQueriesIdx = q * d / NUM_CODEBOOKS + k;
                    int firstCentroidIdx = keyList[i] * d / NUM_CODEBOOKS + k;
                    float expectedValue = queries[firstQueriesIdx] -
                            centroids[firstCentroidIdx];
                    int residualIdx = q * d + k;
                    EXPECT_EQ(residuals[residualIdx], expectedValue);
                }
            }
            for (int q = 0; q < numOfQueries; q++) {
                for (int k = d / NUM_CODEBOOKS; k < d; k++) {
                    int secondQueriesIdx =
                            (numOfQueries + q) * d / NUM_CODEBOOKS +
                            (k - d / NUM_CODEBOOKS);
                    int secondCentroidIdx =
                            (codebookSize + keyList[j]) * d / NUM_CODEBOOKS +
                            (k - d / NUM_CODEBOOKS);
                    float expectedValue = queries[secondQueriesIdx] -
                            centroids[secondCentroidIdx];
                    int residualIdx = q * d + k;
                    EXPECT_EQ(residuals[residualIdx], expectedValue);
                }
            }
        }
    }
}

TEST(VectorResidual, testRunCalcResidualMultiIndex2) {
    std::vector<int> dList = {2, 6, 258};
    std::vector<int> codebookSizeList = {1, 2};
    std::vector<int> numOfQueriesList = {1, 2};
    for (int i = 0; i < dList.size(); i++) {
        for (int j = 0; j < codebookSizeList.size(); j++) {
            for (int k = 0; k < numOfQueriesList.size(); k++) {
                testRunCalcResidualMultiIndex2(
                        numOfQueriesList[k], dList[i], codebookSizeList[j]);
            }
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    return 0;
}
