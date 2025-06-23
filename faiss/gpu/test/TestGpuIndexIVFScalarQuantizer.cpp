/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>

constexpr float kF32MaxRelErr = 0.03f;

struct Options {
    Options() {
        numAdd = 2 * faiss::gpu::randVal(2000, 5000);
        dim = faiss::gpu::randVal(64, 200);

        numCentroids = std::sqrt((float)numAdd / 2);
        numTrain = numCentroids * 40;
        nprobe = faiss::gpu::randVal(std::min(10, numCentroids), numCentroids);
        numQuery = faiss::gpu::randVal(32, 100);

        // Due to the approximate nature of the query and of floating point
        // differences between GPU and CPU, to stay within our error bounds,
        // only use a small k
        k = std::min(faiss::gpu::randVal(10, 30), numAdd / 40);
        indicesOpt = faiss::gpu::randSelect(
                {faiss::gpu::INDICES_CPU,
                 faiss::gpu::INDICES_32_BIT,
                 faiss::gpu::INDICES_64_BIT});

        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    }

    std::string toString() const {
        std::stringstream str;
        str << "IVFFlat device " << device << " numVecs " << numAdd << " dim "
            << dim << " numCentroids " << numCentroids << " nprobe " << nprobe
            << " numQuery " << numQuery << " k " << k << " indicesOpt "
            << indicesOpt;

        return str.str();
    }

    int numAdd;
    int dim;
    int numCentroids;
    int numTrain;
    int nprobe;
    int numQuery;
    int k;
    int device;
    faiss::gpu::IndicesOptions indicesOpt;
};

void runCopyToTest(faiss::ScalarQuantizer::QuantizerType qtype) {
    using namespace faiss;
    using namespace faiss::gpu;

    Options opt;
    std::vector<float> trainVecs = randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = randVecs(opt.numAdd, opt.dim);

    StandardGpuResources res;
    res.noTempMemory();

    auto config = GpuIndexIVFScalarQuantizerConfig();
    config.device = opt.device;

    GpuIndexIVFScalarQuantizer gpuIndex(
            &res, opt.dim, opt.numCentroids, qtype, METRIC_L2, true, config);
    gpuIndex.train(opt.numTrain, trainVecs.data());
    gpuIndex.add(opt.numAdd, addVecs.data());
    gpuIndex.nprobe = opt.nprobe;

    // use garbage values to see if we overwrite then
    IndexFlatL2 cpuQuantizer(1);
    IndexIVFScalarQuantizer cpuIndex(
            &cpuQuantizer,
            1,
            1,
            ScalarQuantizer::QuantizerType::QT_6bit,
            METRIC_L2);
    cpuIndex.nprobe = 1;

    gpuIndex.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);

    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
    EXPECT_EQ(cpuIndex.quantizer->d, gpuIndex.quantizer->d);
    EXPECT_EQ(cpuIndex.d, opt.dim);
    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);

    testIVFEquality(cpuIndex, gpuIndex);

    // Query both objects; results should be equivalent
    compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            kF32MaxRelErr,
            0.1f,
            0.015f);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_fp16) {
    runCopyToTest(faiss::ScalarQuantizer::QuantizerType::QT_fp16);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_8bit) {
    runCopyToTest(faiss::ScalarQuantizer::QuantizerType::QT_8bit);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_8bit_uniform) {
    runCopyToTest(faiss::ScalarQuantizer::QuantizerType::QT_8bit_uniform);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_6bit) {
    runCopyToTest(faiss::ScalarQuantizer::QuantizerType::QT_6bit);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_4bit) {
    runCopyToTest(faiss::ScalarQuantizer::QuantizerType::QT_4bit);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_4bit_uniform) {
    runCopyToTest(faiss::ScalarQuantizer::QuantizerType::QT_4bit_uniform);
}

void runCopyFromTest(faiss::ScalarQuantizer::QuantizerType qtype) {
    using namespace faiss;
    using namespace faiss::gpu;

    Options opt;
    std::vector<float> trainVecs = randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = randVecs(opt.numAdd, opt.dim);

    IndexFlatL2 cpuQuantizer(opt.dim);
    IndexIVFScalarQuantizer cpuIndex(
            &cpuQuantizer, opt.dim, opt.numCentroids, qtype, METRIC_L2);

    cpuIndex.nprobe = opt.nprobe;
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());

    // use garbage values to see if we overwrite then
    StandardGpuResources res;
    res.noTempMemory();

    auto config = GpuIndexIVFScalarQuantizerConfig();
    config.device = opt.device;

    GpuIndexIVFScalarQuantizer gpuIndex(
            &res,
            1,
            1,
            ScalarQuantizer::QuantizerType::QT_4bit,
            METRIC_L2,
            false,
            config);
    gpuIndex.nprobe = 1;

    gpuIndex.copyFrom(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);

    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
    EXPECT_EQ(cpuIndex.d, opt.dim);
    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);

    testIVFEquality(cpuIndex, gpuIndex);

    // Query both objects; results should be equivalent
    compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            kF32MaxRelErr,
            0.1f,
            0.015f);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_fp16) {
    runCopyFromTest(faiss::ScalarQuantizer::QuantizerType::QT_fp16);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_8bit) {
    runCopyFromTest(faiss::ScalarQuantizer::QuantizerType::QT_8bit);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_8bit_uniform) {
    runCopyFromTest(faiss::ScalarQuantizer::QuantizerType::QT_8bit_uniform);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_6bit) {
    runCopyFromTest(faiss::ScalarQuantizer::QuantizerType::QT_6bit);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_4bit) {
    runCopyFromTest(faiss::ScalarQuantizer::QuantizerType::QT_4bit);
}

TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_4bit_uniform) {
    runCopyFromTest(faiss::ScalarQuantizer::QuantizerType::QT_4bit_uniform);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
