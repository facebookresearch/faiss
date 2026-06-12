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
#include <faiss/impl/IDSelector.h>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
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
    config.use_cuvs = false;

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
    config.use_cuvs = false;

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

#if defined USE_NVIDIA_CUVS
void runCuvsSQ8Test(faiss::MetricType metricType) {
    using namespace faiss;
    using namespace faiss::gpu;

    Options opt;
    opt.indicesOpt = INDICES_64_BIT;

    std::vector<float> trainVecs = randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = randVecs(opt.numAdd, opt.dim);
    std::vector<float> queryVecs = randVecs(opt.numQuery, opt.dim);

    StandardGpuResources res;
    res.noTempMemory();

    auto config = GpuIndexIVFScalarQuantizerConfig();
    config.device = opt.device;
    config.indicesOptions = INDICES_64_BIT;
    config.use_cuvs = true;
    if (!should_use_cuvs(config)) {
        GTEST_SKIP() << "cuVS is not enabled for this GPU";
    }

    GpuIndexIVFScalarQuantizer gpuIndex(
            &res,
            opt.dim,
            opt.numCentroids,
            ScalarQuantizer::QT_8bit,
            metricType,
            true,
            config);
    gpuIndex.train(opt.numTrain, trainVecs.data());
    gpuIndex.add(opt.numAdd, addVecs.data());
    gpuIndex.nprobe = opt.nprobe;

    std::vector<float> distances(opt.numQuery * opt.k);
    std::vector<idx_t> labels(opt.numQuery * opt.k);
    gpuIndex.search(
            opt.numQuery,
            queryVecs.data(),
            opt.k,
            distances.data(),
            labels.data());
    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);
    EXPECT_NE(labels[0], idx_t(-1));

    IndexFlatL2 quantizerL2(opt.dim);
    IndexFlatIP quantizerIP(opt.dim);
    Index* quantizer = metricType == METRIC_L2 ? (Index*)&quantizerL2
                                               : (Index*)&quantizerIP;

    IndexIVFScalarQuantizer cpuCopy(
            quantizer,
            opt.dim,
            opt.numCentroids,
            ScalarQuantizer::QT_8bit,
            metricType,
            true);
    gpuIndex.copyTo(&cpuCopy);
    cpuCopy.nprobe = opt.nprobe;

    EXPECT_EQ(cpuCopy.ntotal, gpuIndex.ntotal);
    EXPECT_EQ(cpuCopy.d, gpuIndex.d);
    EXPECT_EQ(cpuCopy.nlist, gpuIndex.getNumLists());
    EXPECT_EQ(cpuCopy.sq.qtype, ScalarQuantizer::QT_8bit);
    EXPECT_EQ(cpuCopy.sq.trained.size(), 2 * opt.dim);
    EXPECT_TRUE(cpuCopy.by_residual);

    compareIndices(
            cpuCopy,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            0.08f,
            0.2f,
            0.05f);

    GpuIndexIVFScalarQuantizer copiedGpuIndex(
            &res, 1, 1, ScalarQuantizer::QT_4bit, METRIC_L2, false, config);
    copiedGpuIndex.copyFrom(&cpuCopy);
    copiedGpuIndex.nprobe = opt.nprobe;
    EXPECT_EQ(copiedGpuIndex.ntotal, cpuCopy.ntotal);
    testIVFEquality(cpuCopy, copiedGpuIndex);

    compareIndices(
            cpuCopy,
            copiedGpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            0.08f,
            0.2f,
            0.05f);

    SearchParametersIVF searchParams;
    searchParams.nprobe = opt.nprobe;
    TestIDSelectorStruct selectorStruct(opt.numAdd);
    for (auto& [selectorName, selector] : selectorStruct.selector_map) {
        searchParams.sel = selector.get();
        testIDSelectorSearch(
                &gpuIndex,
                &searchParams,
                queryVecs,
                opt.numQuery,
                opt.k,
                selectorName);
    }

    GpuIndexIVFScalarQuantizer shiftedGpuIndex(
            &res,
            opt.dim,
            opt.numCentroids,
            ScalarQuantizer::QT_8bit,
            metricType,
            true,
            config);
    shiftedGpuIndex.train(opt.numTrain, trainVecs.data());
    std::vector<idx_t> shiftedIds(opt.numAdd);
    idx_t shiftedIdBase = opt.numAdd * 4 + 123;
    for (int i = 0; i < opt.numAdd; ++i) {
        shiftedIds[i] = shiftedIdBase + i;
    }
    shiftedGpuIndex.add_with_ids(opt.numAdd, addVecs.data(), shiftedIds.data());
    shiftedGpuIndex.nprobe = opt.nprobe;

    SearchParametersIVF shiftedSearchParams;
    shiftedSearchParams.nprobe = opt.nprobe;
    IDSelectorRange shiftedSelector(shiftedIdBase, shiftedIdBase + opt.numAdd);
    shiftedSearchParams.sel = &shiftedSelector;
    std::vector<float> shiftedDistances(opt.numQuery * opt.k);
    std::vector<idx_t> shiftedLabels(opt.numQuery * opt.k, -1);
    shiftedGpuIndex.search(
            opt.numQuery,
            queryVecs.data(),
            opt.k,
            shiftedDistances.data(),
            shiftedLabels.data(),
            &shiftedSearchParams);

    bool foundShiftedLabel = false;
    for (idx_t label : shiftedLabels) {
        if (label >= 0) {
            foundShiftedLabel = true;
            EXPECT_TRUE(shiftedSelector.is_member(label));
        }
    }
    EXPECT_TRUE(foundShiftedLabel);

    auto nanQueries = queryVecs;
    nanQueries[opt.dim + 3] = std::numeric_limits<float>::quiet_NaN();
    gpuIndex.search(
            2, nanQueries.data(), opt.k, distances.data(), labels.data());
    for (int j = 0; j < opt.k; ++j) {
        EXPECT_EQ(labels[opt.k + j], idx_t(-1));
        EXPECT_EQ(distances[opt.k + j], std::numeric_limits<float>::max());
    }

    gpuIndex.reset();
    EXPECT_EQ(gpuIndex.ntotal, 0);
    for (int i = 0; i < opt.numCentroids; ++i) {
        EXPECT_EQ(gpuIndex.getListLength(i), 0);
    }
}

TEST(TestCuvsGpuIndexIVFScalarQuantizer, SQ8_L2) {
    runCuvsSQ8Test(faiss::METRIC_L2);
}

TEST(TestCuvsGpuIndexIVFScalarQuantizer, SQ8_IP) {
    runCuvsSQ8Test(faiss::METRIC_INNER_PRODUCT);
}
#endif

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
