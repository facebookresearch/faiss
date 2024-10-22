/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>

void pickEncoding(int& codes, int& dim) {
    std::vector<int> codeSizes{
            3, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 96};

    // Above 32 doesn't work with no precomputed codes
    std::vector<int> dimSizes{4, 8, 10, 12, 16, 20, 24, 28, 32};

    while (true) {
        codes = codeSizes[faiss::gpu::randVal(0, codeSizes.size() - 1)];
        dim = codes * dimSizes[faiss::gpu::randVal(0, dimSizes.size() - 1)];

        // for such a small test, super-low or high dim is more likely to
        // generate comparison errors
        if (dim < 256 && dim >= 64) {
            return;
        }
    }
}

void pickCuvsEncoding(int& codes, int& dim, int bitsPerCode) {
    // Above 32 doesn't work with no precomputed codes
    std::vector<int> dimSizes{4, 8, 10, 12, 16, 20, 24, 28, 32};

    while (true) {
        codes = faiss::gpu::randVal(0, 96);
        dim = codes * dimSizes[faiss::gpu::randVal(0, dimSizes.size() - 1)];

        // for such a small test, super-low or high dim is more likely to
        // generate comparison errors
        if (dim < 256 && dim >= 64 && (codes * bitsPerCode) % 8 == 0) {
            return;
        }
    }
}

struct Options {
    Options() {
        numAdd = faiss::gpu::randVal(2000, 5000);
        numCentroids = std::sqrt((float)numAdd);
        numTrain = numCentroids * 40;

        pickEncoding(codes, dim);

        // TODO: Change back to `faiss::gpu::randVal(3, 7)` when we
        // officially support non-multiple of 8 subcodes for IVFPQ.
        bitsPerCode = 8;

        nprobe = std::min(faiss::gpu::randVal(40, 1000), numCentroids);
        numQuery = faiss::gpu::randVal(4, 8);

        // Due to the approximate nature of the query and of floating point
        // differences between GPU and CPU, to stay within our error bounds,
        // only use a small k
        k = std::min(faiss::gpu::randVal(5, 20), numAdd / 40);
        usePrecomputed = faiss::gpu::randBool();
        indicesOpt = faiss::gpu::randSelect(
                {faiss::gpu::INDICES_CPU,
                 faiss::gpu::INDICES_32_BIT,
                 faiss::gpu::INDICES_64_BIT});
        if (codes > 48) {
            // large codes can only fit using float16
            useFloat16 = true;
        } else {
            useFloat16 = faiss::gpu::randBool();
        }

        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

        interleavedLayout = false;
        useCuvs = false;
    }

    std::string toString() const {
        std::stringstream str;
        str << "IVFPQ device " << device << " numVecs " << numAdd << " dim "
            << dim << " numCentroids " << numCentroids << " codes " << codes
            << " bitsPerCode " << bitsPerCode << " nprobe " << nprobe
            << " numQuery " << numQuery << " k " << k << " usePrecomputed "
            << usePrecomputed << " indicesOpt " << indicesOpt << " useFloat16 "
            << useFloat16;

        return str.str();
    }

    float getCompareEpsilon() const {
        return 0.035f;
    }

    float getPctMaxDiff1() const {
        return useFloat16 ? 0.30f : 0.10f;
    }

    float getPctMaxDiffN() const {
        return useFloat16 ? 0.05f : 0.02f;
    }

    int numAdd;
    int numCentroids;
    int numTrain;
    int codes;
    int dim;
    int bitsPerCode;
    int nprobe;
    int numQuery;
    int k;
    bool usePrecomputed;
    faiss::gpu::IndicesOptions indicesOpt;
    bool useFloat16;
    int device;
    bool interleavedLayout;
    bool useCuvs;
};

void queryTest(Options opt, faiss::MetricType metricType) {
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 coarseQuantizerL2(opt.dim);
    faiss::IndexFlatIP coarseQuantizerIP(opt.dim);
    faiss::Index* quantizer = metricType == faiss::METRIC_L2
            ? (faiss::Index*)&coarseQuantizerL2
            : (faiss::Index*)&coarseQuantizerIP;

    faiss::IndexIVFPQ cpuIndex(
            quantizer, opt.dim, opt.numCentroids, opt.codes, opt.bitsPerCode);
    cpuIndex.metric_type = metricType;
    cpuIndex.nprobe = opt.nprobe;
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());

    // Use the default temporary memory management to test the memory
    // manager
    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = opt.device;
    config.usePrecomputedTables = opt.usePrecomputed;
    config.indicesOptions = opt.indicesOpt;
    config.useFloat16LookupTables = opt.useFloat16;
    config.interleavedLayout = opt.interleavedLayout;
    config.use_cuvs = opt.useCuvs;

    faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
    gpuIndex.nprobe = opt.nprobe;

    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            opt.getCompareEpsilon(),
            opt.getPctMaxDiff1(),
            opt.getPctMaxDiffN());
}

TEST(TestGpuIndexIVFPQ, Query_L2) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        opt.usePrecomputed = (tries % 2 == 0);
        queryTest(opt, faiss::MetricType::METRIC_L2);
    }
}

TEST(TestGpuIndexIVFPQ, Query_IP) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        queryTest(opt, faiss::MetricType::METRIC_INNER_PRODUCT);
    }
}

// Large batch sizes (>= 65536) should also work
TEST(TestGpuIndexIVFPQ, LargeBatch) {
    for (bool usePrecomputed : {false, true}) {
        Options opt;

        // override for large sizes
        opt.dim = 4;
        opt.numQuery = 100000;
        opt.codes = 2;
        opt.usePrecomputed = usePrecomputed;
        opt.useFloat16 = false;

        queryTest(opt, faiss::MetricType::METRIC_L2);
    }
}

void testMMCodeDistance(faiss::MetricType mt) {
    // Explicitly test the code distance via batch matrix multiplication route
    // (even for dimension sizes that would otherwise be handled by the
    // specialized route (via enabling `useMMCodeDistance`)
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

        faiss::IndexFlat coarseQuantizer(opt.dim, mt);
        faiss::IndexIVFPQ cpuIndex(
                &coarseQuantizer,
                opt.dim,
                opt.numCentroids,
                opt.codes,
                opt.bitsPerCode);
        cpuIndex.nprobe = opt.nprobe;
        cpuIndex.train(opt.numTrain, trainVecs.data());
        cpuIndex.add(opt.numAdd, addVecs.data());

        // Use the default temporary memory management to test the memory
        // manager
        faiss::gpu::StandardGpuResources res;

        faiss::gpu::GpuIndexIVFPQConfig config;
        config.device = opt.device;
        config.usePrecomputedTables = false;
        config.useMMCodeDistance = true;
        config.indicesOptions = opt.indicesOpt;
        config.use_cuvs = false;

        // Make sure that the float16 version works as well
        config.useFloat16LookupTables = (tries % 2 == 0);
        config.flatConfig.useFloat16 = (tries % 2 == 1);

        faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
        gpuIndex.nprobe = opt.nprobe;

        faiss::gpu::compareIndices(
                cpuIndex,
                gpuIndex,
                opt.numQuery,
                opt.dim,
                opt.k,
                opt.toString(),
                opt.getCompareEpsilon(),
                opt.getPctMaxDiff1(),
                opt.getPctMaxDiffN());
    }

    // These sizes are not specialized, they will fall back to the MM version
    for (int dimPerSubQ : {7, 11}) {
        Options opt;

        opt.codes = 12;
        opt.dim = dimPerSubQ * opt.codes;

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

        faiss::IndexFlat coarseQuantizer(opt.dim, mt);
        faiss::IndexIVFPQ cpuIndex(
                &coarseQuantizer,
                opt.dim,
                opt.numCentroids,
                opt.codes,
                opt.bitsPerCode);
        cpuIndex.nprobe = opt.nprobe;
        cpuIndex.train(opt.numTrain, trainVecs.data());
        cpuIndex.add(opt.numAdd, addVecs.data());

        // Use the default temporary memory management to test the memory
        // manager
        faiss::gpu::StandardGpuResources res;

        faiss::gpu::GpuIndexIVFPQConfig config;
        config.device = opt.device;
        config.usePrecomputedTables = false;
        config.indicesOptions = opt.indicesOpt;
        config.use_cuvs = false;

        // Make sure that the float16 version works as well
        config.useFloat16LookupTables = (dimPerSubQ == 7);

        faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
        gpuIndex.nprobe = opt.nprobe;

        faiss::gpu::compareIndices(
                cpuIndex,
                gpuIndex,
                opt.numQuery,
                opt.dim,
                opt.k,
                opt.toString(),
                opt.getCompareEpsilon(),
                opt.getPctMaxDiff1(),
                opt.getPctMaxDiffN());
    }
}

TEST(TestGpuIndexIVFPQ, Query_L2_MMCodeDistance) {
    testMMCodeDistance(faiss::MetricType::METRIC_L2);
}

TEST(TestGpuIndexIVFPQ, Query_IP_MMCodeDistance) {
    testMMCodeDistance(faiss::MetricType::METRIC_INNER_PRODUCT);
}

TEST(TestGpuIndexIVFPQ, Float16Coarse) {
    Options opt;

    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 coarseQuantizer(opt.dim);
    faiss::IndexIVFPQ cpuIndex(
            &coarseQuantizer,
            opt.dim,
            opt.numCentroids,
            opt.codes,
            opt.bitsPerCode);
    cpuIndex.nprobe = opt.nprobe;
    cpuIndex.train(opt.numTrain, trainVecs.data());

    // Use the default temporary memory management to test the memory manager
    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = opt.device;
    config.flatConfig.useFloat16 = true;
    config.usePrecomputedTables = opt.usePrecomputed;
    config.indicesOptions = opt.indicesOpt;
    config.useFloat16LookupTables = opt.useFloat16;
    config.use_cuvs = false;

    faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
    gpuIndex.nprobe = opt.nprobe;

    gpuIndex.add(opt.numAdd, addVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());

    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            opt.getCompareEpsilon(),
            opt.getPctMaxDiff1(),
            opt.getPctMaxDiffN());
}

void addTest(Options opt, faiss::MetricType metricType) {
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 coarseQuantizerL2(opt.dim);
    faiss::IndexFlatIP coarseQuantizerIP(opt.dim);
    faiss::Index* quantizer = metricType == faiss::METRIC_L2
            ? (faiss::Index*)&coarseQuantizerL2
            : (faiss::Index*)&coarseQuantizerIP;

    faiss::IndexIVFPQ cpuIndex(
            quantizer, opt.dim, opt.numCentroids, opt.codes, opt.bitsPerCode);
    cpuIndex.nprobe = opt.nprobe;
    cpuIndex.metric_type = metricType;
    cpuIndex.train(opt.numTrain, trainVecs.data());

    // Use the default temporary memory management to test the memory
    // manager
    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = opt.device;
    config.usePrecomputedTables = opt.usePrecomputed;
    config.indicesOptions = opt.indicesOpt;
    config.useFloat16LookupTables = opt.useFloat16;
    config.interleavedLayout = opt.interleavedLayout;
    config.use_cuvs = opt.useCuvs;

    faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
    gpuIndex.nprobe = opt.nprobe;

    gpuIndex.add(opt.numAdd, addVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());

    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            opt.getCompareEpsilon(),
            opt.getPctMaxDiff1(),
            opt.getPctMaxDiffN());
}

TEST(TestGpuIndexIVFPQ, Add_L2) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        addTest(opt, faiss::METRIC_L2);
    }
}

TEST(TestGpuIndexIVFPQ, Add_IP) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        addTest(opt, faiss::METRIC_INNER_PRODUCT);
    }
}

void copyToTest(Options opt) {
    for (int tries = 0; tries < 2; ++tries) {
        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

        // Use the default temporary memory management to test the memory
        // manager
        faiss::gpu::StandardGpuResources res;

        faiss::gpu::GpuIndexIVFPQConfig config;
        config.device = opt.device;
        config.usePrecomputedTables = false;
        config.indicesOptions = opt.indicesOpt;
        config.useFloat16LookupTables = opt.useFloat16;
        config.interleavedLayout = opt.interleavedLayout;
        config.use_cuvs = opt.useCuvs;

        faiss::gpu::GpuIndexIVFPQ gpuIndex(
                &res,
                opt.dim,
                opt.numCentroids,
                opt.codes,
                opt.bitsPerCode,
                faiss::METRIC_L2,
                config);
        gpuIndex.nprobe = opt.nprobe;
        gpuIndex.train(opt.numTrain, trainVecs.data());
        gpuIndex.add(opt.numAdd, addVecs.data());

        // Use garbage values to see if we overwrite them
        faiss::IndexFlatL2 cpuQuantizer(1);
        faiss::IndexIVFPQ cpuIndex(&cpuQuantizer, 1, 1, 1, 1);

        gpuIndex.copyTo(&cpuIndex);

        EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
        EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);

        EXPECT_EQ(cpuIndex.d, gpuIndex.d);
        EXPECT_EQ(cpuIndex.d, opt.dim);
        EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
        EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);
        EXPECT_EQ(cpuIndex.pq.M, gpuIndex.getNumSubQuantizers());
        EXPECT_EQ(gpuIndex.getNumSubQuantizers(), opt.codes);
        EXPECT_EQ(cpuIndex.pq.nbits, gpuIndex.getBitsPerCode());
        EXPECT_EQ(gpuIndex.getBitsPerCode(), opt.bitsPerCode);

        testIVFEquality(cpuIndex, gpuIndex);

        // Query both objects; results should be equivalent
        faiss::gpu::compareIndices(
                cpuIndex,
                gpuIndex,
                opt.numQuery,
                opt.dim,
                opt.k,
                opt.toString(),
                opt.getCompareEpsilon(),
                opt.getPctMaxDiff1(),
                opt.getPctMaxDiffN());
    }
}

TEST(TestGpuIndexIVFPQ, CopyTo) {
    Options opt;
    copyToTest(opt);
}

void copyFromTest(Options opt) {
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 coarseQuantizer(opt.dim);
    faiss::IndexIVFPQ cpuIndex(
            &coarseQuantizer,
            opt.dim,
            opt.numCentroids,
            opt.codes,
            opt.bitsPerCode);
    cpuIndex.nprobe = opt.nprobe;
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());

    // Use the default temporary memory management to test the memory manager
    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = opt.device;
    config.usePrecomputedTables = opt.usePrecomputed;
    config.indicesOptions = opt.indicesOpt;
    config.useFloat16LookupTables = opt.useFloat16;
    config.interleavedLayout = opt.interleavedLayout;
    config.use_cuvs = opt.useCuvs;

    // Use garbage values to see if we overwrite them
    faiss::gpu::GpuIndexIVFPQ gpuIndex(
            &res, 1, 1, 1, 8, faiss::METRIC_L2, config);
    gpuIndex.nprobe = 1;

    gpuIndex.copyFrom(&cpuIndex);

    // Make sure we are equivalent
    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);

    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
    EXPECT_EQ(cpuIndex.d, opt.dim);
    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);
    EXPECT_EQ(cpuIndex.pq.M, gpuIndex.getNumSubQuantizers());
    EXPECT_EQ(gpuIndex.getNumSubQuantizers(), opt.codes);
    EXPECT_EQ(cpuIndex.pq.nbits, gpuIndex.getBitsPerCode());
    EXPECT_EQ(gpuIndex.getBitsPerCode(), opt.bitsPerCode);

    testIVFEquality(cpuIndex, gpuIndex);

    // Query both objects; results should be equivalent
    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            opt.getCompareEpsilon(),
            opt.getPctMaxDiff1(),
            opt.getPctMaxDiffN());
}

TEST(TestGpuIndexIVFPQ, CopyFrom) {
    Options opt;
    copyFromTest(opt);
}

void queryNaNTest(Options opt) {
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    // Use the default temporary memory management to test the memory manager
    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = opt.device;
    config.usePrecomputedTables = opt.usePrecomputed;
    config.indicesOptions = opt.indicesOpt;
    config.useFloat16LookupTables = opt.useFloat16;
    config.use_cuvs = opt.useCuvs;
    config.interleavedLayout = opt.useCuvs ? true : opt.interleavedLayout;

    faiss::gpu::GpuIndexIVFPQ gpuIndex(
            &res,
            opt.dim,
            opt.numCentroids,
            opt.codes,
            opt.bitsPerCode,
            faiss::METRIC_L2,
            config);

    gpuIndex.nprobe = opt.nprobe;

    gpuIndex.train(opt.numTrain, trainVecs.data());
    gpuIndex.add(opt.numAdd, addVecs.data());

    int numQuery = 5;
    std::vector<float> nans(
            numQuery * opt.dim, std::numeric_limits<float>::quiet_NaN());

    std::vector<float> distances(numQuery * opt.k, 0);
    std::vector<faiss::idx_t> indices(numQuery * opt.k, 0);

    gpuIndex.search(
            numQuery, nans.data(), opt.k, distances.data(), indices.data());

    for (int q = 0; q < numQuery; ++q) {
        for (int k = 0; k < opt.k; ++k) {
            EXPECT_EQ(indices[q * opt.k + k], -1);
            EXPECT_EQ(
                    distances[q * opt.k + k],
                    std::numeric_limits<float>::max());
        }
    }
}

TEST(TestGpuIndexIVFPQ, QueryNaN) {
    Options opt;
    opt.useCuvs = false;
    queryNaNTest(opt);
}

void addNaNTest(Options opt) {
    // Use the default temporary memory management to test the memory manager
    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = opt.device;
    config.usePrecomputedTables = opt.usePrecomputed;
    config.indicesOptions = opt.indicesOpt;
    config.useFloat16LookupTables = opt.useFloat16;
    config.interleavedLayout = opt.interleavedLayout;
    config.use_cuvs = opt.useCuvs;

    faiss::gpu::GpuIndexIVFPQ gpuIndex(
            &res,
            opt.dim,
            opt.numCentroids,
            opt.codes,
            opt.bitsPerCode,
            faiss::METRIC_L2,
            config);

    gpuIndex.nprobe = opt.nprobe;

    int numNans = 10;
    std::vector<float> nans(
            numNans * opt.dim, std::numeric_limits<float>::quiet_NaN());

    // Make one vector valid, which should actually add
    for (int i = 0; i < opt.dim; ++i) {
        nans[i] = 0.0f;
    }

    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    gpuIndex.train(opt.numTrain, trainVecs.data());

    // should not crash
    EXPECT_EQ(gpuIndex.ntotal, 0);
    gpuIndex.add(numNans, nans.data());

    std::vector<float> queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
    std::vector<float> distance(opt.numQuery * opt.k, 0);
    std::vector<faiss::idx_t> indices(opt.numQuery * opt.k, 0);

    // should not crash
    gpuIndex.search(
            opt.numQuery,
            queryVecs.data(),
            opt.k,
            distance.data(),
            indices.data());
}

TEST(TestGpuIndexIVFPQ, AddNaN) {
    Options opt;
    opt.useCuvs = false;
    addNaNTest(opt);
}

#if defined USE_NVIDIA_CUVS
TEST(TestGpuIndexIVFPQ, Query_L2_Cuvs) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        opt.bitsPerCode = faiss::gpu::randVal(4, 8);
        opt.useCuvs = true;
        opt.interleavedLayout = true;
        opt.usePrecomputed = false;
        opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
        pickCuvsEncoding(opt.codes, opt.dim, opt.bitsPerCode);
        queryTest(opt, faiss::MetricType::METRIC_L2);
    }
}

TEST(TestGpuIndexIVFPQ, Query_IP_Cuvs) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        opt.bitsPerCode = faiss::gpu::randVal(4, 8);
        opt.useCuvs = true;
        opt.interleavedLayout = true;
        opt.usePrecomputed = false;
        opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
        pickCuvsEncoding(opt.codes, opt.dim, opt.bitsPerCode);
        queryTest(opt, faiss::MetricType::METRIC_INNER_PRODUCT);
    }
}

// Large batch sizes (>= 65536) should also work
TEST(TestGpuIndexIVFPQ, LargeBatch_Cuvs) {
    Options opt;

    // override for large sizes
    opt.dim = 4;
    opt.numQuery = 100000;
    opt.codes = 2;
    opt.useCuvs = true;
    opt.interleavedLayout = true;
    opt.usePrecomputed = false;
    opt.useFloat16 = false;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    opt.bitsPerCode = 8;

    queryTest(opt, faiss::MetricType::METRIC_L2);
}

TEST(TestGpuIndexIVFPQ, CopyFrom_Cuvs) {
    Options opt;
    opt.useCuvs = true;
    opt.interleavedLayout = true;
    opt.bitsPerCode = faiss::gpu::randVal(4, 8);
    opt.usePrecomputed = false;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    pickCuvsEncoding(opt.codes, opt.dim, opt.bitsPerCode);
    copyFromTest(opt);
}

TEST(TestGpuIndexIVFPQ, Add_L2_Cuvs) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        opt.useCuvs = true;
        opt.interleavedLayout = true;
        opt.bitsPerCode = faiss::gpu::randVal(4, 8);
        opt.usePrecomputed = false;
        opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
        pickCuvsEncoding(opt.codes, opt.dim, opt.bitsPerCode);
        addTest(opt, faiss::METRIC_L2);
    }
}

TEST(TestGpuIndexIVFPQ, Add_IP_Cuvs) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        opt.useCuvs = true;
        opt.interleavedLayout = true;
        opt.bitsPerCode = faiss::gpu::randVal(4, 8);
        opt.usePrecomputed = false;
        opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
        pickCuvsEncoding(opt.codes, opt.dim, opt.bitsPerCode);
        addTest(opt, faiss::METRIC_INNER_PRODUCT);
    }
}

TEST(TestGpuIndexIVFPQ, QueryNaN_Cuvs) {
    Options opt;
    opt.useCuvs = true;
    opt.interleavedLayout = true;
    opt.bitsPerCode = faiss::gpu::randVal(4, 8);
    opt.usePrecomputed = false;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    pickCuvsEncoding(opt.codes, opt.dim, opt.bitsPerCode);
    queryNaNTest(opt);
}

TEST(TestGpuIndexIVFPQ, AddNaN_Cuvs) {
    Options opt;
    opt.useCuvs = true;
    opt.interleavedLayout = true;
    opt.bitsPerCode = faiss::gpu::randVal(4, 8);
    opt.usePrecomputed = false;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    pickCuvsEncoding(opt.codes, opt.dim, opt.bitsPerCode);
    addNaNTest(opt);
}

TEST(TestGpuIndexIVFPQ, CopyTo_Cuvs) {
    Options opt;
    opt.useCuvs = true;
    opt.interleavedLayout = true;
    opt.bitsPerCode = faiss::gpu::randVal(4, 8);
    opt.usePrecomputed = false;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    pickCuvsEncoding(opt.codes, opt.dim, opt.bitsPerCode);
    copyToTest(opt);
}
#endif

TEST(TestGpuIndexIVFPQ, UnifiedMemory) {
    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    if (!faiss::gpu::getFullUnifiedMemSupport(device)) {
        return;
    }

    int dim = 128;

    int numCentroids = 256;
    // Unfortunately it would take forever to add 24 GB in IVFPQ data,
    // so just perform a small test with data allocated in the unified
    // memory address space
    size_t numAdd = 10000;
    size_t numTrain = numCentroids * 40;
    int numQuery = 10;
    int k = 10;
    int nprobe = 8;
    int codes = 8;
    int bitsPerCode = 8;

    std::vector<float> trainVecs = faiss::gpu::randVecs(numTrain, dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(numAdd, dim);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFPQ cpuIndex(
            &quantizer, dim, numCentroids, codes, bitsPerCode);

    cpuIndex.train(numTrain, trainVecs.data());
    cpuIndex.add(numAdd, addVecs.data());
    cpuIndex.nprobe = nprobe;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = device;
    config.memorySpace = faiss::gpu::MemorySpace::Unified;
    config.use_cuvs = false;

    faiss::gpu::GpuIndexIVFPQ gpuIndex(
            &res,
            dim,
            numCentroids,
            codes,
            bitsPerCode,
            faiss::METRIC_L2,
            config);
    gpuIndex.copyFrom(&cpuIndex);
    gpuIndex.nprobe = nprobe;

    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            numQuery,
            dim,
            k,
            "Unified Memory",
            0.015f,
            0.1f,
            0.015f);

#if defined USE_NVIDIA_CUVS
    config.interleavedLayout = true;
    config.use_cuvs = true;
    config.indicesOptions = faiss::gpu::INDICES_64_BIT;

    faiss::gpu::GpuIndexIVFPQ cuvsGpuIndex(
            &res,
            dim,
            numCentroids,
            codes,
            bitsPerCode,
            faiss::METRIC_L2,
            config);
    cuvsGpuIndex.copyFrom(&cpuIndex);
    cuvsGpuIndex.nprobe = nprobe;

    faiss::gpu::compareIndices(
            cpuIndex,
            cuvsGpuIndex,
            numQuery,
            dim,
            k,
            "Unified Memory",
            0.015f,
            0.1f,
            0.015f);
#endif
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
