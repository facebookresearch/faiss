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
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>

// FIXME: figure out a better way to test fp16
constexpr float kF16MaxRelErr = 0.3f;
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

        useRaft = false;
    }

    std::string toString() const {
        std::stringstream str;
        str << "IVFFlat device " << device << " numVecs " << numAdd << " dim "
            << dim << " numCentroids " << numCentroids << " nprobe " << nprobe
            << " numQuery " << numQuery << " k " << k << " indicesOpt "
            << indicesOpt << " useRaft " << useRaft;

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
    bool useRaft;
};

void queryTest(
        Options opt,
        faiss::MetricType metricType,
        bool useFloat16CoarseQuantizer) {
    for (int tries = 0; tries < 2; ++tries) {
        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

        faiss::IndexFlatL2 quantizerL2(opt.dim);
        faiss::IndexFlatIP quantizerIP(opt.dim);
        faiss::Index* quantizer = metricType == faiss::METRIC_L2
                ? (faiss::Index*)&quantizerL2
                : (faiss::Index*)&quantizerIP;

        faiss::IndexIVFFlat cpuIndex(
                quantizer, opt.dim, opt.numCentroids, metricType);
        cpuIndex.train(opt.numTrain, trainVecs.data());
        cpuIndex.add(opt.numAdd, addVecs.data());
        cpuIndex.nprobe = opt.nprobe;

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = opt.device;
        config.indicesOptions = opt.indicesOpt;
        config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;
        config.use_raft = opt.useRaft;

        faiss::gpu::GpuIndexIVFFlat gpuIndex(
                &res, cpuIndex.d, cpuIndex.nlist, cpuIndex.metric_type, config);
        gpuIndex.copyFrom(&cpuIndex);
        gpuIndex.nprobe = opt.nprobe;

        bool compFloat16 = useFloat16CoarseQuantizer;
        faiss::gpu::compareIndices(
                cpuIndex,
                gpuIndex,
                opt.numQuery,
                opt.dim,
                opt.k,
                opt.toString(),
                compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
                // FIXME: the fp16 bounds are
                // useless when math (the accumulator) is
                // in fp16. Figure out another way to test
                compFloat16 ? 0.70f : 0.1f,
                compFloat16 ? 0.65f : 0.015f);
    }
}

void addTest(
        faiss::MetricType metricType,
        bool useFloat16CoarseQuantizer,
        bool useRaft) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

        faiss::IndexFlatL2 quantizerL2(opt.dim);
        faiss::IndexFlatIP quantizerIP(opt.dim);
        faiss::Index* quantizer = metricType == faiss::METRIC_L2
                ? (faiss::Index*)&quantizerL2
                : (faiss::Index*)&quantizerIP;

        faiss::IndexIVFFlat cpuIndex(
                quantizer, opt.dim, opt.numCentroids, metricType);
        cpuIndex.train(opt.numTrain, trainVecs.data());
        cpuIndex.nprobe = opt.nprobe;

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = opt.device;
        config.indicesOptions =
                useRaft ? faiss::gpu::INDICES_64_BIT : opt.indicesOpt;
        config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;
        config.use_raft = useRaft;

        faiss::gpu::GpuIndexIVFFlat gpuIndex(
                &res, cpuIndex.d, cpuIndex.nlist, cpuIndex.metric_type, config);
        gpuIndex.copyFrom(&cpuIndex);
        gpuIndex.nprobe = opt.nprobe;

        cpuIndex.add(opt.numAdd, addVecs.data());
        gpuIndex.add(opt.numAdd, addVecs.data());

        bool compFloat16 = useFloat16CoarseQuantizer;
        faiss::gpu::compareIndices(
                cpuIndex,
                gpuIndex,
                opt.numQuery,
                opt.dim,
                opt.k,
                opt.toString(),
                compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
                compFloat16 ? 0.70f : 0.1f,
                compFloat16 ? 0.30f : 0.015f);
    }
}

void copyToTest(bool useFloat16CoarseQuantizer, bool useRaft) {
    Options opt;
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = opt.device;
    config.indicesOptions =
            useRaft ? faiss::gpu::INDICES_64_BIT : opt.indicesOpt;
    config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;
    config.use_raft = useRaft;

    faiss::gpu::GpuIndexIVFFlat gpuIndex(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpuIndex.train(opt.numTrain, trainVecs.data());
    gpuIndex.add(opt.numAdd, addVecs.data());
    gpuIndex.nprobe = opt.nprobe;

    // use garbage values to see if we overwrite then
    faiss::IndexFlatL2 cpuQuantizer(1);
    faiss::IndexIVFFlat cpuIndex(&cpuQuantizer, 1, 1, faiss::METRIC_L2);
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
    bool compFloat16 = useFloat16CoarseQuantizer;
    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
            compFloat16 ? 0.70f : 0.1f,
            compFloat16 ? 0.30f : 0.015f);
}

void copyFromTest(bool useFloat16CoarseQuantizer, bool useRaft) {
    Options opt;
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 cpuQuantizer(opt.dim);
    faiss::IndexIVFFlat cpuIndex(
            &cpuQuantizer, opt.dim, opt.numCentroids, faiss::METRIC_L2);
    cpuIndex.nprobe = opt.nprobe;
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());

    // use garbage values to see if we overwrite then
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = opt.device;
    config.indicesOptions =
            useRaft ? faiss::gpu::INDICES_64_BIT : opt.indicesOpt;
    config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;
    config.use_raft = useRaft;

    faiss::gpu::GpuIndexIVFFlat gpuIndex(&res, 1, 1, faiss::METRIC_L2, config);
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
    bool compFloat16 = useFloat16CoarseQuantizer;
    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
            compFloat16 ? 0.70f : 0.1f,
            compFloat16 ? 0.30f : 0.015f);
}

TEST(TestGpuIndexIVFFlat, Float32_32_Add_L2) {
    addTest(faiss::METRIC_L2, false, false);

#if defined USE_NVIDIA_RAFT
    addTest(faiss::METRIC_L2, false, true);
#endif
}

TEST(TestGpuIndexIVFFlat, Float32_32_Add_IP) {
    addTest(faiss::METRIC_INNER_PRODUCT, false, false);

#if defined USE_NVIDIA_RAFT
    addTest(faiss::METRIC_INNER_PRODUCT, false, true);
#endif
}

TEST(TestGpuIndexIVFFlat, Float16_32_Add_L2) {
    addTest(faiss::METRIC_L2, true, false);

#if defined USE_NVIDIA_RAFT
    addTest(faiss::METRIC_L2, true, true);
#endif
}

TEST(TestGpuIndexIVFFlat, Float16_32_Add_IP) {
    addTest(faiss::METRIC_INNER_PRODUCT, true, false);

#if defined USE_NVIDIA_RAFT
    addTest(faiss::METRIC_INNER_PRODUCT, true, true);
#endif
}

//
// General query tests
//

TEST(TestGpuIndexIVFFlat, Float32_Query_L2) {
    Options opt;
    queryTest(opt, faiss::METRIC_L2, false);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_L2, false);
#endif
}

TEST(TestGpuIndexIVFFlat, Float32_Query_IP) {
    Options opt;
    queryTest(opt, faiss::METRIC_INNER_PRODUCT, false);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_INNER_PRODUCT, false);
#endif
}

TEST(TestGpuIndexIVFFlat, LargeBatch) {
    Options opt;
    opt.dim = 3;
    opt.numQuery = 100000;
    queryTest(opt, faiss::METRIC_L2, false);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_L2, false);
#endif
}

// float16 coarse quantizer

TEST(TestGpuIndexIVFFlat, Float16_32_Query_L2) {
    Options opt;
    queryTest(opt, faiss::METRIC_L2, true);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_L2, true);
#endif
}

TEST(TestGpuIndexIVFFlat, Float16_32_Query_IP) {
    Options opt;
    queryTest(opt, faiss::METRIC_INNER_PRODUCT, true);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_INNER_PRODUCT, true);
#endif
}

//
// There are IVF list scanning specializations for 64-d and 128-d that we
// make sure we explicitly test here
//

TEST(TestGpuIndexIVFFlat, Float32_Query_L2_64) {
    Options opt;
    opt.dim = 64;
    queryTest(opt, faiss::METRIC_L2, false);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_L2, false);
#endif
}

TEST(TestGpuIndexIVFFlat, Float32_Query_IP_64) {
    Options opt;
    opt.dim = 64;
    queryTest(opt, faiss::METRIC_INNER_PRODUCT, false);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_INNER_PRODUCT, false);
#endif
}

TEST(TestGpuIndexIVFFlat, Float32_Query_L2_128) {
    Options opt;
    opt.dim = 128;
    queryTest(opt, faiss::METRIC_L2, false);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_L2, false);
#endif
}

TEST(TestGpuIndexIVFFlat, Float32_Query_IP_128) {
    Options opt;
    opt.dim = 128;
    queryTest(opt, faiss::METRIC_INNER_PRODUCT, false);

#if defined USE_NVIDIA_RAFT
    opt.useRaft = true;
    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
    queryTest(opt, faiss::METRIC_INNER_PRODUCT, false);
#endif
}

//
// Copy tests
//

TEST(TestGpuIndexIVFFlat, Float32_32_CopyTo) {
    copyToTest(false, false);

#if defined USE_NVIDIA_RAFT
    copyToTest(false, true);
#endif
}

TEST(TestGpuIndexIVFFlat, Float32_32_CopyFrom) {
    copyFromTest(false, false);

#if defined USE_NVIDIA_RAFT
    copyFromTest(false, true);
#endif
}

TEST(TestGpuIndexIVFFlat, Float32_negative) {
    Options opt;

    auto trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    auto addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    // Put all vecs on negative side
    for (auto& f : trainVecs) {
        f = std::abs(f) * -1.0f;
    }

    for (auto& f : addVecs) {
        f *= std::abs(f) * -1.0f;
    }

    faiss::IndexFlatIP quantizerIP(opt.dim);
    faiss::Index* quantizer = (faiss::Index*)&quantizerIP;

    faiss::IndexIVFFlat cpuIndex(
            quantizer, opt.dim, opt.numCentroids, faiss::METRIC_INNER_PRODUCT);
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());
    cpuIndex.nprobe = opt.nprobe;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    // Construct a positive test set
    auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);

    // Put all vecs on positive size
    for (auto& f : queryVecs) {
        f = std::abs(f);
    }

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = opt.device;
    config.indicesOptions = opt.indicesOpt;
    config.use_raft = false;

    faiss::gpu::GpuIndexIVFFlat gpuIndex(
            &res, cpuIndex.d, cpuIndex.nlist, cpuIndex.metric_type, config);
    gpuIndex.copyFrom(&cpuIndex);
    gpuIndex.nprobe = opt.nprobe;

    bool compFloat16 = false;
    faiss::gpu::compareIndices(
            queryVecs,
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
            // FIXME: the fp16 bounds are
            // useless when math (the accumulator) is
            // in fp16. Figure out another way to test
            compFloat16 ? 0.99f : 0.1f,
            compFloat16 ? 0.65f : 0.015f);

#if defined USE_NVIDIA_RAFT
    config.use_raft = true;
    config.indicesOptions = faiss::gpu::INDICES_64_BIT;

    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
            &res, cpuIndex.d, cpuIndex.nlist, cpuIndex.metric_type, config);
    raftGpuIndex.copyFrom(&cpuIndex);
    raftGpuIndex.nprobe = opt.nprobe;

    faiss::gpu::compareIndices(
            queryVecs,
            cpuIndex,
            raftGpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
            // FIXME: the fp16 bounds are
            // useless when math (the accumulator) is
            // in fp16. Figure out another way to test
            compFloat16 ? 0.99f : 0.1f,
            compFloat16 ? 0.65f : 0.015f);
#endif
}

//
// NaN tests
//

TEST(TestGpuIndexIVFFlat, QueryNaN) {
    Options opt;

    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    int numQuery = 10;
    std::vector<float> nans(
            numQuery * opt.dim, std::numeric_limits<float>::quiet_NaN());

    std::vector<float> distances(numQuery * opt.k, 0);
    std::vector<faiss::idx_t> indices(numQuery * opt.k, 0);

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = opt.device;
    config.indicesOptions = opt.indicesOpt;
    config.flatConfig.useFloat16 = faiss::gpu::randBool();
    config.use_raft = false;

    faiss::gpu::GpuIndexIVFFlat gpuIndex(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpuIndex.nprobe = opt.nprobe;

    gpuIndex.train(opt.numTrain, trainVecs.data());
    gpuIndex.add(opt.numAdd, addVecs.data());

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

#if defined USE_NVIDIA_RAFT
    config.use_raft = true;
    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
    std::fill(distances.begin(), distances.end(), 0);
    std::fill(indices.begin(), indices.end(), 0);
    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    raftGpuIndex.nprobe = opt.nprobe;

    raftGpuIndex.train(opt.numTrain, trainVecs.data());
    raftGpuIndex.add(opt.numAdd, addVecs.data());

    raftGpuIndex.search(
            numQuery, nans.data(), opt.k, distances.data(), indices.data());

    for (int q = 0; q < numQuery; ++q) {
        for (int k = 0; k < opt.k; ++k) {
            EXPECT_EQ(indices[q * opt.k + k], -1);
            EXPECT_EQ(
                    distances[q * opt.k + k],
                    std::numeric_limits<float>::max());
        }
    }
#endif
}

TEST(TestGpuIndexIVFFlat, AddNaN) {
    Options opt;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    int numNans = 10;
    std::vector<float> nans(
            numNans * opt.dim, std::numeric_limits<float>::quiet_NaN());

    // Make one vector valid (not the first vector, in order to test offset
    // issues), which should actually add
    for (int i = 0; i < opt.dim; ++i) {
        nans[opt.dim + i] = i;
    }

    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = opt.device;
    config.indicesOptions = opt.indicesOpt;
    config.flatConfig.useFloat16 = faiss::gpu::randBool();
    config.use_raft = false;
    faiss::gpu::GpuIndexIVFFlat gpuIndex(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpuIndex.nprobe = opt.nprobe;
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

#if defined USE_NVIDIA_RAFT
    config.use_raft = true;
    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    raftGpuIndex.nprobe = opt.nprobe;
    raftGpuIndex.train(opt.numTrain, trainVecs.data());

    // should not crash
    EXPECT_EQ(raftGpuIndex.ntotal, 0);
    raftGpuIndex.add(numNans, nans.data());

    // should not crash
    raftGpuIndex.search(
            opt.numQuery,
            queryVecs.data(),
            opt.k,
            distance.data(),
            indices.data());
#endif
}

TEST(TestGpuIndexIVFFlat, UnifiedMemory) {
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

    std::vector<float> trainVecs = faiss::gpu::randVecs(numTrain, dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(numAdd, dim);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFFlat cpuIndex(
            &quantizer, dim, numCentroids, faiss::METRIC_L2);

    cpuIndex.train(numTrain, trainVecs.data());
    cpuIndex.add(numAdd, addVecs.data());
    cpuIndex.nprobe = nprobe;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = device;
    config.memorySpace = faiss::gpu::MemorySpace::Unified;
    config.use_raft = false;

    faiss::gpu::GpuIndexIVFFlat gpuIndex(
            &res, dim, numCentroids, faiss::METRIC_L2, config);
    gpuIndex.copyFrom(&cpuIndex);
    gpuIndex.nprobe = nprobe;

    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            numQuery,
            dim,
            k,
            "Unified Memory",
            kF32MaxRelErr,
            0.1f,
            0.015f);

#if defined USE_NVIDIA_RAFT
    config.use_raft = true;
    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
            &res, dim, numCentroids, faiss::METRIC_L2, config);
    raftGpuIndex.copyFrom(&cpuIndex);
    raftGpuIndex.nprobe = nprobe;

    faiss::gpu::compareIndices(
            cpuIndex,
            raftGpuIndex,
            numQuery,
            dim,
            k,
            "Unified Memory",
            kF32MaxRelErr,
            0.1f,
            0.015f);
#endif
}

TEST(TestGpuIndexIVFFlat, LongIVFList) {
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

    // Skip this device if we do not have sufficient memory
    constexpr size_t kMem = size_t(24) * 1024 * 1024 * 1024;

    if (faiss::gpu::getFreeMemory(device) < kMem) {
        std::cout << "TestGpuIndexIVFFlat.LongIVFList: skipping due "
                     "to insufficient device memory\n";
        return;
    }

    std::cout << "Running LongIVFList test\n";

    // Test functionality where a single IVF list has more than 2B code values
    int dim = 64;

    int numCentroids = 1;
    size_t numAdd = (size_t(1024) * 1024 * 1024 * 2 + 100000) / dim;
    size_t numTrain = 100;
    int numQuery = 5;
    int k = 10;

    std::vector<float> trainVecs = faiss::gpu::randVecs(numTrain, dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(numAdd, dim);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFFlat cpuIndex(
            &quantizer, dim, numCentroids, faiss::METRIC_L2);

    cpuIndex.train(numTrain, trainVecs.data());
    cpuIndex.add(numAdd, addVecs.data());
    cpuIndex.nprobe = 1;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = device;
    config.use_raft = false;

    faiss::gpu::GpuIndexIVFFlat gpuIndex(
            &res, dim, numCentroids, faiss::METRIC_L2, config);
    gpuIndex.train(numTrain, trainVecs.data());
    gpuIndex.add(numAdd, addVecs.data());
    gpuIndex.nprobe = 1;

    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            numQuery,
            dim,
            k,
            "Unified Memory",
            kF32MaxRelErr,
            0.1f,
            0.015f);

#if defined USE_NVIDIA_RAFT
    config.use_raft = true;
    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
            &res, dim, numCentroids, faiss::METRIC_L2, config);
    raftGpuIndex.train(numTrain, trainVecs.data());
    raftGpuIndex.add(numAdd, addVecs.data());
    raftGpuIndex.nprobe = 1;

    faiss::gpu::compareIndices(
            cpuIndex,
            raftGpuIndex,
            numQuery,
            dim,
            k,
            "Unified Memory",
            kF32MaxRelErr,
            0.1f,
            0.015f);
#endif
}

TEST(TestGpuIndexIVFFlat, Reconstruct_n) {
    Options opt;

    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 cpuQuantizer(opt.dim);
    faiss::IndexIVFFlat cpuIndex(
            &cpuQuantizer, opt.dim, opt.numCentroids, faiss::METRIC_L2);
    cpuIndex.nprobe = opt.nprobe;
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = opt.device;
    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
    config.use_raft = false;

    faiss::gpu::GpuIndexIVFFlat gpuIndex(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpuIndex.nprobe = opt.nprobe;

    gpuIndex.train(opt.numTrain, trainVecs.data());
    gpuIndex.add(opt.numAdd, addVecs.data());

    std::vector<float> gpuVals(opt.numAdd * opt.dim);

    gpuIndex.reconstruct_n(0, gpuIndex.ntotal, gpuVals.data());

    std::vector<float> cpuVals(opt.numAdd * opt.dim);

    cpuIndex.reconstruct_n(0, cpuIndex.ntotal, cpuVals.data());

    EXPECT_EQ(gpuVals, cpuVals);

    config.indicesOptions = faiss::gpu::INDICES_32_BIT;

    faiss::gpu::GpuIndexIVFFlat gpuIndex1(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpuIndex1.nprobe = opt.nprobe;

    gpuIndex1.train(opt.numTrain, trainVecs.data());
    gpuIndex1.add(opt.numAdd, addVecs.data());

    gpuIndex1.reconstruct_n(0, gpuIndex1.ntotal, gpuVals.data());

    EXPECT_EQ(gpuVals, cpuVals);

    config.indicesOptions = faiss::gpu::INDICES_CPU;

    faiss::gpu::GpuIndexIVFFlat gpuIndex2(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpuIndex2.nprobe = opt.nprobe;

    gpuIndex2.train(opt.numTrain, trainVecs.data());
    gpuIndex2.add(opt.numAdd, addVecs.data());

    gpuIndex2.reconstruct_n(0, gpuIndex2.ntotal, gpuVals.data());

    EXPECT_EQ(gpuVals, cpuVals);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
