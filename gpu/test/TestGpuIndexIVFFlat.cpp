
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "../../IndexFlat.h"
#include "../../IndexIVF.h"
#include "../GpuIndexIVFFlat.h"
#include "../StandardGpuResources.h"
#include "../utils/DeviceUtils.h"
#include "../test/TestUtils.h"
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

// FIXME: figure out a better way to test fp16
constexpr float kF16MaxRelErr = 0.3f;
constexpr float kF32MaxRelErr = 0.03f;


struct Options {
  Options() {
    numAdd = faiss::gpu::randVal(4000, 20000);
    dim = faiss::gpu::randVal(64, 200);

    numCentroids = std::sqrt((float) numAdd);
    numTrain = numCentroids * 40;
    nprobe = faiss::gpu::randVal(10, numCentroids);
    numQuery = faiss::gpu::randVal(32, 100);
    k = std::min(faiss::gpu::randVal(10, 30), numAdd / 40);
    indicesOpt = faiss::gpu::randSelect({
        faiss::gpu::INDICES_CPU,
          faiss::gpu::INDICES_32_BIT,
          faiss::gpu::INDICES_64_BIT});

    device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
  }

  std::string toString() const {
    std::stringstream str;
    str << "IVFFlat device " << device
        << " numVecs " << numAdd
        << " dim " << dim
        << " numCentroids " << numCentroids
        << " nprobe " << nprobe
        << " numQuery " << numQuery
        << " k " << k
        << " indicesOpt " << indicesOpt;

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

void queryTest(faiss::MetricType metricType,
               bool useFloat16CoarseQuantizer,
               bool useFloat16,
               int dimOverride = -1) {
  for (int tries = 0; tries < 3; ++tries) {
    faiss::gpu::newTestSeed();

    Options opt;
    opt.dim = dimOverride != -1 ? dimOverride : opt.dim;

    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 quantizerL2(opt.dim);
    faiss::IndexFlatIP quantizerIP(opt.dim);
    faiss::Index* quantizer =
      metricType == faiss::METRIC_L2 ?
      (faiss::Index*) &quantizerL2 : (faiss::Index*) &quantizerIP;

    faiss::IndexIVFFlat cpuIndex(quantizer,
                                 opt.dim, opt.numCentroids, metricType);
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());
    cpuIndex.nprobe = opt.nprobe;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlat gpuIndex(&res,
                                         opt.device,
                                         useFloat16CoarseQuantizer,
                                         useFloat16,
                                         cpuIndex.d,
                                         cpuIndex.nlist,
                                         opt.indicesOpt,
                                         cpuIndex.metric_type);
    gpuIndex.copyFrom(&cpuIndex);
    gpuIndex.setNumProbes(opt.nprobe);

    bool compFloat16 = useFloat16CoarseQuantizer || useFloat16;
    faiss::gpu::compareIndices(cpuIndex, gpuIndex,
                               opt.numQuery, opt.dim, opt.k, opt.toString(),
                               compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
                               // FIXME: the fp16 bounds are
                               // useless when math (the accumulator) is
                               // in fp16. Figure out another way to test
                               compFloat16 ? 0.99f : 0.1f,
                               compFloat16 ? 0.65f : 0.015f);
  }
}

void addTest(faiss::MetricType metricType,
             bool useFloat16CoarseQuantizer,
             bool useFloat16) {
  for (int tries = 0; tries < 5; ++tries) {
    faiss::gpu::newTestSeed();

    Options opt;

    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 quantizerL2(opt.dim);
    faiss::IndexFlatIP quantizerIP(opt.dim);
    faiss::Index* quantizer =
      metricType == faiss::METRIC_L2 ?
      (faiss::Index*) &quantizerL2 : (faiss::Index*) &quantizerIP;

    faiss::IndexIVFFlat cpuIndex(quantizer,
                                 opt.dim,
                                 opt.numCentroids,
                                 metricType);
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.nprobe = opt.nprobe;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlat gpuIndex(&res,
                                         opt.device,
                                         useFloat16CoarseQuantizer,
                                         useFloat16,
                                         cpuIndex.d,
                                         cpuIndex.nlist,
                                         opt.indicesOpt,
                                         cpuIndex.metric_type);
    gpuIndex.copyFrom(&cpuIndex);
    gpuIndex.setNumProbes(opt.nprobe);

    cpuIndex.add(opt.numAdd, addVecs.data());
    gpuIndex.add(opt.numAdd, addVecs.data());

    bool compFloat16 = useFloat16CoarseQuantizer || useFloat16;
    faiss::gpu::compareIndices(cpuIndex, gpuIndex,
                               opt.numQuery, opt.dim, opt.k, opt.toString(),
                               compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
                               compFloat16 ? 0.70f : 0.1f,
                               compFloat16 ? 0.30f : 0.015f);
  }
}

void copyToTest(bool useFloat16CoarseQuantizer,
                bool useFloat16) {
  faiss::gpu::newTestSeed();

  Options opt;
  std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
  std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

  faiss::gpu::StandardGpuResources res;
  res.noTempMemory();

  faiss::gpu::GpuIndexIVFFlat gpuIndex(&res,
                                       opt.device,
                                       useFloat16CoarseQuantizer,
                                       useFloat16,
                                       opt.dim,
                                       opt.numCentroids,
                                       opt.indicesOpt,
                                       faiss::METRIC_L2);
  gpuIndex.train(opt.numTrain, trainVecs.data());
  gpuIndex.add(opt.numAdd, addVecs.data());
  gpuIndex.setNumProbes(opt.nprobe);

  // use garbage values to see if we overwrite then
  faiss::IndexFlatL2 cpuQuantizer(1);
  faiss::IndexIVFFlat cpuIndex(&cpuQuantizer, 1, 1, faiss::METRIC_L2);
  cpuIndex.nprobe = 1;

  gpuIndex.copyTo(&cpuIndex);

  EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
  EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);

  EXPECT_EQ(cpuIndex.d, gpuIndex.d);
  EXPECT_EQ(cpuIndex.d, opt.dim);
  EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
  EXPECT_EQ(cpuIndex.nprobe, gpuIndex.getNumProbes());

  // Query both objects; results should be equivalent
  bool compFloat16 = useFloat16CoarseQuantizer || useFloat16;
  faiss::gpu::compareIndices(cpuIndex, gpuIndex,
                             opt.numQuery, opt.dim, opt.k, opt.toString(),
                             compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
                             compFloat16 ? 0.70f : 0.1f,
                             compFloat16 ? 0.30f : 0.015f);
}

void copyFromTest(bool useFloat16CoarseQuantizer,
                  bool useFloat16) {
  faiss::gpu::newTestSeed();

  Options opt;
  std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
  std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

  faiss::IndexFlatL2 cpuQuantizer(opt.dim);
  faiss::IndexIVFFlat cpuIndex(&cpuQuantizer,
                               opt.dim,
                               opt.numCentroids,
                               faiss::METRIC_L2);
  cpuIndex.nprobe = opt.nprobe;
  cpuIndex.train(opt.numTrain, trainVecs.data());
  cpuIndex.add(opt.numAdd, addVecs.data());

  // use garbage values to see if we overwrite then
  faiss::gpu::StandardGpuResources res;
  res.noTempMemory();

  faiss::gpu::GpuIndexIVFFlat gpuIndex(&res,
                                       opt.device,
                                       useFloat16CoarseQuantizer,
                                       useFloat16,
                                       1,
                                       1,
                                       opt.indicesOpt,
                                       faiss::METRIC_L2);
  gpuIndex.setNumProbes(1);

  gpuIndex.copyFrom(&cpuIndex);

  EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
  EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);

  EXPECT_EQ(cpuIndex.d, gpuIndex.d);
  EXPECT_EQ(cpuIndex.d, opt.dim);
  EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
  EXPECT_EQ(cpuIndex.nprobe, gpuIndex.getNumProbes());

  // Query both objects; results should be equivalent
  bool compFloat16 = useFloat16CoarseQuantizer || useFloat16;
  faiss::gpu::compareIndices(cpuIndex, gpuIndex,
                             opt.numQuery, opt.dim, opt.k, opt.toString(),
                             compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
                             compFloat16 ? 0.70f : 0.1f,
                             compFloat16 ? 0.30f : 0.015f);
}

TEST(TestGpuIndexIVFFlat, Float32_32_Add_L2) {
  addTest(faiss::METRIC_L2, false, false);
}

TEST(TestGpuIndexIVFFlat, Float32_32_Add_IP) {
  addTest(faiss::METRIC_INNER_PRODUCT, false, false);
}

TEST(TestGpuIndexIVFFlat, Float32_16_Add_L2) {
  addTest(faiss::METRIC_L2, false, true);
}

TEST(TestGpuIndexIVFFlat, Float32_16_Add_IP) {
  addTest(faiss::METRIC_INNER_PRODUCT, false, true);
}

TEST(TestGpuIndexIVFFlat, Float16_32_Add_L2) {
  addTest(faiss::METRIC_L2, true, false);
}

TEST(TestGpuIndexIVFFlat, Float16_32_Add_IP) {
  addTest(faiss::METRIC_INNER_PRODUCT, true, false);
}

//
// General query tests
//

TEST(TestGpuIndexIVFFlat, Float32_Query_L2) {
  queryTest(faiss::METRIC_L2, false, false);
}

TEST(TestGpuIndexIVFFlat, Float32_Query_IP) {
  queryTest(faiss::METRIC_INNER_PRODUCT, false, false);
}

TEST(TestGpuIndexIVFFlat, Float16_Query_L2) {
  queryTest(faiss::METRIC_L2, false, true);
}

TEST(TestGpuIndexIVFFlat, Float16_Query_IP) {
  queryTest(faiss::METRIC_INNER_PRODUCT, false, true);
}

// float16 coarse quantizer

TEST(TestGpuIndexIVFFlat, Float16_32_Query_L2) {
  queryTest(faiss::METRIC_L2, true, false);
}

TEST(TestGpuIndexIVFFlat, Float16_32_Query_IP) {
  queryTest(faiss::METRIC_INNER_PRODUCT, true, false);
}

//
// There are IVF list scanning specializations for 64-d and 128-d that we
// make sure we explicitly test here
//

TEST(TestGpuIndexIVFFlat, Float32_Query_L2_64) {
  queryTest(faiss::METRIC_L2, false, false, 64);
}

TEST(TestGpuIndexIVFFlat, Float32_Query_IP_64) {
  queryTest(faiss::METRIC_INNER_PRODUCT, false, false, 64);
}

TEST(TestGpuIndexIVFFlat, Float16_Query_L2_64) {
  queryTest(faiss::METRIC_L2, false, true, 64);
}

TEST(TestGpuIndexIVFFlat, Float16_Query_IP_64) {
  queryTest(faiss::METRIC_INNER_PRODUCT, false, true, 64);
}

TEST(TestGpuIndexIVFFlat, Float32_Query_L2_128) {
  queryTest(faiss::METRIC_L2, false, false, 128);
}

TEST(TestGpuIndexIVFFlat, Float32_Query_IP_128) {
  queryTest(faiss::METRIC_INNER_PRODUCT, false, false, 128);
}

TEST(TestGpuIndexIVFFlat, Float16_Query_L2_128) {
  queryTest(faiss::METRIC_L2, false, true, 128);
}

TEST(TestGpuIndexIVFFlat, Float16_Query_IP_128) {
  queryTest(faiss::METRIC_INNER_PRODUCT, false, true, 128);
}

// For 256-d, only float16 is specialized

TEST(TestGpuIndexIVFFlat, Float16_Query_L2_256) {
  queryTest(faiss::METRIC_L2, false, true, 256);
}

TEST(TestGpuIndexIVFFlat, Float16_Query_IP_256) {
  queryTest(faiss::METRIC_INNER_PRODUCT, false, true, 256);
}

//
// Copy tests
//

TEST(TestGpuIndexIVFFlat, Float32_16_CopyTo) {
  copyToTest(false, true);
}

TEST(TestGpuIndexIVFFlat, Float32_32_CopyTo) {
  copyToTest(false, false);
}

//
// NaN tests
//

TEST(TestGpuIndexIVFFlat, QueryNaN) {
  faiss::gpu::newTestSeed();

  Options opt;

  std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
  std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

  faiss::gpu::StandardGpuResources res;
  res.noTempMemory();

  faiss::gpu::GpuIndexIVFFlat gpuIndex(&res,
                                       opt.device,
                                       faiss::gpu::randBool(),
                                       faiss::gpu::randBool(),
                                       opt.dim,
                                       opt.numCentroids,
                                       opt.indicesOpt,
                                       faiss::METRIC_L2);
  gpuIndex.setNumProbes(opt.nprobe);

  gpuIndex.train(opt.numTrain, trainVecs.data());
  gpuIndex.add(opt.numAdd, addVecs.data());

  int numQuery = 10;
  std::vector<float> nans(numQuery * opt.dim,
                          std::numeric_limits<float>::quiet_NaN());

  std::vector<float> distances(numQuery * opt.k, 0);
  std::vector<faiss::Index::idx_t> indices(numQuery * opt.k, 0);

  gpuIndex.search(numQuery,
                  nans.data(),
                  opt.k,
                  distances.data(),
                  indices.data());

  for (int q = 0; q < numQuery; ++q) {
    for (int k = 0; k < opt.k; ++k) {
      EXPECT_EQ(indices[q * opt.k + k], -1);
      EXPECT_EQ(distances[q * opt.k + k], std::numeric_limits<float>::max());
    }
  }
}

TEST(TestGpuIndexIVFFlat, AddNaN) {
  faiss::gpu::newTestSeed();

  Options opt;

  faiss::gpu::StandardGpuResources res;
  res.noTempMemory();

  faiss::gpu::GpuIndexIVFFlat gpuIndex(&res,
                                       opt.device,
                                       faiss::gpu::randBool(),
                                       faiss::gpu::randBool(),
                                       opt.dim,
                                       opt.numCentroids,
                                       opt.indicesOpt,
                                       faiss::METRIC_L2);
  gpuIndex.setNumProbes(opt.nprobe);

  int numNans = 10;
  std::vector<float> nans(numNans * opt.dim,
                          std::numeric_limits<float>::quiet_NaN());

  // Make one vector valid, which should actually add
  for (int i = 0; i < opt.dim; ++i) {
    nans[i] = 0.0f;
  }

  std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
  gpuIndex.train(opt.numTrain, trainVecs.data());

  // should not crash
  EXPECT_EQ(gpuIndex.ntotal, 0);
  gpuIndex.add(numNans, nans.data());

  // Only the single valid vector should have added
  EXPECT_EQ(gpuIndex.ntotal, 1);

  std::vector<float> queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
  std::vector<float> distance(opt.numQuery * opt.k, 0);
  std::vector<faiss::Index::idx_t> indices(opt.numQuery * opt.k, 0);

  // should not crash
  gpuIndex.search(opt.numQuery, queryVecs.data(), opt.k,
                  distance.data(), indices.data());

}
