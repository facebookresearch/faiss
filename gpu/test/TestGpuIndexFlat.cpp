
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "../../IndexFlat.h"
#include "../GpuIndexFlat.h"
#include "../StandardGpuResources.h"
#include "../utils/DeviceUtils.h"
#include "../test/TestUtils.h"
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

// FIXME: figure out a better way to test fp16
constexpr float kF16MaxRelErr = 0.07f;
constexpr float kF32MaxRelErr = 6e-3f;

void testFlat(bool useL2, bool useFloat16, int kOverride = -1) {
  int numVecs = faiss::gpu::randVal(1000, 20000);
  int dim = faiss::gpu::randVal(50, 800);
  int numQuery = faiss::gpu::randVal(1, 512);

  // Due to loss of precision in a float16 accumulator, for large k,
  // the number of differences is pretty huge. Restrict ourselves to a
  // fairly small `k` for float16
  int k = useFloat16 ?
    std::min(faiss::gpu::randVal(1, 50), numVecs) :
    std::min(faiss::gpu::randVal(1, 1024), numVecs);
  if (kOverride > 0) {
    k = kOverride;
  }

  faiss::IndexFlatIP cpuIndexIP(dim);
  faiss::IndexFlatL2 cpuIndexL2(dim);

  faiss::IndexFlat* cpuIndex =
    useL2 ? (faiss::IndexFlat*) &cpuIndexL2 : (faiss::IndexFlat*) &cpuIndexIP;

  // Construct on a random device to test multi-device, if we have
  // multiple devices
  int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);

  faiss::gpu::StandardGpuResources res;
  res.noTempMemory();
  faiss::gpu::GpuIndexFlatIP gpuIndexIP(&res, device, dim, useFloat16);
  faiss::gpu::GpuIndexFlatL2 gpuIndexL2(&res, device, dim, useFloat16);

  faiss::gpu::GpuIndexFlat* gpuIndex =
    useL2 ? (faiss::gpu::GpuIndexFlat*) &gpuIndexL2 :
    (faiss::gpu::GpuIndexFlat*) &gpuIndexIP;

  std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
  cpuIndex->add(numVecs, vecs.data());
  gpuIndex->add(numVecs, vecs.data());

  std::stringstream str;
  str << (useL2 ? "L2" : "IP") << " numVecs " << numVecs
      << " dim " << dim
      << " useFloat16 " << useFloat16
      << " numQuery " << numQuery
      << " k " << k;

  // To some extent, we depend upon the relative error for the test
  // for float16
  faiss::gpu::compareIndices(*cpuIndex, *gpuIndex, numQuery, dim, k, str.str(),
                             useFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
                             // FIXME: the fp16 bounds are
                             // useless when math (the accumulator) is
                             // in fp16. Figure out another way to test
                             useFloat16 ? 0.99f : 0.1f,
                             useFloat16 ? 0.65f : 0.015f);
}

TEST(TestGpuIndexFlat, IP_Float32) {
  for (int tries = 0; tries < 10; ++tries) {
    faiss::gpu::newTestSeed();
    testFlat(false, false);
  }
}

TEST(TestGpuIndexFlat, L2_Float32) {
  for (int tries = 0; tries < 10; ++tries) {
    faiss::gpu::newTestSeed();
    testFlat(true, false);
  }
}

// test specialized k == 1 codepath
TEST(TestGpuIndexFlat, L2_Float32_K1) {
  for (int tries = 0; tries < 5; ++tries) {
    faiss::gpu::newTestSeed();
    testFlat(true, false, 1);
  }
}

TEST(TestGpuIndexFlat, IP_Float16) {
  for (int tries = 0; tries < 10; ++tries) {
    faiss::gpu::newTestSeed();
    testFlat(false, true);
  }
}

TEST(TestGpuIndexFlat, L2_Float16) {
  for (int tries = 0; tries < 10; ++tries) {
    faiss::gpu::newTestSeed();
    testFlat(true, true);
  }
}

// test specialized k == 1 codepath
TEST(TestGpuIndexFlat, L2_Float16_K1) {
  for (int tries = 0; tries < 5; ++tries) {
    faiss::gpu::newTestSeed();
    testFlat(true, true, 1);
  }
}

TEST(TestGpuIndexFlat, QueryEmpty) {
  faiss::gpu::StandardGpuResources res;
  res.noTempMemory();

  int dim = 128;
  faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, 0, dim, false);

  // Querying an empty index should not blow up, and just return
  // (FLT_MAX, -1)
  int numQuery = 10;
  int k = 50;
  std::vector<float> queries(numQuery * dim, 1.0f);

  std::vector<float> dist(numQuery * k, 0);
  std::vector<faiss::Index::idx_t> ind(numQuery * k);

  gpuIndex.search(numQuery, queries.data(), k, dist.data(), ind.data());

  for (auto d : dist) {
    EXPECT_EQ(d, std::numeric_limits<float>::max());
  }

  for (auto i : ind) {
    EXPECT_EQ(i, -1);
  }
}

TEST(TestGpuIndexFlat, CopyFrom) {
  faiss::gpu::newTestSeed();

  int numVecs = faiss::gpu::randVal(100, 200);
  int dim = faiss::gpu::randVal(1, 1000);

  faiss::IndexFlatL2 cpuIndex(dim);

  std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
  cpuIndex.add(numVecs, vecs.data());

  faiss::gpu::StandardGpuResources res;
  res.noTempMemory();

  // Fill with garbage values
  int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
  faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, device, 2000, false);
  gpuIndex.copyFrom(&cpuIndex);

  EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
  EXPECT_EQ(gpuIndex.ntotal, numVecs);

  EXPECT_EQ(cpuIndex.d, gpuIndex.d);
  EXPECT_EQ(cpuIndex.d, dim);

  int idx = faiss::gpu::randVal(0, numVecs - 1);

  std::vector<float> gpuVals(dim);
  gpuIndex.reconstruct(idx, gpuVals.data());

  std::vector<float> cpuVals(dim);
  cpuIndex.reconstruct(idx, cpuVals.data());

  EXPECT_EQ(gpuVals, cpuVals);
}

TEST(TestGpuIndexFlat, CopyTo) {
  faiss::gpu::newTestSeed();

  faiss::gpu::StandardGpuResources res;
  res.noTempMemory();

  int numVecs = faiss::gpu::randVal(100, 200);
  int dim = faiss::gpu::randVal(1, 1000);

  int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
  faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, device, dim, false);

  std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
  gpuIndex.add(numVecs, vecs.data());

  // Fill with garbage values
  faiss::IndexFlatL2 cpuIndex(2000);
  gpuIndex.copyTo(&cpuIndex);

  EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
  EXPECT_EQ(gpuIndex.ntotal, numVecs);

  EXPECT_EQ(cpuIndex.d, gpuIndex.d);
  EXPECT_EQ(cpuIndex.d, dim);

  int idx = faiss::gpu::randVal(0, numVecs - 1);

  std::vector<float> gpuVals(dim);
  gpuIndex.reconstruct(idx, gpuVals.data());

  std::vector<float> cpuVals(dim);
  cpuIndex.reconstruct(idx, cpuVals.data());

  EXPECT_EQ(gpuVals, cpuVals);
}
