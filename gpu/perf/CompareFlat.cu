
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "../../IndexFlat.h"
#include "../../utils.h"
#include "../GpuIndexFlat.h"
#include "IndexWrapper.h"
#include "../test/TestUtils.h"
#include "../utils/DeviceTensor.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/HostTensor.cuh"
#include "../utils/Timer.h"
#include <gflags/gflags.h>
#include <map>
#include <memory>
#include <vector>

#include <cuda_profiler_api.h>

DEFINE_int32(k, 3, "final number of closest results returned");
DEFINE_int32(num, 128, "# of vecs");
DEFINE_int32(dim, 128, "# of dimensions");
DEFINE_int32(num_queries, 3, "number of query vectors");
DEFINE_bool(diff, true, "show exact distance + index output discrepancies");
DEFINE_bool(use_float16, false, "use encodings in float16 instead of float32");
DEFINE_int64(seed, -1, "specify random seed");
DEFINE_int32(num_gpus, 1, "number of gpus to use");
DEFINE_int64(pinned_mem, 0, "pinned memory allocation to use");

using namespace faiss::gpu;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  cudaProfilerStop();

  auto seed = FLAGS_seed != -1L ? FLAGS_seed : time(nullptr);
  printf("using seed %ld\n", seed);

  auto numQueries = FLAGS_num_queries;

  auto index = std::unique_ptr<faiss::IndexFlatL2>(
    new faiss::IndexFlatL2(FLAGS_dim));

  HostTensor<float, 2, true> vecs({FLAGS_num, FLAGS_dim});
  faiss::float_rand(vecs.data(), vecs.numElements(), seed);

  index->add(FLAGS_num, vecs.data());

  printf("Database: dim %d num vecs %d\n", FLAGS_dim, FLAGS_num);
  printf("L2 lookup: %d queries, total k %d\n",
         numQueries, FLAGS_k);
  printf("float16 encoding %s\n", FLAGS_use_float16 ? "enabled" : "disabled");

  // Convert to GPU index
  printf("Copying index to %d GPU(s)...\n", FLAGS_num_gpus);

  auto initFn = [&index](faiss::gpu::GpuResources* res, int dev) ->
    std::unique_ptr<faiss::gpu::GpuIndexFlatL2> {
    ((faiss::gpu::StandardGpuResources*) res)->setPinnedMemory(
      FLAGS_pinned_mem);

    auto p = std::unique_ptr<faiss::gpu::GpuIndexFlatL2>(
      new faiss::gpu::GpuIndexFlatL2(res, dev, FLAGS_use_float16, index.get()));
    return p;
  };

  IndexWrapper<faiss::gpu::GpuIndexFlatL2> gpuIndex(FLAGS_num_gpus, initFn);
  printf("copy done\n");

  // Build query vectors
  HostTensor<float, 2, true> cpuQuery({numQueries, FLAGS_dim});
  faiss::float_rand(cpuQuery.data(), cpuQuery.numElements(), seed);

  // Time faiss CPU
  HostTensor<float, 2, true> cpuDistances({numQueries, FLAGS_k});
  HostTensor<faiss::Index::idx_t, 2, true> cpuIndices({numQueries, FLAGS_k});

  float cpuTime = 0.0f;

  {
    CpuTimer timer;
    index->search(numQueries,
                  cpuQuery.data(),
                  FLAGS_k,
                  cpuDistances.data(),
                  cpuIndices.data());

    cpuTime = timer.elapsedMilliseconds();
  }

  printf("CPU time %.3f ms\n", cpuTime);

  HostTensor<float, 2, true> gpuDistances({numQueries, FLAGS_k});
  HostTensor<faiss::Index::idx_t, 2, true> gpuIndices({numQueries, FLAGS_k});

  CUDA_VERIFY(cudaProfilerStart());
  faiss::gpu::synchronizeAllDevices();

  float gpuTime = 0.0f;

  // Time GPU
  {
    CpuTimer timer;

    gpuIndex.getIndex()->search(cpuQuery.getSize(0),
                                cpuQuery.data(),
                                FLAGS_k,
                                gpuDistances.data(),
                                gpuIndices.data());

    // There is a device -> host copy above, so no need to time
    // additional synchronization with the GPU
    gpuTime = timer.elapsedMilliseconds();
  }

  CUDA_VERIFY(cudaProfilerStop());
  printf("GPU time %.3f ms\n", gpuTime);

  compareLists(cpuDistances.data(), cpuIndices.data(),
               gpuDistances.data(), gpuIndices.data(),
               numQueries, FLAGS_k,
               "", true, FLAGS_diff, false);

  CUDA_VERIFY(cudaDeviceSynchronize());
  // printf("\ncudaMalloc usage %zd\n",
  //        resources.getMemoryManager().getHighWaterCudaMalloc());

  return 0;
}
