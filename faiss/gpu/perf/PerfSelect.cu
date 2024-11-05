/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/WarpSelectKernel.cuh>
#include <sstream>
#include <unordered_map>
#include <vector>

DEFINE_int32(rows, 10000, "rows in matrix");
DEFINE_int32(cols, 40000, "cols in matrix");
DEFINE_int32(k, 100, "k");
DEFINE_bool(dir, false, "direction of sort");
DEFINE_bool(warp, false, "warp select");
DEFINE_int32(iter, 5, "iterations to run");
DEFINE_bool(k_powers, false, "test k powers of 2 from 1 -> max k");

int main(int argc, char** argv) {
    using namespace faiss::gpu;

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::vector<float> v = randVecs(FLAGS_rows, FLAGS_cols);
    HostTensor<float, 2, true> hostVal({FLAGS_rows, FLAGS_cols});

    for (int r = 0; r < FLAGS_rows; ++r) {
        for (int c = 0; c < FLAGS_cols; ++c) {
            hostVal[r][c] = v[r * FLAGS_cols + c];
        }
    }

    StandardGpuResources res;
    res.noTempMemory();

    auto resUse = res.getResources();

    // Select top-k on GPU
    DeviceTensor<float, 2, true> gpuVal(
            resUse.get(), makeDevAlloc(AllocType::Other, 0), hostVal);

    int startK = FLAGS_k;
    int limitK = FLAGS_k;

    if (FLAGS_k_powers) {
        startK = 1;
        limitK = GPU_MAX_SELECTION_K;
    }

    for (int k = startK; k <= limitK; k *= 2) {
        DeviceTensor<float, 2, true> gpuOutVal(
                resUse.get(),
                makeDevAlloc(AllocType::Other, 0),
                {FLAGS_rows, k});
        DeviceTensor<faiss::idx_t, 2, true> gpuOutInd(
                resUse.get(),
                makeDevAlloc(AllocType::Other, 0),
                {FLAGS_rows, k});

        for (int i = 0; i < FLAGS_iter; ++i) {
            if (FLAGS_warp) {
                runWarpSelect(gpuVal, gpuOutVal, gpuOutInd, FLAGS_dir, k, 0);
            } else {
                runBlockSelect(gpuVal, gpuOutVal, gpuOutInd, FLAGS_dir, k, 0);
            }
        }
    }

    cudaDeviceSynchronize();
}
