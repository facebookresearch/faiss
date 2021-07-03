/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/ICM.cuh>

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Pair.cuh>
#include <faiss/gpu/utils/Reductions.cuh>

namespace faiss {
namespace gpu {

template <size_t M, size_t K>
__global__ void runIcmEncodeStep(
        const float* u, // [n, M, K]
        const float* b, // [M1, M2, K1, K2]
        int32_t* codes, // [n, M]
        size_t n,
        size_t m) {
    int id = blockIdx.x;
    int code = threadIdx.x;

    __shared__ Pair<float, int> smem[K / kWarpSize];
    Pair<float, int> obj(0.0f, code);

    obj.k = u[id * M * K + m * K + code];
    const float* mb = b + m * M * K * K;

#pragma unroll
    for (size_t other_m = 0; other_m < M; other_m++) {
        if (other_m == m) {
            continue;
        }

        int32_t code2 = codes[id * M + other_m];
        obj.k += mb[other_m * K * K + code * K + code2];
    }

    __syncthreads();

    obj = blockReduceAll<Pair<float, int>, Min<Pair<float, int>>, false, false>(
            obj, Min<Pair<float, int>>(), smem);

    if (code == 0) {
        codes[id * M + m] = obj.v;
    }
}

void IcmEncoder::set_unary_term(size_t n, const float* unaries) {
    auto device = getCurrentDevice();
    auto stream = res->getDefaultStreamCurrentDevice();
    uterm = toDeviceNonTemporary<float, 3>(
            res.get(),
            device,
            const_cast<float*>(unaries),
            stream,
            {int(n), int(M), int(K)});
}

void IcmEncoder::set_binary_term(const float* binaries) {
    auto device = getCurrentDevice();
    auto stream = res->getDefaultStreamCurrentDevice();
    bterm = toDeviceNonTemporary<float, 4>(
            res.get(),
            device,
            const_cast<float*>(binaries),
            stream,
            {int(M), int(M), int(K), int(K)});
}

void IcmEncoder::encode(int32_t* codes_host, size_t n) const {
    auto device = getCurrentDevice();
    auto stream = res->getDefaultStreamCurrentDevice();
    auto codes = toDeviceTemporary<int32_t, 2>(
            res.get(), device, codes_host, stream, {int(n), int(M)});

    constexpr size_t K = 256;
    constexpr size_t M = 8;
    size_t smem = sizeof(Pair<float, int>) * K / kWarpSize;

#pragma unroll
    for (size_t m = 0; m < M; m++) {
        runIcmEncodeStep<M, K><<<n, K, smem, stream>>>(
                uterm.data(), bterm.data(), codes.data(), n, m);
    }

    // copy back to host memory
    fromDevice<int32_t, 2>(codes, codes_host, stream);
}

} // namespace gpu
} // namespace faiss
