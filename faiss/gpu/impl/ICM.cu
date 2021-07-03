/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/ICM.cuh>

#include <faiss/gpu/GpuResources.h>
#include <faiss/impl/FaissAssert.h>
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

    __shared__ Pair<float, int> smem[(K + kWarpSize - 1) / kWarpSize];
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

IcmEncoder::IcmEncoder(size_t M, size_t K, GpuResourcesProvider* prov)
        : M(M), K(K), prov(prov) {
    res = prov->getResources();
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

void IcmEncoder::encode(int32_t* codes, size_t n) const {
    // FAISS_THROW_IF_NOT(K % kWarpSize == 0);
    FAISS_THROW_IF_NOT(K <= (1 << 16));

#define DISPATCH_M(NCB)                      \
    case (NCB):                              \
        encode_dispatch_k<NCB>(codes, n, K); \
        break

    switch (M) {
        DISPATCH_M(1);
        DISPATCH_M(2);
        DISPATCH_M(3);
        DISPATCH_M(4);
        DISPATCH_M(5);
        DISPATCH_M(6);
        DISPATCH_M(7);
        DISPATCH_M(8);
        DISPATCH_M(9);
        DISPATCH_M(10);
        DISPATCH_M(11);
        DISPATCH_M(12);
        DISPATCH_M(13);
        DISPATCH_M(14);
        DISPATCH_M(15);
        DISPATCH_M(16);
        DISPATCH_M(17);
        DISPATCH_M(18);
        DISPATCH_M(19);
        DISPATCH_M(21);
        DISPATCH_M(22);
        DISPATCH_M(23);
        DISPATCH_M(24);
        DISPATCH_M(25);
        DISPATCH_M(26);
        DISPATCH_M(27);
        DISPATCH_M(28);
        DISPATCH_M(29);
        DISPATCH_M(30);
        DISPATCH_M(31);
        DISPATCH_M(32);
        DISPATCH_M(64);
        DISPATCH_M(128);
        default:
            FAISS_THROW_MSG("Invalid number of codebooks");
    }
#undef DISPATCH_M
}

template <size_t kNumCodebooks>
void IcmEncoder::encode_dispatch_k(int32_t* codes, size_t n, size_t K) const {
#define DISPATCH_K(nbits)                                   \
    case (1 << nbits):                                      \
        encode_impl<kNumCodebooks, (1 << nbits)>(codes, n); \
        break

    switch (K) {
        DISPATCH_K(1);
        DISPATCH_K(2);
        DISPATCH_K(3);
        DISPATCH_K(4);
        DISPATCH_K(5);
        DISPATCH_K(6);
        DISPATCH_K(7);
        DISPATCH_K(8); // K = 256
        DISPATCH_K(9);
        DISPATCH_K(10);
        DISPATCH_K(11);
        DISPATCH_K(12);
        DISPATCH_K(13);
        DISPATCH_K(14);
        DISPATCH_K(15);
        DISPATCH_K(16);
        default:
            FAISS_THROW_MSG("Invalid codebook size");
    }
#undef DISPATCH_K
}

template <size_t kNumCodebooks, size_t kCodebookSize>
void IcmEncoder::encode_impl(int32_t* codes_host, size_t n) const {
    auto device = getCurrentDevice();
    auto stream = res->getDefaultStreamCurrentDevice();
    auto codes = toDeviceTemporary<int32_t, 2>(
            res.get(), device, codes_host, stream, {int(n), int(M)});
    size_t smem = sizeof(Pair<float, int>) * (kCodebookSize + kWarpSize - 1) / kWarpSize;

#pragma unroll
    for (size_t m = 0; m < kNumCodebooks; m++) {
        runIcmEncodeStep<kNumCodebooks, kCodebookSize>
                <<<n, kCodebookSize, smem, stream>>>(
                        uterm.data(), bterm.data(), codes.data(), n, m);
    }

    // copy back to host memory
    fromDevice<int32_t, 2>(codes, codes_host, stream);
}

} // namespace gpu
} // namespace faiss
