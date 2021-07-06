/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/IcmEncoder.cuh>

#include <faiss/gpu/GpuResources.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/Pair.cuh>
#include <faiss/gpu/utils/Reductions.cuh>

#include <curand_kernel.h>

namespace faiss {
namespace gpu {

extern __shared__ char smem[];

/** encode using iterative conditional mode
 *
 * For every subcode ci (i = 1, ..., M) of a vector, we fix the other
 * subcodes cj (j != i) and then find the optimal value of ci such
 * that minimizing the objective function.
 *
 * @param uterm  precomputed unary terms, size (n, M, K)
 * @param bterm  precomputed binary terms, size (M1, M2, K1, K2)
 * @param codes  output vector encodings, size (n, M)
 * @param M      number of codebooks
 * @param K      number of codewords in a codebook
 * @param m      identify which subcode to condition on
 */
__global__ void runIcmEncodeStep(
        const float* uterm,
        const float* bterm,
        int32_t* codes,
        int M,
        int K,
        int m) {
    using KVPair = Pair<float, int>;
    // constexpr int smemSize = (K + kWarpSize - 1) / kWarpSize;

    int id = blockIdx.x;
    int code = threadIdx.x;
    // __shared__ KVPair smem[smemSize];

    KVPair obj(0.0f, code);
    obj.k = uterm[id * K + code];

    // unrolling this loop does not improve speed
    for (int m2 = 0; m2 < M; m2++) {
        if (m2 == m) {
            continue;
        }
        int32_t code2 = codes[id * M + m2];
        obj.k += bterm[m2 * K * K + code * K + code2];
    }

    __syncthreads();
    obj = blockReduceAll<KVPair, Min<KVPair>, false, false>(
            obj, Min<KVPair>(), (KVPair*)smem);

    if (code == 0) {
        codes[id * M + m] = obj.v;
    }
}

__global__ void runEvaluate(
        const float* x,
        const float* codebooks,
        const int32_t* codes,
        float* obj, // output
        int n,
        int M,
        int K,
        int dims) {
    int id = blockIdx.x; // index of the vector
    int d = threadIdx.x; // dimension
    // extern __shared__ float smem[];

    float acc = 0.0f;

    // TODO: unroll M ?
    for (int m = 0; m < M; m++) {
        int32_t code = codes[id * M + m];
        acc += codebooks[m * K * dims + code * dims + d];
    }

    acc -= x[id * dims + d];
    acc = acc * acc;

    __syncthreads();
    acc = blockReduceAllSum<float, false, false>(acc, (float*)smem);

    if (d == 0) {
        obj[id] = acc;
    }
}

__global__ void runPerturbCodes(
        int seed,
        int32_t* codes,
        int n,
        int M,
        int K,
        int nperts) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // index of the vector

    if (id >= n) {
        return;
    }

    // we have to initialize the state
    curandState_t state;
    curand_init(seed, id, 0, &state);

    for (int i = 0; i < nperts; i++) {
        int pos = int(curand_uniform(&state) * M);
        int32_t val = int32_t(curand_uniform(&state) * K);
        codes[id * M + pos] = val;
    }
}

__global__ void runSelectBest(
        int32_t* bestCodes,
        float* bestObjs,
        const int32_t* codes,
        const float* objs,
        int n,
        int M) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // index of the vector

    if (id >= n || objs[id] >= bestObjs[id]) {
        return;
    }

    bestObjs[id] = objs[id];
    for (int m = 0; m < M; m++) {
        bestCodes[id * M + m] = codes[id * M + m];
    }
}

__global__ void runNormAdd(float* bterm, const float* norm, int K) {
    int id = blockIdx.x;
    int code = threadIdx.x;

    bterm[id * K + code] += norm[code];
}

IcmEncoderImpl::IcmEncoderImpl(
        int M,
        int K,
        int dims,
        GpuResourcesProvider* prov,
        int device)
        : M(M), K(K), dims(dims), prov(prov), device(device) {
    res = prov->getResources();
}

void IcmEncoderImpl::computeUnaryTerms(
        float* uterm,           // output, [M, n, K]
        const float* x,         // [n, d]
        const float* codebooks, // [M, K, d]
        int n) const {
    auto stream = res->getDefaultStreamCurrentDevice();
    auto handle = res->getBlasHandleCurrentDevice();

    DeviceTensor<float, 2, true> vecs(const_cast<float*>(x), {n, dims});
    for (int m = 0; m < M; m++) {
        auto cPtr = const_cast<float*>(codebooks + m * K * dims);
        auto bPtr = uterm + m * n * K;
        DeviceTensor<float, 2, true> ci(cPtr, {K, dims});
        DeviceTensor<float, 2, true> bi(bPtr, {n, K});
        runMatrixMult(
                bi, false, vecs, false, ci, true, -2.0f, 0.0f, handle, stream);
    }

    DeviceTensor<float, 2, true> c(
            const_cast<float*>(codebooks), {M * K, dims});
    DeviceTensor<float, 1, true> norm(
            res.get(), makeTempAlloc(AllocType::Other, stream), {M * K});
    runL2Norm(c, true, norm, true, stream);

    for (int m = 0; m < M; m++) {
        auto bPtr = uterm + m * n * K;
        auto nPtr = norm.data() + m * K;
        runNormAdd<<<n, K, 0, stream>>>(bPtr, nPtr, K);
    }
}

void IcmEncoderImpl::computeBinaryTerms(float* bterm, const float* codebooks)
        const {
    auto stream = res->getDefaultStreamCurrentDevice();
    auto handle = res->getBlasHandleCurrentDevice();

    for (int m1 = 0; m1 < M; m1++) {
        for (int m2 = 0; m2 < M; m2++) {
            auto ptr1 = const_cast<float*>(codebooks + m1 * K * dims);
            auto ptr2 = const_cast<float*>(codebooks + m2 * K * dims);
            auto ptr3 = bterm + m1 * M * K * K + m2 * K * K;
            DeviceTensor<float, 2, true> c1(ptr1, {K, dims});
            DeviceTensor<float, 2, true> c2(ptr2, {K, dims});
            DeviceTensor<float, 2, true> b(ptr3, {K, K});
            runMatrixMult(
                    b, false, c1, false, c2, true, 2.0f, 0.0f, handle, stream);
        }
    }
}


void IcmEncoderImpl::setBinaryTerm(const float* codebooksHost) {
    DeviceScope scope(device);
    auto device = getCurrentDevice();
    auto stream = res->getDefaultStreamCurrentDevice();

    codebooks = toDeviceNonTemporary<float, 3>(
            res.get(),
            device,
            const_cast<float*>(codebooksHost),
            stream,
            {M, K, dims});
    bterm = DeviceTensor<float, 4, true>(
            res.get(), makeDevAlloc(AllocType::Other, stream), {M, M, K, K});
    computeBinaryTerms(bterm.data(), codebooks.data());
}

void IcmEncoderImpl::encodeImpl(
        int32_t* codesHost,
        const float* xHost,
        const float* codebooksHost,
        std::mt19937& gen,
        int n,
        int K,
        int nperts,
        int ilsIters,
        int icmIters) const {
    DeviceScope scope(device);
    auto device = getCurrentDevice();
    auto stream = res->getDefaultStreamCurrentDevice();

    auto codes = toDeviceTemporary<int32_t, 2>(
            res.get(), device, const_cast<int32_t*>(codesHost), stream, {n, M});
    auto x = toDeviceTemporary<float, 2>(
            res.get(), device, const_cast<float*>(xHost), stream, {n, dims});

    DeviceTensor<float, 3, true> uterm(
            res.get(), makeTempAlloc(AllocType::Other, stream), {M, n, K});
    computeUnaryTerms(uterm.data(), x.data(), codebooks.data(), n);

    DeviceTensor<int32_t, 2, true> bestCodes(
            res.get(), makeTempAlloc(AllocType::Other, stream), {n, M});
    fromDevice<int32_t, 2>(codes, bestCodes.data(), stream);

    DeviceTensor<float, 1, true> bestObjs(
            res.get(), makeTempAlloc(AllocType::Other, stream), {n});

    DeviceTensor<float, 1, true> objs(
            res.get(), makeTempAlloc(AllocType::Other, stream), {n});

    const int evaluateSmem = sizeof(float) * (dims + kWarpSize - 1) / kWarpSize;
    const int encodeSmem =
            sizeof(Pair<float, int>) * (K + kWarpSize - 1) / kWarpSize;

    runEvaluate<<<n, dims, evaluateSmem, stream>>>(
            x.data(),
            codebooks.data(),
            codes.data(),
            bestObjs.data(),
            n,
            M,
            K,
            dims);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int i = 0; i < ilsIters; i++) {
        runPerturbCodes<<<numBlocks, blockSize, 0, stream>>>(
                gen(), codes.data(), n, M, K, nperts);

        for (int j = 0; j < icmIters; j++) {
            for (int m = 0; m < M; m++) {
                runIcmEncodeStep<<<n, K, encodeSmem, stream>>>(
                        uterm[m].data(),
                        bterm[m].data(),
                        codes.data(),
                        M,
                        K,
                        m);
            }
        }

        runEvaluate<<<n, dims, evaluateSmem, stream>>>(
                x.data(),
                codebooks.data(),
                codes.data(),
                objs.data(),
                n,
                M,
                K,
                dims);

        runSelectBest<<<numBlocks, blockSize, 0, stream>>>(
                bestCodes.data(),
                bestObjs.data(),
                codes.data(),
                objs.data(),
                n,
                M);

        codes.copyFrom(bestCodes, stream);
    }

    // copy back to host memory
    fromDevice<int32_t, 2>(bestCodes, codesHost, stream);
}

void IcmEncoderImpl::encode(
        int32_t* codes,
        const float* x,
        const float* codebooks,
        std::mt19937& gen,
        int n,
        int nperts,
        int ilsIters,
        int icmIters) const {
    FAISS_THROW_IF_NOT(K <= (1 << 16));

    encodeImpl(
            codes, x, codebooks, gen, n, K, nperts, ilsIters, icmIters);
}

} // namespace gpu
} // namespace faiss
