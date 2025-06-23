/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
 * For subcode cm of a vector, we fix the other subcodes cj (j != m)
 * and then find the optimal value of cm (cm = 1,...,K) such that
 * minimizing the objective function.
 *
 * @param uterm  precomputed unary terms, size (M, n, K)
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

    auto id = blockIdx.x;    // each block takes care of one vector
    auto code = threadIdx.x; // each thread takes care of one possible code

    // compute the objective value by look-up tables
    KVPair obj(0.0f, code);
    obj.k = uterm[id * K + code];

#pragma unroll
    for (int m2 = 0; m2 < M; m2++) {
        if (m2 == m) {
            continue;
        }
        int32_t code2 = codes[id * M + m2];
        obj.k += bterm[m2 * K * K + code * K + code2];
    }

    // find the minimum objective value and the corresponding code
    __syncthreads();
    obj = blockReduceAll<KVPair, Min<KVPair>, false, false>(
            obj, Min<KVPair>(), (KVPair*)smem);

    if (code == 0) {
        codes[id * M + m] = obj.v;
    }
}

/** compute reconstruction error for each vector
 *
 * decoded_x[i] = \sum codebooks[m][codes[i][m]], m = 1,..,M
 * obj[i] = ||x[i] - decoded_x[i]||^2
 *
 * @param x      input vectors, size [n, dims]
 * @param codebooks  codebooks, size [M, K, dims]
 * @param codes  vector codes, size [n, M]
 * @param obj    output reconstruction errors, size [n]
 * @param n      number of input vectors
 * @param K      number of codewords in a codebook
 * @param M      number of codebooks
 */
__global__ void runEvaluation(
        const float* x,
        const float* codebooks,
        const int32_t* codes,
        float* obj, // output
        int n,
        int M,
        int K,
        int dims) {
    auto id = blockIdx.x; // each block takes care of one vector
    auto d = threadIdx.x; // each thread takes care of one dimension
    float acc = 0.0f;

#pragma unroll
    for (int m = 0; m < M; m++) {
        int32_t code = codes[id * M + m];
        acc += codebooks[m * K * dims + code * dims + d];
    }

    acc -= x[id * dims + d];
    acc = acc * acc;

    // sum values of all dimensions together
    __syncthreads();
    acc = blockReduceAllSum<float, false, false>(acc, (float*)smem);

    if (d == 0) {
        obj[id] = acc;
    }
}

/** perturb vector codes
 *
 * repeat nperts times:
 *   codes[i][randint(0, M)] = randint(0, K)
 *
 * @param seed   random seed
 * @param codes  vector codes, size [n, M]
 * @param n      number of input vectors
 * @param M      number of codebooks
 * @param K      number of codewords in a codebook
 * @param nperts number of subcode to be perturbed in a vector
 */
__global__ void runCodesPerturbation(
        int seed,
        int32_t* codes,
        int n,
        int M,
        int K,
        int nperts) {
    // each thread takes care of one vector
    auto id = blockIdx.x * blockDim.x + threadIdx.x;

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

/** select the best codes by reconstruction errors
 *
 * if objs[i] < best_objs[i]:
 *     best_objs[i] = objs[i]
 *     best_codes[i] = codes[i]
 *
 * @param bestCodes the best codes we've encountered, size [n, M]
 * @param bestObjs  min reconstruction errors we've encountered, size [n]
 * @param codes     input vector codes, size [n, M]
 * @param objs      reconstruction errors of input vector codes, size [n]
 * @param n         number of input vectors
 */
__global__ void runCodesSelection(
        int32_t* bestCodes,
        float* bestObjs,
        const int32_t* codes,
        const float* objs,
        int n,
        int M) {
    // each thread takes care of one vector
    auto id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= n || objs[id] >= bestObjs[id]) {
        return;
    }

    bestObjs[id] = objs[id];
#pragma unroll
    for (int m = 0; m < M; m++) {
        bestCodes[id * M + m] = codes[id * M + m];
    }
}

/** add L2 norm of codewords in a codebook to the unary terms
 *
 * uterm[i][k] = norm[k]
 *
 * @param uterm unary terms, size [n, K]
 * @param norm  L2 norm of each codeword in a codebook, size [K]
 * @param K     number of codewords in a codebook
 */
__global__ void runNormAddition(float* uterm, const float* norm, int K) {
    auto id = blockIdx.x;
    auto code = threadIdx.x;

    uterm[id * K + code] += norm[code];
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
        auto uPtr = uterm + m * n * K;
        auto nPtr = norm.data() + m * K;
        runNormAddition<<<n, K, 0, stream>>>(uPtr, nPtr, K);
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

    // copy from host to device memory
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

void IcmEncoderImpl::encode(
        int32_t* codesHost,
        const float* xHost,
        const float* codebooksHost,
        std::mt19937& gen,
        int n,
        int nperts,
        int ilsIters,
        int icmIters) const {
    DeviceScope scope(device);
    auto device = getCurrentDevice();
    auto stream = res->getDefaultStreamCurrentDevice();

    // copy from host to device memory
    auto codes = toDeviceTemporary<int32_t, 2>(
            res.get(), device, const_cast<int32_t*>(codesHost), stream, {n, M});
    auto x = toDeviceTemporary<float, 2>(
            res.get(), device, const_cast<float*>(xHost), stream, {n, dims});

    // compute unary terms
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

    // compute how much shared memory we need
    int warpSize = getWarpSizeCurrentDevice();
    const int evaluateSmem = sizeof(float) * (dims + warpSize - 1) / warpSize;
    const int encodeSmem =
            sizeof(Pair<float, int>) * (K + warpSize - 1) / warpSize;

    // compute the reconstruction error for each vector
    runEvaluation<<<n, dims, evaluateSmem, stream>>>(
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
        runCodesPerturbation<<<numBlocks, blockSize, 0, stream>>>(
                gen(), codes.data(), n, M, K, nperts);

        // perform icm encoding
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

        // compute the reconstruction error for each vector given codes
        runEvaluation<<<n, dims, evaluateSmem, stream>>>(
                x.data(),
                codebooks.data(),
                codes.data(),
                objs.data(),
                n,
                M,
                K,
                dims);

        // if objs[i] < best_objs[i], replace best_codes[i] with codes[i]
        runCodesSelection<<<numBlocks, blockSize, 0, stream>>>(
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

} // namespace gpu
} // namespace faiss
