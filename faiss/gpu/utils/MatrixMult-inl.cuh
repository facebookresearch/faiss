/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cublas_v2.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

template <typename T>
struct GetCudaType;

template <>
struct GetCudaType<float> {
    static constexpr cudaDataType_t Type = CUDA_R_32F;
};

template <>
struct GetCudaType<half> {
    static constexpr cudaDataType_t Type = CUDA_R_16F;
};

template <typename AT, typename BT>
cublasStatus_t rawGemm(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const float fAlpha,
        const void* A,
        int lda,
        const void* B,
        int ldb,
        const float fBeta,
        float* C,
        int ldc) {
    auto cAT = GetCudaType<AT>::Type;
    auto cBT = GetCudaType<BT>::Type;

    // FIXME: some weird CUDA 11 bug? where cublasSgemmEx on
    // f16 (8, 64) x f16 (64, 64)' = f32 (8, 64) returns "not supported".
    // cublasGemmEx using CUBLAS_COMPUTE_32F also fails, but
    // CUBLAS_COMPUTE_32F_PEDANTIC does not fail (as seen on a V100).
    //
    // Only use the PEDANTIC implementation if the input matrices are f16
    // and we are on CUDA 11+
#if CUDA_VERSION >= 11000
    if (cAT == CUDA_R_16F || cBT == CUDA_R_16F) {
        return cublasGemmEx(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                &fAlpha,
                A,
                cAT,
                lda,
                B,
                cBT,
                ldb,
                &fBeta,
                C,
                CUDA_R_32F,
                ldc,
                CUBLAS_COMPUTE_32F_PEDANTIC,
                CUBLAS_GEMM_DEFAULT);
    }
#endif

    // Always accumulate in f32
    return cublasSgemmEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            &fAlpha,
            A,
            cAT,
            lda,
            B,
            cBT,
            ldb,
            &fBeta,
            C,
            CUDA_R_32F,
            ldc);
}

template <typename AT, typename BT>
cublasStatus_t rawBatchGemm(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const float fAlpha,
        const void* A,
        int lda,
        long long int strideA,
        const void* B,
        int ldb,
        long long int strideB,
        const float fBeta,
        float* C,
        int ldc,
        long long int strideC,
        int batchCount) {
    auto cAT = GetCudaType<AT>::Type;
    auto cBT = GetCudaType<BT>::Type;

    // Always accumulate in f32
    return cublasGemmStridedBatchedEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            &fAlpha,
            A,
            cAT,
            lda,
            strideA,
            B,
            cBT,
            ldb,
            strideB,
            &fBeta,
            C,
            CUDA_R_32F,
            ldc,
            strideC,
            batchCount,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
}

template <typename AT, typename BT>
void runMatrixMult(
        Tensor<float, 2, true>& c,
        bool transC,
        Tensor<AT, 2, true>& a,
        bool transA,
        Tensor<BT, 2, true>& b,
        bool transB,
        float alpha,
        float beta,
        cublasHandle_t handle,
        cudaStream_t stream) {
    cublasSetStream(handle, stream);

    // Check that we have (m x k) * (k x n) = (m x n)
    // using the input row-major layout
    int aM = transA ? a.getSize(1) : a.getSize(0);
    int aK = transA ? a.getSize(0) : a.getSize(1);

    int bK = transB ? b.getSize(1) : b.getSize(0);
    int bN = transB ? b.getSize(0) : b.getSize(1);

    int cM = transC ? c.getSize(1) : c.getSize(0);
    int cN = transC ? c.getSize(0) : c.getSize(1);

    FAISS_ASSERT(aM == cM);
    FAISS_ASSERT(aK == bK);
    FAISS_ASSERT(bN == cN);

    FAISS_ASSERT(a.getStride(1) == 1);
    FAISS_ASSERT(b.getStride(1) == 1);
    FAISS_ASSERT(c.getStride(1) == 1);

    // Now, we have to represent the matrix multiplication in
    // column-major layout
    float* pC = c.data();

    int m = c.getSize(1); // stride 1 size
    int n = c.getSize(0); // other size
    int k = transA ? a.getSize(0) : a.getSize(1);

    int lda = transC ? a.getStride(0) : b.getStride(0);
    int ldb = transC ? b.getStride(0) : a.getStride(0);
    int ldc = c.getStride(0);

    auto gemmTrA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto gemmTrB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    if (transC) {
        gemmTrA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;
        gemmTrB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
    }

    cublasStatus_t err;

    if (transC) {
        err = rawGemm<AT, BT>(
                handle,
                gemmTrA,
                gemmTrB,
                m,
                n,
                k,
                alpha,
                a.data(),
                lda,
                b.data(),
                ldb,
                beta,
                pC,
                ldc);
    } else {
        err = rawGemm<AT, BT>(
                handle,
                gemmTrA,
                gemmTrB,
                m,
                n,
                k,
                alpha,
                b.data(),
                lda,
                a.data(),
                ldb,
                beta,
                pC,
                ldc);
    }

    FAISS_ASSERT_FMT(
            err == CUBLAS_STATUS_SUCCESS,
            "cublas failed (%d): "
            "(%d, %d)%s x (%d, %d)%s = (%d, %d)%s "
            "gemm params m %d n %d k %d trA %s trB %s lda %d ldb %d ldc %d",
            (int)err,
            a.getSize(0),
            a.getSize(1),
            transA ? "'" : "",
            b.getSize(0),
            b.getSize(1),
            transB ? "'" : "",
            c.getSize(0),
            c.getSize(1),
            transC ? "'" : "",
            m,
            n,
            k,
            gemmTrA == CUBLAS_OP_T ? "T" : "N",
            gemmTrB == CUBLAS_OP_T ? "T" : "N",
            lda,
            ldb,
            ldc);
    CUDA_TEST_ERROR();
}

template <typename AT, typename BT>
void runBatchMatrixMult(
        Tensor<float, 3, true>& c,
        bool transC,
        Tensor<AT, 3, true>& a,
        bool transA,
        Tensor<BT, 3, true>& b,
        bool transB,
        float alpha,
        float beta,
        cublasHandle_t handle,
        cudaStream_t stream) {
    FAISS_ASSERT(c.getSize(0) == a.getSize(0));
    FAISS_ASSERT(a.getSize(0) == b.getSize(0));

    // This uses the strided batch MM, which assumes a uniform stride
    FAISS_ASSERT(a.getStride(0) == a.getSize(1) * a.getSize(2));
    FAISS_ASSERT(b.getStride(0) == b.getSize(1) * b.getSize(2));
    FAISS_ASSERT(c.getStride(0) == c.getSize(1) * c.getSize(2));

    cublasSetStream(handle, stream);

    // Check that we have (m x k) * (k x n) = (m x n)
    // using the input row-major layout
    int aM = transA ? a.getSize(2) : a.getSize(1);
    int aK = transA ? a.getSize(1) : a.getSize(2);

    int bK = transB ? b.getSize(2) : b.getSize(1);
    int bN = transB ? b.getSize(1) : b.getSize(2);

    int cM = transC ? c.getSize(2) : c.getSize(1);
    int cN = transC ? c.getSize(1) : c.getSize(2);

    FAISS_ASSERT(aM == cM);
    FAISS_ASSERT(aK == bK);
    FAISS_ASSERT(bN == cN);

    // Now, we have to represent the matrix multiplication in
    // column-major layout
    void* pA = transC ? (void*)a.data() : (void*)b.data();
    void* pB = transC ? (void*)b.data() : (void*)a.data();
    float* pC = c.data();

    int m = c.getSize(2); // stride 1 size
    int n = c.getSize(1); // other size
    int k = transA ? a.getSize(1) : a.getSize(2);

    int lda = transC ? a.getStride(1) : b.getStride(1);
    int ldb = transC ? b.getStride(1) : a.getStride(1);
    int ldc = c.getStride(1);

    auto gemmTrA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto gemmTrB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    if (transC) {
        gemmTrA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;
        gemmTrB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
    }

    long long int gemmStrideA = transC ? a.getStride(0) : b.getStride(0);
    long long int gemmStrideB = transC ? b.getStride(0) : a.getStride(0);
    long long int gemmStrideC = c.getStride(0);

    auto err = rawBatchGemm<AT, BT>(
            handle,
            gemmTrA,
            gemmTrB,
            m,
            n,
            k,
            alpha,
            pA,
            lda,
            gemmStrideA,
            pB,
            ldb,
            gemmStrideB,
            beta,
            pC,
            ldc,
            gemmStrideC,
            a.getSize(0));

    FAISS_ASSERT_FMT(
            err == CUBLAS_STATUS_SUCCESS,
            "cublasGemmStridedBatchedEx failed (%d)",
            (int)err);
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
