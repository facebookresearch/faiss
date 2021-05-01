/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LocalSearchQuantizer.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

#include <algorithm>

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

extern "C" {
    // LU decomoposition of a general matrix
    void sgetrf_(int* M, int *N, float* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void sgetri_(int* N, float* A, int* lda, int* IPIV, float* WORK, int* lwork, int* INFO);

    int sgemm_(
            const char* transa,
            const char* transb,
            FINTEGER* m,
            FINTEGER* n,
            FINTEGER* k,
            const float* alpha,
            const float* a,
            FINTEGER* lda,
            const float* b,
            FINTEGER* ldb,
            float* beta,
            float* c,
            FINTEGER* ldc);
}

namespace {

void fvec_sub(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

// c and a and b can overlap
void fvec_add(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

void fmat_inverse(float* a, int n) {
    int *ipiv = new int[n];
    int lwork = n * n;
    float *workspace = new float[lwork];
    int info;

    sgetrf_(&n, &n, a, &n, ipiv, &info);
    FAISS_THROW_IF_NOT(info == 0);
    sgetri_(&n, a, &n, ipiv, workspace, &lwork, &info);
    FAISS_THROW_IF_NOT(info == 0);

    delete[] ipiv;
    delete[] workspace;
}

// void fmat_mutiply()


void random_int32(std::vector<int32_t> *x, int32_t min,
        int32_t max, std::mt19937 &gen) {
    std::uniform_int_distribution<int32_t> distrib(min, max);
    for (size_t i = 0; i < x->size(); i++) {
        (*x)[i] = distrib(gen);
    }
}

void random_float(std::vector<float> *x, float min,
        float max, std::mt19937 &gen) {
    std::uniform_real_distribution<float> distrib(min, max);
    for (size_t i = 0; i < x->size(); i++) {
        (*x)[i] = distrib(gen);
    }
}

} // anonymous namespace

namespace faiss {

LocalSearchQuantizer::LocalSearchQuantizer(size_t d, size_t M, size_t nbits) {
    FAISS_THROW_IF_NOT(nbits == 8);
    this->d = d;
    this->M = M;
    this->nbits = nbits;

    verbose = false;
    code_size = M * (nbits / 8); // in bytes

    train_iters = 25;
    encode_iters = 25;
    ils_iters = 8;
    icm_iters = 4;
}


void LocalSearchQuantizer::train(size_t n, const float* x) {
    if (verbose) {
        printf("Training LocalSearchQuantizer, with %zd subcodes on %zd %zdD vectors\n",
               M,
               n,
               size_t(d));
    }

    size_t h = (1 << nbits);           // number of codes per codebook
    codebooks.resize(M * h * d);       // [M, h, d]
    std::vector<int32_t> codes(n * M); // [n, M]

    std::mt19937 gen(12345);
    random_int32(&codes, 0, h - 1, gen);

    // TODO: add SR-D
    // cov = np.diag(np.cov(x.T))
    // mean = np.zeros((d,))

    float obj = evaluate(codes.data(), x, n);
    printf("Init obj: %lf\n", obj);

    for (size_t i = 0; i < train_iters; i++) {
        update_codebooks(x, codes.data(), n);
        float obj = evaluate(codes.data(), x, n);
        printf("iter %zd, obj: %lf\n", i, obj);
    }

}

void LocalSearchQuantizer::compute_codes(
        const float* x,
        uint8_t* codes_out,
        size_t n) const {
    FAISS_THROW_MSG("Not implemented yet!");
}


void LocalSearchQuantizer::pack_codes(
        size_t n,
        const int32_t* codes,
        uint8_t* packed_codes,
        int64_t ld_codes) const {
    FAISS_THROW_MSG("Not implemented yet!");
}

void LocalSearchQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
    FAISS_THROW_MSG("Not implemented yet!");
}

/**
 *  
    Input:
        x: [n, d]
        codebooks: [m, h, d]
        b: [n, m]
    Formulation:
        B = sparse(b): [n, m*h]
        C: [m*h, d]
        L = (X - BC)^2
        X = BC
        => B'X = B'BC
        => C = (B'B + lambda * I)^(-1) B' X
        
        Let BB = B'B, BX = B'X 
 */
void LocalSearchQuantizer::update_codebooks(const float *x, const int32_t *codes, size_t n) {
    size_t h = (1 << nbits);

    // allocate memory
    std::vector<float> bb(M * h * M * h, 0.0f); // [M * h, M * h]
    std::vector<float> bx(M * h * d, 0.0f);     // [M * h, d]

    // compute B'B
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            int32_t code1 = codes[i * M + m];
            int32_t idx1 = m * h + code1;
            bb[idx1 * M * h + idx1] += 1;

            for (size_t k = m + 1; k < M; k++) {
                int32_t code2 = codes[i * M + k];
                int32_t idx2 = k * h + code2;
                bb[idx1 * M * h + idx2] += 1;
                bb[idx2 * M * h + idx1] += 1;
            }
        }
    }

    // for (size_t i = 0; i < M * h; i++) {
    //     printf("%lf ", bb[i]);
    // }
    // printf("\n");

    // compute BX
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            int32_t code = codes[i * M + m];
            float *data = bx.data() + (m * h + code) * d;
            fvec_add(d, data, x + i * d, data);
        }
    }

    // add a regularization term to B'B, make it inversible
    constexpr float lambda = 0.0001;
#pragma omp parallel for
    for (size_t i = 0; i < M * h; i++) {
        bb[i * (M * h) + i] += lambda;
    }

    // C = inv(bb) @ bx

    // compute inv(bb)
    fmat_inverse(bb.data(), M * h);  // [M*h, M*h]

    // compute inv(bb) @ bx

    // NOTE: LAPACK use column major order
    //
    // out = bb @ bx
    // out^T = bx^T @ bb^T
    //         [d, M*h] @ [M*h, M*h]
    //
    // out = alpha * op(A) * op(B) + beta * C
    // [M*h, M*h] @ [M*h, d]
    FINTEGER nrows_A = d;
    FINTEGER ncols_A = M * h;

    FINTEGER nrows_B = M * h;
    FINTEGER ncols_B = M * h;

    float alpha = 1.0f;
    float beta = 0.0f;
    sgemm_("Not Transposed",
            "Not Transposed",
            &nrows_A,  // nrows of op(A)
            &ncols_B,  // ncols of op(B)
            &ncols_A,  // ncols of op(A)
            &alpha,
            bx.data(),
            &nrows_A,  // nrows of A
            bb.data(),
            &nrows_B,  // nrows of B
            &beta,
            codebooks.data(),
            &nrows_A);  // nrows of output
}


float LocalSearchQuantizer::evaluate(const int32_t *codes,
        const float *x, size_t n) const {
    // decode
    size_t h = (1 << nbits);
    std::vector<float> decoded_x(n * d, 0.0f);
    float obj = 0.0f;

#pragma omp parallel for reduction(+ : obj)
    for (size_t i = 0; i < n; i++) {
        const auto code = codes + i * M;
        const auto decoded_i = decoded_x.data() + i * d;
        for (size_t m = 0; m < M; m++) {
            const auto c = codebooks.data() + m * h * d + code[m] * d; // codebooks[m, code[m]]
            fvec_add(d, decoded_i, c, decoded_i);
        }

        float err = fvec_L2sqr(x + i * d, decoded_i, d);
        obj += err;
    }

    obj = obj / n;
    return obj;
}

} // namespace faiss
