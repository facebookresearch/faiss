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
void sgetrf_(
        FINTEGER* m,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        FINTEGER* ipiv,
        FINTEGER* info);

// generate inverse of a matrix given its LU decomposition
void sgetri_(
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        FINTEGER* ipiv,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

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
    int* ipiv = new int[n];
    int lwork = n * n;
    float* workspace = new float[lwork];
    int info;

    sgetrf_(&n, &n, a, &n, ipiv, &info);
    FAISS_THROW_IF_NOT(info == 0);
    sgetri_(&n, a, &n, ipiv, workspace, &lwork, &info);
    FAISS_THROW_IF_NOT(info == 0);

    delete[] ipiv;
    delete[] workspace;
}

// void fmat_mutiply()

void random_int32(
        std::vector<int32_t>* x,
        int32_t min,
        int32_t max,
        std::mt19937& gen) {
    std::uniform_int_distribution<int32_t> distrib(min, max);
    for (size_t i = 0; i < x->size(); i++) {
        (*x)[i] = distrib(gen);
    }
}

void random_float(
        std::vector<float>* x,
        float min,
        float max,
        std::mt19937& gen) {
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

    nperts = 4;
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
        printf("iter %zd, updated obj: %lf\n", i, obj);

        icm_encode(x, codes.data(), n);
    }

    obj = evaluate(codes.data(), x, n);
    printf("Training obj: %lf\n", obj);
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

void LocalSearchQuantizer::decode(const uint8_t* code, float* x, size_t n)
        const {
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
void LocalSearchQuantizer::update_codebooks(
        const float* x,
        const int32_t* codes,
        size_t n) {
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

    // compute BX
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            int32_t code = codes[i * M + m];
            float* data = bx.data() + (m * h + code) * d;
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
    fmat_inverse(bb.data(), M * h); // [M*h, M*h]

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
           &nrows_A, // nrows of op(A)
           &ncols_B, // ncols of op(B)
           &ncols_A, // ncols of op(A)
           &alpha,
           bx.data(),
           &nrows_A, // nrows of A
           bb.data(),
           &nrows_B, // nrows of B
           &beta,
           codebooks.data(),
           &nrows_A); // nrows of output
}

void LocalSearchQuantizer::icm_encode(
        const float* x,
        int32_t* codes,
        size_t n) {
    size_t h = (1 << nbits);

    std::vector<float> binaries(M * M * h * h); // [M, M, h, h]
    compute_binary_terms(binaries.data());

    std::vector<float> unaries(n * M * h); // [n, M, h]
    compute_unary_terms(x, unaries.data(), n);

    std::vector<int32_t> best_codes;
    best_codes.assign(codes, codes + n * M);

    std::vector<float> best_objs(n, 0.0f);
    evaluate(codes, x, n, best_objs.data());

    FAISS_THROW_IF_NOT(nperts <= M);
    for (size_t iter1 = 0; iter1 < ils_iters; iter1++) {
        perturb_codes(codes, n);

        for (size_t iter2 = 0; iter2 < icm_iters; iter2++) {
            for (size_t m = 0; m < M; m++) {
                std::vector<float> objs(n * h);
#pragma omp parallel for
                for (size_t i = 0; i < n; i++) {
                    auto u = unaries.data() + i * (M * h) + m * h;
                    memcpy(objs.data() + i * h, u, sizeof(float) * h);
                }

                for (size_t other_m = 0; other_m < M; other_m++) {
                    if (other_m == m) {
                        continue;
                    }

#pragma omp parallel for
                    for (size_t i = 0; i < n; i++) {
                        for (int32_t code = 0; code < h; code++) {
                            int32_t code2 = codes[i * M + other_m];
                            size_t binary_idx = m * M * h * h +
                                    other_m * h * h + code * h + code2;
                            objs[i * h + code] +=
                                    binaries[binary_idx]; // binaries[m,
                                                          // other_m, code,
                                                          // code2]
                        }
                    }
                }

#pragma omp parallel for
                for (size_t i = 0; i < n; i++) {
                    float best_obj = 9999999.0f;
                    int32_t best_code = 0;
                    for (size_t code = 0; code < h; code++) {
                        float obj = objs[i * h + code];
                        if (obj < best_obj) {
                            best_obj = obj;
                            best_code = code;
                        }
                    }
                    codes[i * M + m] = best_code;
                }

            } // loop M
        }     // loop icm_iters

        std::vector<float> icm_objs(n, 0.0f);
        evaluate(codes, x, n, icm_objs.data());
        size_t n_betters = 0;
        float mean_obj = 0.0f;

#pragma omp parallel for reduction(+ : n_betters, mean_obj)
        for (size_t i = 0; i < n; i++) {
            if (icm_objs[i] < best_objs[i]) {
                best_objs[i] = icm_objs[i];
                memcpy(best_codes.data() + i * M,
                       codes + i * M,
                       sizeof(int32_t) * M);
                n_betters += 1;
            }
            mean_obj += best_objs[i];
        }
        mean_obj /= n;

        memcpy(codes, best_codes.data(), sizeof(int32_t) * n * M);

        printf("\tils_iter %zd: obj = %lf, n_betters = %zd\n",
               iter1,
               mean_obj,
               n_betters);
    } // loop ils_iters
}

void LocalSearchQuantizer::perturb_codes(int32_t* codes, size_t n) {
    size_t h = (1 << nbits);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < nperts; j++) {
            /// TODO: replace rand() by uniform_int_distribution
            size_t m = rand() % M;
            codes[i * M + m] = rand() % h;
        }
    }
}

void LocalSearchQuantizer::compute_binary_terms(float* binaries) {
    size_t h = (1 << nbits);

#pragma omp parallel for
    for (size_t m12 = 0; m12 < M * M; m12++) {
        size_t m1 = m12 / M;
        size_t m2 = m12 % M;

        for (size_t code1 = 0; code1 < h; code1++) {
            for (size_t code2 = 0; code2 < h; code2++) {
                float* c1 = codebooks.data() + m1 * h * d + code1 * d;
                float* c2 = codebooks.data() + m2 * h * d + code2 * d;
                float ip = fvec_inner_product(c1, c2, d);
                binaries[m1 * M * h * h + m2 * h * h + code1 * h + code2] =
                        ip * 2;
            }
        }
    }
}

void LocalSearchQuantizer::compute_unary_terms(
        const float* x,
        float* unaries,
        size_t n) {
    size_t h = (1 << nbits);

    // NOTE: LAPACK use column major order
    //
    // out = x @ codebooks^T
    // out^T = codebooks @ x^T
    //         [m*h, d] @ [d, n]

    // out = alpha * op(A) * op(B) + beta * C
    // [M*h, M*h] @ [M*h, d]
    FINTEGER nrows_A = M * h;
    FINTEGER ncols_A = d;

    FINTEGER nrows_B = d;
    FINTEGER ncols_B = n;

    float alpha = -2.0f;
    float beta = 0.0f;
    sgemm_("Transposed",
           "Not Transposed",
           &nrows_A, // nrows of op(A)
           &ncols_B, // ncols of op(B)
           &ncols_A, // ncols of op(A)
           &alpha,
           codebooks.data(),
           &ncols_A, // nrows of A
           x,
           &nrows_B, // nrows of B
           &beta,
           unaries,
           &nrows_A); // nrows of output

    std::vector<float> norms(M * h);
    fvec_norms_L2sqr(norms.data(), codebooks.data(), d, M * h);

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float* u = unaries + i * (M * h);
        fvec_add(M * h, u, norms.data(), u);
    }
}

float LocalSearchQuantizer::evaluate(
        const int32_t* codes,
        const float* x,
        size_t n,
        float* objs) const {
    // decode
    size_t h = (1 << nbits);
    std::vector<float> decoded_x(n * d, 0.0f);
    float obj = 0.0f;

    double t0 = getmillisecs();

#pragma omp parallel for reduction(+ : obj)
    for (size_t i = 0; i < n; i++) {
        const auto code = codes + i * M;
        const auto decoded_i = decoded_x.data() + i * d;
        for (size_t m = 0; m < M; m++) {
            const auto c = codebooks.data() + m * h * d +
                    code[m] * d; // codebooks[m, code[m]]
            fvec_add(d, decoded_i, c, decoded_i);
        }

        float err = fvec_L2sqr(x + i * d, decoded_i, d);
        obj += err;

        if (objs) {
            objs[i] = err;
        }
    }

    double t1 = getmillisecs();
    printf("\t\tevaluate time: %lf\n", (t1 - t0) / 1000);

    obj = obj / n;
    return obj;
}

} // namespace faiss
