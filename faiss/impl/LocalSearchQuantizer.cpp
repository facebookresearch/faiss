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

LSQTimer lsq_timer;

LocalSearchQuantizer::LocalSearchQuantizer(size_t d, size_t M, size_t nbits) {
    this->d = d;
    this->M = M;
    this->nbits = nbits;

    log_level = 0;

    K = (1 << nbits);
    code_size = M * (nbits / 8); // in bytes

    train_iters = 25;
    train_ils_iters = 8;
    icm_iters = 4;

    encode_ils_iters = 16;

    p = 0.5f;
    lambda = 0.0001f;

    chunk_size = 10000;
    nperts = 4;
}

void LocalSearchQuantizer::train(size_t n, const float* x) {
    lsq_timer.reset();
    if (log_level > 0) {
        lsq_timer.start("train");
        printf("Training LSQ++, with %zd subcodes on %zd %zdD vectors\n",
               M,
               n,
               d);
    }

    K = (1 << nbits); // reset it

    codebooks.resize(M * K * d);       // [M, K, d]
    std::vector<int32_t> codes(n * M); // [n, M]

    std::mt19937 gen(12345);
    random_int32(&codes, 0, K - 1, gen);

    // SR-D
    std::vector<float> stddev(d, 0);

#pragma omp parallel for
    for (size_t i = 0; i < d; i++) {
        float mean = 0;
        for (size_t j = 0; j < n; j++) {
            mean += x[j * d + i];
        }
        mean = mean / n;

        float sum = 0;
        for (size_t j = 0; j < n; j++) {
            float xi = x[j * d + i] - mean;
            sum += xi * xi;
        }
        stddev[i] = sqrt(sum / n);
    }

    if (log_level > 0) {
        float obj = evaluate(codes.data(), x, n);
        printf("Before training: obj = %lf\n", obj);
    }

    for (size_t i = 0; i < train_iters; i++) {
        update_codebooks(x, codes.data(), n);

        if (log_level > 0) {
            float obj = evaluate(codes.data(), x, n);
            printf("iter %zd:\n", i);
            printf("\tafter updating codebooks: obj = %lf\n", obj);
        }

        // SR-D: perturb codebooks
        float T = pow((1.0f - (i + 1.0f) / train_iters), p);
        perturb_codebooks(T, stddev, gen);

        if (log_level > 0) {
            float obj = evaluate(codes.data(), x, n);
            printf("\tafter perturbing codebooks: obj = %lf\n", obj);
        }

        icm_encode(x, codes.data(), n, train_ils_iters);

        if (log_level > 0) {
            float obj = evaluate(codes.data(), x, n);
            printf("\tafter updating codes: obj = %lf\n", obj);
        }
    }

    if (log_level > 0) {
        lsq_timer.end("train");
        float obj = evaluate(codes.data(), x, n);
        printf("After training: obj = %lf\n", obj);

        printf("Time statistic:\n");
        for (const auto& it : lsq_timer.duration) {
            printf("\t%s time: %lf s\n", it.first.data(), it.second);
        }
    }
}

void LocalSearchQuantizer::perturb_codebooks(
        float T,
        const std::vector<float>& stddev,
        std::mt19937& gen) {
    lsq_timer.start("perturb_codebooks");

    std::vector<std::normal_distribution<float>> distribs;
    for (size_t i = 0; i < d; i++) {
        distribs.emplace_back(0.0f, stddev[i]);
    }

    for (size_t m = 0; m < M; m++) {
        for (size_t k = 0; k < K; k++) {
            for (size_t i = 0; i < d; i++) {
                codebooks[m * K * d + k * d + i] += T * distribs[i](gen) / M;
            }
        }
    }

    lsq_timer.end("perturb_codebooks");
}

void LocalSearchQuantizer::compute_codes(
        const float* x,
        uint8_t* codes_out,
        size_t n) const {
    if (log_level > 0) {
        lsq_timer.reset();
        printf("Encoding %zd vectors...\n", n);
        lsq_timer.start("encode");
    }

    std::vector<int32_t> codes(n * M);
    std::mt19937 gen(1234);
    random_int32(&codes, 0, K - 1, gen);

    icm_encode(x, codes.data(), n, encode_ils_iters);
    pack_codes(n, codes.data(), codes_out);

    if (log_level > 0) {
        lsq_timer.end("encode");
        double t = lsq_timer.get("encode");
        printf("Time to encode %zd vectors: %lf s\n", n, t);
    }
}

void LocalSearchQuantizer::pack_codes(
        size_t n,
        const int32_t* codes,
        uint8_t* packed_codes) const {
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codes1 = codes + i * M;
        BitstringWriter bsw(packed_codes + i * code_size, code_size);
        for (int m = 0; m < M; m++) {
            bsw.write(codes1[m], nbits);
        }
    }
}

void LocalSearchQuantizer::decode(const uint8_t* codes, float* x, size_t n)
        const {
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        BitstringReader bsr(codes + i * code_size, code_size);
        float* xi = x + i * d;
        for (int m = 0; m < M; m++) {
            int code = bsr.read(nbits);
            const float* c = codebooks.data() + (m * K + code) * d;
            if (m == 0) {
                memcpy(xi, c, sizeof(float) * d);
            } else {
                fvec_add(d, xi, c, xi);
            }
        }
    }
}

/**
 *
    Input:
        x: [n, d]
        codebooks: [m, K, d]
        b: [n, m]
    Formulation:
        B = sparse(b): [n, m*K]
        C: [m*K, d]
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
    lsq_timer.start("update_codebooks");

    // allocate memory
    std::vector<float> bb(M * K * M * K, 0.0f); // [M * K, M * K]
    std::vector<float> bx(M * K * d, 0.0f);     // [M * K, d]

    // compute B'B
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            int32_t code1 = codes[i * M + m];
            int32_t idx1 = m * K + code1;
            bb[idx1 * M * K + idx1] += 1;

            for (size_t m2 = m + 1; m2 < M; m2++) {
                int32_t code2 = codes[i * M + m2];
                int32_t idx2 = m2 * K + code2;
                bb[idx1 * M * K + idx2] += 1;
                bb[idx2 * M * K + idx1] += 1;
            }
        }
    }

    // compute BX
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            int32_t code = codes[i * M + m];
            float* data = bx.data() + (m * K + code) * d;
            fvec_add(d, data, x + i * d, data);
        }
    }

    // add a regularization term to B'B, make it inversible
#pragma omp parallel for
    for (size_t i = 0; i < M * K; i++) {
        bb[i * (M * K) + i] += lambda;
    }

    // C = inv(bb) @ bx

    // compute inv(bb)
    fmat_inverse(bb.data(), M * K); // [M*K, M*K]

    // compute inv(bb) @ bx

    // NOTE: LAPACK use column major order
    //
    // out = bb @ bx
    // out^T = bx^T @ bb^T
    //         [d, M*K] @ [M*K, M*K]
    //
    // out = alpha * op(A) * op(B) + beta * C
    // [M*K, M*K] @ [M*K, d]
    FINTEGER nrows_A = d;
    FINTEGER ncols_A = M * K;

    FINTEGER nrows_B = M * K;
    FINTEGER ncols_B = M * K;

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

    lsq_timer.end("update_codebooks");
}

void LocalSearchQuantizer::icm_encode(
        const float* x,
        int32_t* codes,
        size_t n,
        size_t ils_iters) const {
    lsq_timer.start("icm_encode");

    std::vector<float> binaries(M * M * K * K); // [M, M, K, K]
    compute_binary_terms(binaries.data());

    const size_t n_chunks = (n + chunk_size - 1) / chunk_size;
    for (size_t i = 0; i < n_chunks; i++) {
        if (log_level == 2) {
            printf("\r\ticm encoding %zd/%zd ...", i + 1, n_chunks);
            fflush(stdout);
            if (i == n_chunks - 1 or i == 0) {
                printf("\n");
            }
        }

        size_t ni = std::min(chunk_size, n - i * chunk_size);
        const float* xi = x + i * chunk_size * d;
        int32_t* codesi = codes + i * chunk_size * M;
        icm_encode_partial(i, xi, codesi, ni, binaries.data(), ils_iters);
    }

    lsq_timer.end("icm_encode");
}

void LocalSearchQuantizer::icm_encode_partial(
        size_t index,
        const float* x,
        int32_t* codes,
        size_t n,
        const float* binaries,
        size_t ils_iters) const {
    std::vector<float> unaries(n * M * K); // [n, M, K]
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
                std::vector<float> objs(n * K);
#pragma omp parallel for
                for (size_t i = 0; i < n; i++) {
                    auto u = unaries.data() + i * (M * K) + m * K;
                    memcpy(objs.data() + i * K, u, sizeof(float) * K);
                }

                for (size_t other_m = 0; other_m < M; other_m++) {
                    if (other_m == m) {
                        continue;
                    }

#pragma omp parallel for
                    for (size_t i = 0; i < n; i++) {
                        for (int32_t code = 0; code < K; code++) {
                            int32_t code2 = codes[i * M + other_m];
                            size_t binary_idx = m * M * K * K +
                                    other_m * K * K + code * K + code2;
                            objs[i * K + code] +=
                                    binaries[binary_idx]; // binaries[m,
                                                          // other_m, code,
                                                          // code2]
                        }
                    }
                }

#pragma omp parallel for
                for (size_t i = 0; i < n; i++) {
                    float best_obj = HUGE_VALF;
                    int32_t best_code = 0;
                    for (size_t code = 0; code < K; code++) {
                        float obj = objs[i * K + code];
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

        if (log_level == 2 && index == 0) {
            printf("\tils_iter %zd: obj = %lf, n_betters/n = %zd/%zd\n",
                   iter1,
                   mean_obj,
                   n_betters,
                   n);
        }
    } // loop ils_iters
}

void LocalSearchQuantizer::perturb_codes(int32_t* codes, size_t n) const {
    lsq_timer.start("perturb_codes");

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < nperts; j++) {
            /// TODO: replace rand() by uniform_int_distribution
            size_t m = rand() % M;
            codes[i * M + m] = rand() % K;
        }
    }

    lsq_timer.end("perturb_codes");
}

void LocalSearchQuantizer::compute_binary_terms(float* binaries) const {
    lsq_timer.start("compute_binary_terms");

#pragma omp parallel for
    for (size_t m12 = 0; m12 < M * M; m12++) {
        size_t m1 = m12 / M;
        size_t m2 = m12 % M;

        for (size_t code1 = 0; code1 < K; code1++) {
            for (size_t code2 = 0; code2 < K; code2++) {
                const float* c1 = codebooks.data() + m1 * K * d + code1 * d;
                const float* c2 = codebooks.data() + m2 * K * d + code2 * d;
                float ip = fvec_inner_product(c1, c2, d);
                binaries[m1 * M * K * K + m2 * K * K + code1 * K + code2] =
                        ip * 2;
            }
        }
    }

    lsq_timer.end("compute_binary_terms");
}

void LocalSearchQuantizer::compute_unary_terms(
        const float* x,
        float* unaries,
        size_t n) const {
    lsq_timer.start("compute_unary_terms");

    // NOTE: LAPACK use column major order
    //
    // out = x @ codebooks^T
    // out^T = codebooks @ x^T
    //         [m*K, d] @ [d, n]

    // out = alpha * op(A) * op(B) + beta * C
    // [M*K, M*K] @ [M*K, d]
    FINTEGER nrows_A = M * K;
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

    std::vector<float> norms(M * K);
    fvec_norms_L2sqr(norms.data(), codebooks.data(), d, M * K);

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float* u = unaries + i * (M * K);
        fvec_add(M * K, u, norms.data(), u);
    }

    lsq_timer.end("compute_unary_terms");
}

float LocalSearchQuantizer::evaluate(
        const int32_t* codes,
        const float* x,
        size_t n,
        float* objs) const {
    lsq_timer.start("evaluate");

    // decode
    std::vector<float> decoded_x(n * d, 0.0f);
    float obj = 0.0f;

#pragma omp parallel for reduction(+ : obj)
    for (size_t i = 0; i < n; i++) {
        const auto code = codes + i * M;
        const auto decoded_i = decoded_x.data() + i * d;
        for (size_t m = 0; m < M; m++) {
            const auto c = codebooks.data() + m * K * d +
                    code[m] * d; // codebooks[m, code[m]]
            fvec_add(d, decoded_i, c, decoded_i);
        }

        float err = fvec_L2sqr(x + i * d, decoded_i, d);
        obj += err;

        if (objs) {
            objs[i] = err;
        }
    }

    lsq_timer.end("evaluate");

    obj = obj / n;
    return obj;
}

} // namespace faiss
