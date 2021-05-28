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

#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h> // BitstringWriter
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

// solves a system of linear equations
void sgetrs_(
        const char* trans,
        FINTEGER* n,
        FINTEGER* nrhs,
        float* A,
        FINTEGER* lda,
        FINTEGER* ipiv,
        float* b,
        FINTEGER* ldb,
        FINTEGER* info);

// general matrix multiplication
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

// c and a and b can overlap
void fvec_add(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

void fmat_inverse(float* a, int n) {
    int info;
    int lwork = n * n;
    std::vector<int> ipiv(n);
    std::vector<float> workspace(lwork);

    sgetrf_(&n, &n, a, &n, ipiv.data(), &info);
    FAISS_THROW_IF_NOT(info == 0);
    sgetri_(&n, a, &n, ipiv.data(), workspace.data(), &lwork, &info);
    FAISS_THROW_IF_NOT(info == 0);
}

void random_int32(
        std::vector<int32_t>& x,
        int32_t min,
        int32_t max,
        std::mt19937& gen) {
    std::uniform_int_distribution<int32_t> distrib(min, max);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = distrib(gen);
    }
}

} // anonymous namespace

namespace faiss {

LSQTimer lsq_timer;

LocalSearchQuantizer::LocalSearchQuantizer(size_t d, size_t M, size_t nbits) {
    FAISS_THROW_IF_NOT((M * nbits) % 8 == 0);

    this->d = d;
    this->M = M;
    this->nbits = std::vector<size_t>(M, nbits);

    // set derived values
    set_derived_values();

    is_trained = false;
    verbose = false;

    K = (1 << nbits);

    train_iters = 25;
    train_ils_iters = 8;
    icm_iters = 4;

    encode_ils_iters = 16;

    p = 0.5f;
    lambd = 1e-2f;

    chunk_size = 10000;
    nperts = 4;

    random_seed = 0x12345;
    std::srand(random_seed);
}

void LocalSearchQuantizer::train(size_t n, const float* x) {
    FAISS_THROW_IF_NOT(K == (1 << nbits[0]));
    FAISS_THROW_IF_NOT(nperts <= M);

    lsq_timer.reset();
    if (verbose) {
        lsq_timer.start("train");
        printf("Training LSQ, with %zd subcodes on %zd %zdD vectors\n",
               M,
               n,
               d);
    }

    // allocate memory for codebooks, size [M, K, d]
    codebooks.resize(M * K * d);

    // randomly intialize codes
    std::mt19937 gen(random_seed);
    std::vector<int32_t> codes(n * M); // [n, M]
    random_int32(codes, 0, K - 1, gen);

    // compute standard derivations of each dimension
    std::vector<float> stddev(d, 0);

#pragma omp parallel for
    for (int64_t i = 0; i < d; i++) {
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
        stddev[i] = sqrtf(sum / n);
    }

    if (verbose) {
        float obj = evaluate(codes.data(), x, n);
        printf("Before training: obj = %lf\n", obj);
    }

    for (size_t i = 0; i < train_iters; i++) {
        // 1. update codebooks given x and codes
        // 2. add perturbation to codebooks (SR-D)
        // 3. refine codes given x and codebooks using icm

        // update codebooks
        update_codebooks(x, codes.data(), n);

        if (verbose) {
            float obj = evaluate(codes.data(), x, n);
            printf("iter %zd:\n", i);
            printf("\tafter updating codebooks: obj = %lf\n", obj);
        }

        // SR-D: perturb codebooks
        float T = pow((1.0f - (i + 1.0f) / train_iters), p);
        perturb_codebooks(T, stddev, gen);

        if (verbose) {
            float obj = evaluate(codes.data(), x, n);
            printf("\tafter perturbing codebooks: obj = %lf\n", obj);
        }

        // refine codes
        icm_encode(x, codes.data(), n, train_ils_iters, gen);

        if (verbose) {
            float obj = evaluate(codes.data(), x, n);
            printf("\tafter updating codes: obj = %lf\n", obj);
        }
    }

    if (verbose) {
        lsq_timer.end("train");
        float obj = evaluate(codes.data(), x, n);
        printf("After training: obj = %lf\n", obj);

        printf("Time statistic:\n");
        for (const auto& it : lsq_timer.duration) {
            printf("\t%s time: %lf s\n", it.first.data(), it.second);
        }
    }

    is_trained = true;
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
    FAISS_THROW_IF_NOT_MSG(is_trained, "LSQ is not trained yet.");
    if (verbose) {
        lsq_timer.reset();
        printf("Encoding %zd vectors...\n", n);
        lsq_timer.start("encode");
    }

    std::vector<int32_t> codes(n * M);
    std::mt19937 gen(random_seed);
    random_int32(codes, 0, K - 1, gen);

    icm_encode(x, codes.data(), n, encode_ils_iters, gen);
    pack_codes(n, codes.data(), codes_out);

    if (verbose) {
        lsq_timer.end("encode");
        double t = lsq_timer.get("encode");
        printf("Time to encode %zd vectors: %lf s\n", n, t);
    }
}

/** update codebooks given x and codes
 *
 * Let B denote the sparse matrix of codes, size [n, M * K].
 * Let C denote the codebooks, size [M * K, d].
 * Let X denote the training vectors, size [n, d]
 *
 * objective function:
 *     L = (X - BC)^2
 *
 * To minimize L, we have:
 *     C = (B'B)^(-1)B'X
 * where ' denote transposed
 *
 * Add a regularization term to make B'B inversible:
 *     C = (B'B + lambd * I)^(-1)B'X
 */
void LocalSearchQuantizer::update_codebooks(
        const float* x,
        const int32_t* codes,
        size_t n) {
    lsq_timer.start("update_codebooks");

    // allocate memory
    // bb = B'B, bx = BX
    std::vector<float> bb(M * K * M * K, 0.0f); // [M * K, M * K]
    std::vector<float> bx(M * K * d, 0.0f);     // [M * K, d]

    // compute B'B
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

    // add a regularization term to B'B
    for (int64_t i = 0; i < M * K; i++) {
        bb[i * (M * K) + i] += lambd;
    }

    // compute (B'B)^(-1)
    fmat_inverse(bb.data(), M * K); // [M*K, M*K]

    // compute BX
    for (size_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            int32_t code = codes[i * M + m];
            float* data = bx.data() + (m * K + code) * d;
            fvec_add(d, data, x + i * d, data);
        }
    }

    // compute C = (B'B)^(-1) @ BX
    //
    // NOTE: LAPACK use column major order
    // out = alpha * op(A) * op(B) + beta * C
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

/** encode using iterative conditional mode
 *
 * iterative conditional mode:
 *     For every subcode ci (i = 1, ..., M) of a vector, we fix the other
 *     subcodes cj (j != i) and then find the optimal value of ci such
 *     that minimizing the objective function.

 * objective function:
 *     L = (X - \sum cj)^2, j = 1, ..., M
 *     L = X^2 - 2X * \sum cj + (\sum cj)^2
 *
 * X^2 is negligable since it is the same for all possible value
 * k of the m-th subcode.
 *
 * 2X * \sum cj is the unary term
 * (\sum cj)^2 is the binary term
 * These two terms can be precomputed and store in a look up table.
 */
void LocalSearchQuantizer::icm_encode(
        const float* x,
        int32_t* codes,
        size_t n,
        size_t ils_iters,
        std::mt19937& gen) const {
    lsq_timer.start("icm_encode");

    std::vector<float> binaries(M * M * K * K); // [M, M, K, K]
    compute_binary_terms(binaries.data());

    const size_t n_chunks = (n + chunk_size - 1) / chunk_size;
    for (size_t i = 0; i < n_chunks; i++) {
        size_t ni = std::min(chunk_size, n - i * chunk_size);

        if (verbose) {
            printf("\r\ticm encoding %zd/%zd ...", i * chunk_size + ni, n);
            fflush(stdout);
            if (i == n_chunks - 1 || i == 0) {
                printf("\n");
            }
        }

        const float* xi = x + i * chunk_size * d;
        int32_t* codesi = codes + i * chunk_size * M;
        icm_encode_partial(i, xi, codesi, ni, binaries.data(), ils_iters, gen);
    }

    lsq_timer.end("icm_encode");
}

void LocalSearchQuantizer::icm_encode_partial(
        size_t index,
        const float* x,
        int32_t* codes,
        size_t n,
        const float* binaries,
        size_t ils_iters,
        std::mt19937& gen) const {
    std::vector<float> unaries(n * M * K); // [n, M, K]
    compute_unary_terms(x, unaries.data(), n);

    std::vector<int32_t> best_codes;
    best_codes.assign(codes, codes + n * M);

    std::vector<float> best_objs(n, 0.0f);
    evaluate(codes, x, n, best_objs.data());

    FAISS_THROW_IF_NOT(nperts <= M);
    for (size_t iter1 = 0; iter1 < ils_iters; iter1++) {
        // add perturbation to codes
        perturb_codes(codes, n, gen);

        for (size_t iter2 = 0; iter2 < icm_iters; iter2++) {
            icm_encode_step(unaries.data(), binaries, codes, n);
        }

        std::vector<float> icm_objs(n, 0.0f);
        evaluate(codes, x, n, icm_objs.data());
        size_t n_betters = 0;
        float mean_obj = 0.0f;

        // select the best code for every vector xi
#pragma omp parallel for reduction(+ : n_betters, mean_obj)
        for (int64_t i = 0; i < n; i++) {
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

        if (verbose && index == 0) {
            printf("\tils_iter %zd: obj = %lf, n_betters/n = %zd/%zd\n",
                   iter1,
                   mean_obj,
                   n_betters,
                   n);
        }
    } // loop ils_iters
}

void LocalSearchQuantizer::icm_encode_step(
        const float* unaries,
        const float* binaries,
        int32_t* codes,
        size_t n) const {
    // condition on the m-th subcode
    for (size_t m = 0; m < M; m++) {
        std::vector<float> objs(n * K);
#pragma omp parallel for
        for (int64_t i = 0; i < n; i++) {
            auto u = unaries + i * (M * K) + m * K;
            memcpy(objs.data() + i * K, u, sizeof(float) * K);
        }

        // compute objective function by adding unary
        // and binary terms together
        for (size_t other_m = 0; other_m < M; other_m++) {
            if (other_m == m) {
                continue;
            }

#pragma omp parallel for
            for (int64_t i = 0; i < n; i++) {
                for (int32_t code = 0; code < K; code++) {
                    int32_t code2 = codes[i * M + other_m];
                    size_t binary_idx =
                            m * M * K * K + other_m * K * K + code * K + code2;
                    // binaries[m, other_m, code, code2]
                    objs[i * K + code] += binaries[binary_idx];
                }
            }
        }

        // find the optimal value of the m-th subcode
#pragma omp parallel for
        for (int64_t i = 0; i < n; i++) {
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
}

void LocalSearchQuantizer::perturb_codes(
        int32_t* codes,
        size_t n,
        std::mt19937& gen) const {
    lsq_timer.start("perturb_codes");

    std::uniform_int_distribution<size_t> m_distrib(0, M - 1);
    std::uniform_int_distribution<int32_t> k_distrib(0, K - 1);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < nperts; j++) {
            size_t m = m_distrib(gen);
            codes[i * M + m] = k_distrib(gen);
        }
    }

    lsq_timer.end("perturb_codes");
}

void LocalSearchQuantizer::compute_binary_terms(float* binaries) const {
    lsq_timer.start("compute_binary_terms");

#pragma omp parallel for
    for (int64_t m12 = 0; m12 < M * M; m12++) {
        size_t m1 = m12 / M;
        size_t m2 = m12 % M;

        for (size_t code1 = 0; code1 < K; code1++) {
            for (size_t code2 = 0; code2 < K; code2++) {
                const float* c1 = codebooks.data() + m1 * K * d + code1 * d;
                const float* c2 = codebooks.data() + m2 * K * d + code2 * d;
                float ip = fvec_inner_product(c1, c2, d);
                // binaries[m1, m2, code1, code2] = ip * 2
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

    // compute x * codebooks^T
    //
    // NOTE: LAPACK use column major order
    // out = alpha * op(A) * op(B) + beta * C
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
    for (int64_t i = 0; i < n; i++) {
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
    for (int64_t i = 0; i < n; i++) {
        const auto code = codes + i * M;
        const auto decoded_i = decoded_x.data() + i * d;
        for (size_t m = 0; m < M; m++) {
            // c = codebooks[m, code[m]]
            const auto c = codebooks.data() + m * K * d + code[m] * d;
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

double LSQTimer::get(const std::string& name) {
    return duration[name];
}

void LSQTimer::start(const std::string& name) {
    FAISS_THROW_IF_NOT_MSG(!started[name], " timer is already running");
    started[name] = true;
    t0[name] = getmillisecs();
}

void LSQTimer::end(const std::string& name) {
    FAISS_THROW_IF_NOT_MSG(started[name], " timer is not running");
    double t1 = getmillisecs();
    double sec = (t1 - t0[name]) / 1000;
    duration[name] += sec;
    started[name] = false;
}

void LSQTimer::reset() {
    duration.clear();
    t0.clear();
    started.clear();
}

} // namespace faiss
